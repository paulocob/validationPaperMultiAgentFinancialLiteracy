"""
LoRA Fine-tuning do OpenCLIP ViT-B-32 para portugues financeiro lusofono.

Ajusta o encoder de texto do CLIP usando pares (pergunta, resposta) e
(texto, descricao_visual) do FinEduBench-Luso com contrastive loss.
Usa LoRA (Low-Rank Adaptation) para eficiencia de parametros.

Uso:
    python -m embeddings.clip_lora_finetune
    python -m embeddings.clip_lora_finetune --epochs 30 --rank 8
    python -m embeddings.clip_lora_finetune --smoke  # teste rapido (2 epochs)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import QAPair, load_all_countries

console = Console()


# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning do CLIP para portugues financeiro"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank (default: 4)")
    parser.add_argument("--alpha", type=float, default=8.0, help="LoRA alpha (default: 8.0)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--smoke", action="store_true", help="Teste rapido (2 epochs)")
    parser.add_argument(
        "--cross-lingual", action="store_true",
        help="Incluir pares PT<->EN para ancoragem cross-lingual",
    )
    parser.add_argument(
        "--cl-weight", type=float, default=0.3,
        help="Peso da loss cross-lingual (default: 0.3)",
    )
    parser.add_argument(
        "--num-anchors", type=int, default=0,
        help="Numero de anchors cross-lingual (0 = todos, default: 0)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).parent / "lora_output"),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CLIPFineTuneDataset(Dataset):
    """
    Dataset para fine-tuning contrastivo do CLIP.

    Cada item retorna um par (texto_a, texto_b) onde:
    - text-text: (pergunta, resposta)
    - text-image: (pergunta, descricao_visual)

    Hard negatives sao gerados on-the-fly pelo batch (in-batch negatives).
    """

    def __init__(self, pairs: list[QAPair], include_image_pairs: bool = True):
        self.items: list[tuple[str, str]] = []

        for qa in pairs:
            # Par text-text: pergunta <-> resposta
            self.items.append((qa.question, qa.answer))

            # Par text-image: pergunta <-> descricao visual
            if include_image_pairs and qa.image_description:
                self.items.append((qa.question, qa.image_description))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.items[idx]


class CrossLingualDataset(Dataset):
    """
    Dataset de pares PT<->EN para ancoragem cross-lingual.

    Cada item retorna (texto_pt, texto_en) que devem ser alinhados
    no espaco de embeddings. Inclui:
    - pergunta_pt <-> pergunta_en
    - resposta_pt <-> resposta_en
    """

    def __init__(self, anchors: list[dict]):
        self.items: list[tuple[str, str]] = []

        for a in anchors:
            # Alinhar perguntas PT<->EN
            self.items.append((a["pt_question"], a["en_question"]))
            # Alinhar respostas PT<->EN
            self.items.append((a["pt_answer"], a["en_answer"]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.items[idx]


# ---------------------------------------------------------------------------
# LoRA Layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Adaptador LoRA para camada Linear."""

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Congela pesos originais
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # Adaptadores LoRA
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)

        # Inicializacao Kaiming para A, zero para B
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self) -> torch.Tensor:
        """Expoe weight do layer original (necessario para OpenCLIP internals)."""
        return self.original.weight

    @property
    def bias(self) -> torch.Tensor | None:
        """Expoe bias do layer original."""
        return self.original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_out * self.scaling


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    target_modules: list[str] | None = None,
) -> int:
    """
    Aplica adaptadores LoRA ao encoder de texto do CLIP.

    Retorna o numero de parametros treinaveis adicionados.
    """
    if target_modules is None:
        # Alvo: projecoes Q, K, V e out nos attention layers do texto
        target_modules = ["out_proj", "in_proj_weight"]

    total_lora_params = 0
    replaced = 0

    # Iterar sobre o transformer de texto
    text_encoder = model.transformer if hasattr(model, "transformer") else model.text

    for name, module in text_encoder.named_modules():
        if isinstance(module, nn.Linear):
            # Verificar se o nome bate com algum target
            if any(t in name for t in target_modules):
                # Encontrar o modulo pai e o nome do atributo
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = dict(text_encoder.named_modules())[parent_name]
                else:
                    parent = text_encoder
                    attr_name = name

                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, attr_name, lora_layer)

                n_params = lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
                total_lora_params += n_params
                replaced += 1

    return total_lora_params


def apply_lora_simple(model: nn.Module, rank: int = 4, alpha: float = 8.0) -> int:
    """
    Aplica LoRA de forma simplificada: congela tudo e adiciona adaptadores
    nas camadas de projecao do text encoder.
    """
    # Congela todos os parametros
    for param in model.parameters():
        param.requires_grad = False

    total_params = 0

    # Encontrar camadas lineares no text transformer
    text_module = None
    for attr in ["transformer", "text"]:
        if hasattr(model, attr):
            text_module = getattr(model, attr)
            break

    if text_module is None:
        raise ValueError("Nao encontrou text encoder no modelo CLIP")

    # Aplicar LoRA em todas as camadas de atencao (q, k, v, out)
    for name, module in list(text_module.named_modules()):
        if isinstance(module, nn.Linear) and any(
            kw in name for kw in ["attn", "mlp", "proj"]
        ):
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(text_module.named_modules())[parent_name]
            else:
                parent = text_module
                attr_name = name

            lora = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, attr_name, lora)
            n = lora.lora_A.numel() + lora.lora_B.numel()
            total_params += n

    # Tambem descongelar o text_projection se existir
    if hasattr(model, "text_projection") and model.text_projection is not None:
        if isinstance(model.text_projection, nn.Parameter):
            model.text_projection.requires_grad = True
            total_params += model.text_projection.numel()

    return total_params


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CLIPContrastiveLoss(nn.Module):
    """
    InfoNCE / CLIP contrastive loss simetrica.

    Dentro de um batch de N pares (a_i, b_i), os positivos sao (a_i, b_i)
    e todos os outros (a_i, b_j) com j != i sao negativos (in-batch).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor
    ) -> torch.Tensor:
        # Normaliza
        a = F.normalize(embeddings_a, dim=-1)
        b = F.normalize(embeddings_b, dim=-1)

        # Similaridade coseno: (N, N)
        logits = (a @ b.T) / self.temperature

        # Labels: diagonal (i-esimo par e o positivo)
        labels = torch.arange(len(logits), device=logits.device)

        # Loss simetrica
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)

        return (loss_a + loss_b) / 2


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

def encode_batch(
    texts: list[str], model, tokenizer, device: str
) -> torch.Tensor:
    """Codifica batch de textos e retorna embeddings."""
    tokens = tokenizer(texts).to(device)
    features = model.encode_text(tokens)
    return features


def train_epoch(
    model,
    tokenizer,
    dataloader: DataLoader,
    optimizer,
    criterion,
    device: str,
    cl_loader: DataLoader | None = None,
    cl_weight: float = 0.3,
) -> float:
    """Treina por uma epoch e retorna loss media."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    cl_iter = iter(cl_loader) if cl_loader else None

    for texts_a, texts_b in dataloader:
        optimizer.zero_grad()

        emb_a = encode_batch(list(texts_a), model, tokenizer, device)
        emb_b = encode_batch(list(texts_b), model, tokenizer, device)

        loss = criterion(emb_a, emb_b)

        # Cross-lingual alignment loss
        if cl_iter is not None:
            try:
                cl_pt, cl_en = next(cl_iter)
            except StopIteration:
                cl_iter = iter(cl_loader)
                cl_pt, cl_en = next(cl_iter)

            emb_pt = encode_batch(list(cl_pt), model, tokenizer, device)
            emb_en = encode_batch(list(cl_en), model, tokenizer, device)

            # MSE entre embeddings normalizados (alinha PT com EN)
            pt_norm = F.normalize(emb_pt, dim=-1)
            en_norm = F.normalize(emb_en, dim=-1)
            cl_loss = 1.0 - (pt_norm * en_norm).sum(dim=-1).mean()

            loss = (1.0 - cl_weight) * loss + cl_weight * cl_loss

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(
    model, tokenizer, dataloader: DataLoader, criterion, device: str
) -> tuple[float, float]:
    """Avalia e retorna (loss, similaridade media dos positivos)."""
    model.eval()
    total_loss = 0.0
    total_sim = 0.0
    n_batches = 0

    for texts_a, texts_b in dataloader:
        emb_a = encode_batch(list(texts_a), model, tokenizer, device)
        emb_b = encode_batch(list(texts_b), model, tokenizer, device)

        loss = criterion(emb_a, emb_b)

        # Similaridade dos positivos (diagonal)
        a_norm = F.normalize(emb_a, dim=-1)
        b_norm = F.normalize(emb_b, dim=-1)
        pos_sim = (a_norm * b_norm).sum(dim=-1).mean()

        total_loss += loss.item()
        total_sim += pos_sim.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_sim = total_sim / max(n_batches, 1)
    return avg_loss, avg_sim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.smoke:
        args.epochs = 2
        args.batch_size = 16
        console.print("[bold yellow]Modo smoke test (2 epochs)[/bold yellow]")

    # Reproducibilidade
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    console.print(f"[bold]Device:[/bold] {device}")

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Carregar dados ---
    console.print("\n[bold]1. Carregando dataset...[/bold]")
    dataset_dir = _PROJECT_ROOT / "dataset"
    all_pairs = load_all_countries(dataset_dir)
    console.print(f"   {len(all_pairs)} QA pairs carregados")

    # Shuffle e split
    random.shuffle(all_pairs)
    n_val = int(len(all_pairs) * args.val_split)
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    train_dataset = CLIPFineTuneDataset(train_pairs)
    val_dataset = CLIPFineTuneDataset(val_pairs)

    console.print(f"   Train: {len(train_dataset)} pares | Val: {len(val_dataset)} pares")

    # Cross-lingual anchors
    cl_loader = None
    if args.cross_lingual:
        try:
            from dataset.expansions.en_anchors import EN_ANCHORS
            anchors = EN_ANCHORS
            if args.num_anchors > 0 and args.num_anchors < len(anchors):
                random.seed(args.seed)
                anchors = random.sample(anchors, args.num_anchors)
            cl_dataset = CrossLingualDataset(anchors)
            cl_loader = DataLoader(
                cl_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
            )
            console.print(
                f"   Cross-lingual: {len(cl_dataset)} pares PT<->EN "
                f"(weight={args.cl_weight})"
            )
        except ImportError:
            console.print("   [yellow]AVISO: en_anchors.py nao encontrado, sem cross-lingual[/yellow]")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # --- Carregar CLIP ---
    console.print("\n[bold]2. Carregando OpenCLIP ViT-B-32...[/bold]")
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.float()  # float32 para compatibilidade MPS

    # --- Aplicar LoRA (antes de mover para device) ---
    console.print(f"\n[bold]3. Aplicando LoRA (rank={args.rank}, alpha={args.alpha})...[/bold]")
    n_lora_params = apply_lora_simple(model, rank=args.rank, alpha=args.alpha)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    console.print(f"   Parametros totais:     {total_params:,}")
    console.print(f"   Parametros treinaveis: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    console.print(f"   Parametros LoRA:       {n_lora_params:,}")

    # --- Baseline pre-LoRA ---
    console.print("\n[bold]4. Avaliando baseline (pre-fine-tuning)...[/bold]")
    criterion = CLIPContrastiveLoss(temperature=args.temperature)
    baseline_loss, baseline_sim = eval_epoch(model, tokenizer, val_loader, criterion, device)
    console.print(f"   Baseline val loss: {baseline_loss:.4f} | sim positivos: {baseline_sim:.4f}")

    # --- Treinar ---
    console.print(f"\n[bold]5. Treinando ({args.epochs} epochs)...[/bold]")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, tokenizer, train_loader, optimizer, criterion, device,
            cl_loader=cl_loader, cl_weight=args.cl_weight,
        )
        val_loss, val_sim = eval_epoch(model, tokenizer, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_sim": val_sim,
            "lr": lr,
        })

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Salvar melhor modelo (apenas LoRA weights)
            save_lora_weights(model, output_dir / "best_lora.pt")
            marker = " *best*"
        else:
            patience_counter += 1

        console.print(
            f"  Ep {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"sim={val_sim:.4f} | lr={lr:.2e} | "
            f"{elapsed:.1f}s{marker}"
        )

        if patience_counter >= args.patience:
            console.print(f"  [yellow]Early stopping (patience={args.patience})[/yellow]")
            break

    total_time = time.time() - start_time
    console.print(f"\n  Treinamento concluido em {total_time:.1f}s")

    # --- Salvar ---
    save_lora_weights(model, output_dir / "final_lora.pt")

    # Salvar historico
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Salvar config
    config = {
        "model": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "rank": args.rank,
        "alpha": args.alpha,
        "epochs_trained": len(history),
        "best_val_loss": best_val_loss,
        "baseline_val_loss": baseline_loss,
        "baseline_val_sim": baseline_sim,
        "final_val_sim": history[-1]["val_sim"],
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "temperature": args.temperature,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "cross_lingual": args.cross_lingual,
        "cl_weight": args.cl_weight,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Avaliacao final ---
    console.print("\n[bold]6. Avaliacao final...[/bold]")

    # Carregar melhor modelo
    load_lora_weights(model, output_dir / "best_lora.pt")
    final_loss, final_sim = eval_epoch(model, tokenizer, val_loader, criterion, device)

    # Tabela comparativa
    table = Table(title="Comparacao Pre vs Pos Fine-tuning")
    table.add_column("Metrica", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Fine-tuned", justify="right")
    table.add_column("Delta", justify="right")

    table.add_row(
        "Val Loss",
        f"{baseline_loss:.4f}",
        f"{final_loss:.4f}",
        f"{final_loss - baseline_loss:+.4f}",
    )
    table.add_row(
        "Sim Positivos",
        f"{baseline_sim:.4f}",
        f"{final_sim:.4f}",
        f"{final_sim - baseline_sim:+.4f}",
    )
    console.print(table)

    # --- Plot ---
    plot_training_history(history, output_dir / "training_curves.png")

    console.print(f"\n[bold]Artefatos salvos em:[/bold] {output_dir}")
    console.print(f"  best_lora.pt       — pesos LoRA do melhor checkpoint")
    console.print(f"  final_lora.pt      — pesos LoRA finais")
    console.print(f"  config.json        — configuracao do treinamento")
    console.print(f"  training_history.json")
    console.print(f"  training_curves.png")


# ---------------------------------------------------------------------------
# Auxiliares de I/O
# ---------------------------------------------------------------------------

def save_lora_weights(model: nn.Module, path: Path) -> None:
    """Salva apenas os parametros treinaveis (LoRA + text_projection)."""
    state = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(state, path)


def load_lora_weights(model: nn.Module, path: Path) -> None:
    """Carrega pesos LoRA salvos."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model_state = model.state_dict()
    model_state.update(state)
    model.load_state_dict(model_state)


def plot_training_history(history: list[dict], path: Path) -> None:
    """Plota curvas de treinamento."""
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_sim = [h["val_sim"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, train_loss, "b-", label="Train Loss")
    ax1.plot(epochs, val_loss, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Contrastive Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Similaridade
    ax2.plot(epochs, val_sim, "g-", label="Val Positive Similarity")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Positive Pair Similarity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Curvas salvas: {path.name}")


if __name__ == "__main__":
    main()

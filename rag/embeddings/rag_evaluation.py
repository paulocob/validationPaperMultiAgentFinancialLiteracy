"""
RAG End-to-End Evaluation: mede precision@k, recall@k e MRR
para retrieval real usando diferentes modelos de embedding.

Pipeline:
  1. Indexa todas as respostas + image_descriptions como documentos
  2. Para cada pergunta, recupera top-k documentos mais similares
  3. Avalia se a resposta correta esta no top-k (retrieval accuracy)
  4. Compara: Baseline CLIP vs LoRA+CL vs XLM-R vs e5-multilingual

Uso:
    python -m embeddings.rag_evaluation
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from rich.console import Console
from rich.table import Table

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import load_all_countries, QAPair

console = Console()

K_VALUES = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# Document corpus: each QA pair generates 1-2 documents
# ---------------------------------------------------------------------------

def build_corpus(qa_pairs: list[QAPair]) -> list[dict]:
    """Build retrieval corpus from QA pairs."""
    docs = []
    for qa in qa_pairs:
        # Document 1: answer text
        docs.append({
            "id": f"{qa.id}_answer",
            "qa_id": qa.id,
            "text": qa.answer,
            "type": "answer",
            "country": qa.country,
            "topic": qa.topic,
        })
        # Document 2: image description (if exists)
        if qa.image_description:
            docs.append({
                "id": f"{qa.id}_image",
                "qa_id": qa.id,
                "text": qa.image_description,
                "type": "image_desc",
                "country": qa.country,
                "topic": qa.topic,
            })
    return docs


# ---------------------------------------------------------------------------
# Model loaders (reuse from model_benchmark but return encode_fn)
# ---------------------------------------------------------------------------

def _get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_openclip_base():
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    device = _get_device()
    model = model.float().to(device).eval()

    def encode(texts: list[str]) -> np.ndarray:
        tokens = tokenizer(texts)[:, :77].to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()
    return encode, "OpenCLIP ViT-B-32"


def load_openclip_lora_cl():
    import open_clip
    from embeddings.clip_lora_finetune import apply_lora_simple, load_lora_weights

    lora_path = _PROJECT_ROOT / "embeddings" / "lora_output_cl" / "best_lora.pt"
    if not lora_path.exists():
        return None, "LoRA+CL (not found)"

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.float()

    config_path = lora_path.parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    apply_lora_simple(model, rank=config.get("rank", 4), alpha=config.get("alpha", 8.0))
    load_lora_weights(model, lora_path)

    device = _get_device()
    model = model.to(device).eval()

    def encode(texts: list[str]) -> np.ndarray:
        tokens = tokenizer(texts)[:, :77].to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()
    return encode, "OpenCLIP + LoRA+CL"


def load_xlm_clip():
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "xlm-roberta-base-ViT-B-32", pretrained="laion5b_s13b_b90k"
    )
    tokenizer = open_clip.get_tokenizer("xlm-roberta-base-ViT-B-32")
    device = _get_device()
    model = model.float().to(device).eval()

    def encode(texts: list[str]) -> np.ndarray:
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()
    return encode, "XLM-R + ViT-B-32"


def load_e5_multilingual():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    def encode(texts: list[str]) -> np.ndarray:
        prefixed = [f"query: {t}" for t in texts]
        return model.encode(prefixed, normalize_embeddings=True)
    return encode, "multilingual-e5-large"


# ---------------------------------------------------------------------------
# Batch encoding with progress
# ---------------------------------------------------------------------------

def encode_batch(texts: list[str], encode_fn, batch_size: int = 64) -> np.ndarray:
    """Encode texts in batches."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = encode_fn(batch)
        all_embs.append(emb)
    return np.vstack(all_embs)


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    qa_pairs: list[QAPair],
    corpus: list[dict],
    encode_fn,
    k_values: list[int] = K_VALUES,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    For each question, retrieve top-k documents and check if the
    correct answer/image_desc is among them.
    """
    # Encode corpus
    corpus_texts = [d["text"] for d in corpus]
    corpus_embs = encode_batch(corpus_texts, encode_fn, batch_size)

    # Build qa_id -> corpus indices mapping
    qa_to_corpus = {}
    for idx, doc in enumerate(corpus):
        qa_id = doc["qa_id"]
        if qa_id not in qa_to_corpus:
            qa_to_corpus[qa_id] = []
        qa_to_corpus[qa_id].append(idx)

    # Encode queries
    queries = [qa.question for qa in qa_pairs]
    query_embs = encode_batch(queries, encode_fn, batch_size)

    # Compute similarities (query x corpus)
    # Do this in chunks to avoid memory issues
    max_k = max(k_values)
    records = []

    for i, qa in enumerate(qa_pairs):
        q_emb = query_embs[i:i+1]  # (1, d)
        sims = (q_emb @ corpus_embs.T).flatten()  # (n_docs,)

        # Get top-k indices
        top_indices = np.argsort(sims)[::-1][:max_k]

        # Ground truth: corpus indices for this QA pair
        gt_indices = set(qa_to_corpus.get(qa.id, []))

        # Compute metrics for each k
        row = {
            "qa_id": qa.id,
            "country": qa.country,
            "topic": qa.topic,
        }

        # MRR: reciprocal rank of first correct hit
        rr = 0.0
        for rank, idx in enumerate(top_indices):
            if idx in gt_indices:
                rr = 1.0 / (rank + 1)
                break
        row["mrr"] = rr

        for k in k_values:
            retrieved = set(top_indices[:k].tolist())
            hits = len(retrieved & gt_indices)
            row[f"precision@{k}"] = hits / k
            row[f"recall@{k}"] = hits / len(gt_indices) if gt_indices else 0.0
            row[f"hit@{k}"] = 1.0 if hits > 0 else 0.0

        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_retrieval_comparison(all_results: dict[str, pd.DataFrame], output_dir: Path):
    """Plot retrieval metrics comparison across models."""
    models = list(all_results.keys())
    countries = ["BR", "PT", "AO"]

    # --- Hit@k by model (overall) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Hit@k
    ax = axes[0]
    x = np.arange(len(K_VALUES))
    width = 0.8 / len(models)
    for i, model_name in enumerate(models):
        df = all_results[model_name]
        hits = [df[f"hit@{k}"].mean() for k in K_VALUES]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, hits, width, label=model_name)
        for bar, val in zip(bars, hits):
            ax.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"@{k}" for k in K_VALUES])
    ax.set_xlabel("k")
    ax.set_ylabel("Hit Rate")
    ax.set_title("Hit@k (overall)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # MRR by country
    ax = axes[1]
    x = np.arange(len(countries))
    width = 0.8 / len(models)
    for i, model_name in enumerate(models):
        df = all_results[model_name]
        mrrs = [df[df["country"] == c]["mrr"].mean() for c in countries]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, mrrs, width, label=model_name)
        for bar, val in zip(bars, mrrs):
            ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.set_xlabel("Country")
    ax.set_ylabel("Mean Reciprocal Rank")
    ax.set_title("MRR by Country")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("FinEduBench-Luso: RAG Retrieval Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "rag_retrieval_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Chart salvo: {path.name}")

    # --- Hit@k by country per model ---
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for i, model_name in enumerate(models):
        ax = axes[i]
        df = all_results[model_name]
        for c, color in [("BR", "#2196F3"), ("PT", "#4CAF50"), ("AO", "#FF9800")]:
            sub = df[df["country"] == c]
            hits = [sub[f"hit@{k}"].mean() for k in K_VALUES]
            ax.plot(K_VALUES, hits, marker="o", label=c, color=color, linewidth=2)
        ax.set_xlabel("k")
        ax.set_ylabel("Hit Rate")
        ax.set_title(model_name, fontsize=11)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Hit@k by Country per Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "rag_hit_by_country.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Chart salvo: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_LOADERS = [
    load_openclip_base,
    load_openclip_lora_cl,
    load_xlm_clip,
    load_e5_multilingual,
]


def main():
    console.print("\n[bold magenta]===== FinEduBench-Luso: RAG End-to-End Evaluation =====[/bold magenta]\n")

    output_dir = _PROJECT_ROOT / "embeddings" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    qa_pairs = load_all_countries(_PROJECT_ROOT / "dataset")
    console.print(f"  {len(qa_pairs)} QA pairs carregados.")

    # Build corpus
    corpus = build_corpus(qa_pairs)
    console.print(f"  {len(corpus)} documentos no corpus (respostas + descricoes de imagem).\n")

    all_results: dict[str, pd.DataFrame] = {}
    summary_rows = []

    for loader_fn in MODEL_LOADERS:
        encode_fn, model_name = loader_fn()
        if encode_fn is None:
            console.print(f"  [yellow]Pulando {model_name}[/yellow]\n")
            continue

        console.print(f"[bold green]Avaliando: {model_name}...[/bold green]")
        t0 = time.time()

        df = evaluate_retrieval(qa_pairs, corpus, encode_fn)
        elapsed = time.time() - t0

        all_results[model_name] = df

        # Overall metrics
        overall = {
            "model": model_name,
            "mrr": df["mrr"].mean(),
            "time_s": elapsed,
        }
        for k in K_VALUES:
            overall[f"hit@{k}"] = df[f"hit@{k}"].mean()
            overall[f"p@{k}"] = df[f"precision@{k}"].mean()
            overall[f"r@{k}"] = df[f"recall@{k}"].mean()

        # Per-country MRR
        for c in ["BR", "PT", "AO"]:
            sub = df[df["country"] == c]
            overall[f"mrr_{c.lower()}"] = sub["mrr"].mean()
            overall[f"hit@5_{c.lower()}"] = sub["hit@5"].mean()

        summary_rows.append(overall)

        console.print(
            f"  MRR={overall['mrr']:.4f} | "
            f"Hit@1={overall['hit@1']:.3f} Hit@5={overall['hit@5']:.3f} Hit@10={overall['hit@10']:.3f} | "
            f"{elapsed:.1f}s\n"
        )

        # Free memory
        del encode_fn
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # --- Summary Table ---
    table = Table(title="RAG Retrieval: Resumo", show_lines=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("MRR", justify="right")
    table.add_column("Hit@1", justify="right")
    table.add_column("Hit@5", justify="right")
    table.add_column("Hit@10", justify="right")
    table.add_column("P@5", justify="right")
    table.add_column("R@5", justify="right")

    for row in summary_rows:
        table.add_row(
            row["model"],
            f"{row['mrr']:.4f}",
            f"{row['hit@1']:.3f}",
            f"{row['hit@5']:.3f}",
            f"{row['hit@10']:.3f}",
            f"{row['p@5']:.3f}",
            f"{row['r@5']:.3f}",
        )
    console.print(table)

    # Per-country MRR table
    mrr_table = Table(title="MRR por Pais", show_lines=True)
    mrr_table.add_column("Model", style="bold cyan")
    mrr_table.add_column("BR", justify="right")
    mrr_table.add_column("PT", justify="right")
    mrr_table.add_column("AO", justify="right")
    mrr_table.add_column("Overall", justify="right", style="bold")

    for row in summary_rows:
        mrr_table.add_row(
            row["model"],
            f"{row['mrr_br']:.4f}",
            f"{row['mrr_pt']:.4f}",
            f"{row['mrr_ao']:.4f}",
            f"{row['mrr']:.4f}",
        )
    console.print(mrr_table)

    # Per-country Hit@5
    hit_table = Table(title="Hit@5 por Pais", show_lines=True)
    hit_table.add_column("Model", style="bold cyan")
    hit_table.add_column("BR", justify="right")
    hit_table.add_column("PT", justify="right")
    hit_table.add_column("AO", justify="right")

    for row in summary_rows:
        hit_table.add_row(
            row["model"],
            f"{row['hit@5_br']:.3f}",
            f"{row['hit@5_pt']:.3f}",
            f"{row['hit@5_ao']:.3f}",
        )
    console.print(hit_table)

    # Plots
    console.print("\n[bold green]Gerando visualizacoes...[/bold green]")
    plot_retrieval_comparison(all_results, output_dir)

    # Save CSV
    csv_rows = []
    for model_name, df in all_results.items():
        for _, row in df.iterrows():
            r = row.to_dict()
            r["model"] = model_name
            csv_rows.append(r)

    pd.DataFrame(csv_rows).to_csv(output_dir / "rag_retrieval_results.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "rag_retrieval_summary.csv", index=False)

    console.print(f"\n[bold magenta]Resultados salvos em {output_dir}[/bold magenta]\n")


if __name__ == "__main__":
    main()

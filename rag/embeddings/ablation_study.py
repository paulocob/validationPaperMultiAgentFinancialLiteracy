"""
Ablation Study: avalia impacto de hiperparametros no LoRA+CL fine-tuning.

Varia sistematicamente:
  1. LoRA rank: 2, 4, 8, 16  (cl_weight=0.3, anchors=300)
  2. CL weight: 0.0, 0.1, 0.3, 0.5, 0.7  (rank=4, anchors=300)
  3. Num anchors: 50, 100, 200, 300  (rank=4, cl_weight=0.3)

Para cada configuracao, treina o modelo e avalia degradacao no benchmark.

Uso:
    python -m embeddings.ablation_study
    python -m embeddings.ablation_study --axis rank      # so rank
    python -m embeddings.ablation_study --axis cl_weight  # so cl_weight
    python -m embeddings.ablation_study --axis anchors    # so anchors
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EMBED_DIR = _PROJECT_ROOT / "embeddings"
_RESULTS_DIR = _EMBED_DIR / "results"

console = Console()

# ---------------------------------------------------------------------------
# Configuracoes de ablation
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "rank": {
        "variable": "rank",
        "values": [2, 4, 8, 16],
        "fixed": {"cl_weight": 0.3, "num_anchors": 0},
        "label": "LoRA Rank",
    },
    "cl_weight": {
        "variable": "cl_weight",
        "values": [0.0, 0.1, 0.3, 0.5, 0.7],
        "fixed": {"rank": 4, "num_anchors": 0},
        "label": "Cross-Lingual Weight ($\\lambda_{cl}$)",
    },
    "anchors": {
        "variable": "num_anchors",
        "values": [50, 100, 200, 300],
        "fixed": {"rank": 4, "cl_weight": 0.3},
        "label": "Number of Anchor Pairs",
    },
}


def run_training(rank: int, cl_weight: float, num_anchors: int, output_dir: Path) -> dict | None:
    """Run a single training + evaluation cycle."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "embeddings.clip_lora_finetune",
        "--epochs", "20",
        "--rank", str(rank),
        "--alpha", str(rank * 2.0),  # alpha = 2 * rank
        "--batch-size", "32",
        "--lr", "1e-4",
        "--output-dir", str(output_dir),
    ]

    if cl_weight > 0:
        cmd.extend(["--cross-lingual", "--cl-weight", str(cl_weight)])
        if num_anchors > 0:
            cmd.extend(["--num-anchors", str(num_anchors)])

    console.print(f"  [dim]CMD: {' '.join(cmd[-8:])}[/dim]")

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_PROJECT_ROOT)
    )

    if result.returncode != 0:
        console.print(f"  [red]ERRO no treinamento:[/red]")
        console.print(result.stderr[-500:] if result.stderr else "sem stderr")
        return None

    # Read config.json for results
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def evaluate_model(lora_path: Path, config: dict) -> dict:
    """Evaluate a trained model on the benchmark (text-text gaps)."""
    import torch
    import open_clip

    from embeddings.clip_lora_finetune import apply_lora_simple, load_lora_weights
    from dataset.schema import load_all_countries

    ENGLISH_REFERENCE_PAIRS = [
        {"question": "What is the 50-30-20 budget rule?",
         "answer": "The 50-30-20 rule divides your after-tax income: 50% for needs, 30% for wants, and 20% for savings and debt repayment."},
        {"question": "How does a savings account work?",
         "answer": "A savings account earns interest on your deposited money. The bank pays you a percentage (APY) for keeping your money with them."},
        {"question": "What is a credit score and how to improve it?",
         "answer": "A credit score is a number (300-850) that represents your creditworthiness. Pay bills on time, keep credit utilization low, and avoid opening too many accounts."},
        {"question": "How do index funds work?",
         "answer": "Index funds track a market index like the S&P 500. They offer diversification, low fees, and historically strong returns for long-term investors."},
        {"question": "What is compound interest?",
         "answer": "Compound interest is interest earned on both the principal and previously accumulated interest. It accelerates wealth growth over time."},
        {"question": "How to build an emergency fund?",
         "answer": "Save 3-6 months of living expenses in a liquid, low-risk account. Start small and contribute consistently each month."},
        {"question": "What is inflation and how does it affect my money?",
         "answer": "Inflation is the rate at which prices increase over time, reducing purchasing power. A dollar today buys less than a dollar ten years ago."},
        {"question": "How do credit cards work and what are the risks?",
         "answer": "Credit cards allow you to borrow money for purchases up to a credit limit. If you don't pay the full balance, interest accrues at high APR rates."},
        {"question": "What is diversification in investing?",
         "answer": "Diversification means spreading investments across different asset classes to reduce risk. Don't put all your eggs in one basket."},
        {"question": "How does a mortgage work?",
         "answer": "A mortgage is a loan to buy property. You pay monthly installments of principal and interest over 15-30 years, with the property as collateral."},
    ]

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load model with LoRA
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.float()

    rank = config.get("rank", 4)
    alpha = config.get("alpha", rank * 2.0)
    apply_lora_simple(model, rank=rank, alpha=alpha)
    load_lora_weights(model, lora_path)
    model = model.to(device).eval()

    def encode(texts):
        tokens = tokenizer(texts)[:, :77].to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()

    # Compute EN baseline
    en_sims = []
    for pair in ENGLISH_REFERENCE_PAIRS:
        emb = encode([pair["question"], pair["answer"]])
        en_sims.append(float(np.dot(emb[0], emb[1])))
    en_tt_mean = np.mean(en_sims)

    # Compute per-country
    qa_pairs = load_all_countries(_PROJECT_ROOT / "dataset")
    country_sims = {"BR": [], "PT": [], "AO": []}
    for qa in qa_pairs:
        emb = encode([qa.question, qa.answer])
        sim = float(np.dot(emb[0], emb[1]))
        if qa.country in country_sims:
            country_sims[qa.country].append(sim)

    results = {"en_tt_mean": en_tt_mean}
    gaps = []
    for c in ["BR", "PT", "AO"]:
        c_mean = np.mean(country_sims[c])
        gap = (en_tt_mean - c_mean) / en_tt_mean * 100
        results[f"{c.lower()}_tt_mean"] = c_mean
        results[f"{c.lower()}_tt_gap"] = gap
        gaps.append(gap)
    results["mean_tt_gap"] = np.mean(gaps)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


def run_ablation_axis(axis_name: str, all_results: list[dict]) -> list[dict]:
    """Run all configs for one ablation axis."""
    cfg = ABLATION_CONFIGS[axis_name]
    variable = cfg["variable"]
    values = cfg["values"]
    fixed = cfg["fixed"]

    console.print(f"\n[bold magenta]--- Ablation: {cfg['label']} ---[/bold magenta]")
    console.print(f"  Variavel: {variable} = {values}")
    console.print(f"  Fixos: {fixed}\n")

    axis_results = []

    for val in values:
        # Build params
        params = dict(fixed)
        params[variable] = val

        rank = params.get("rank", 4)
        cl_weight = params.get("cl_weight", 0.3)
        num_anchors = params.get("num_anchors", 0)

        label = f"r{rank}_cl{cl_weight}_a{num_anchors if num_anchors > 0 else 300}"
        output_dir = _EMBED_DIR / "ablation" / axis_name / label

        console.print(f"[green]Config: {label}[/green]")

        # Check if already trained
        config_path = output_dir / "config.json"
        if config_path.exists():
            console.print(f"  [yellow]Ja treinado, carregando resultados...[/yellow]")
            with open(config_path) as f:
                train_config = json.load(f)
        else:
            train_config = run_training(rank, cl_weight, num_anchors, output_dir)

        if train_config is None:
            console.print(f"  [red]Falhou, pulando.[/red]")
            continue

        # Evaluate
        lora_path = output_dir / "best_lora.pt"
        if not lora_path.exists():
            console.print(f"  [red]best_lora.pt nao encontrado, pulando.[/red]")
            continue

        eval_results = evaluate_model(lora_path, train_config)

        row = {
            "axis": axis_name,
            "variable": variable,
            "value": val,
            "rank": rank,
            "cl_weight": cl_weight,
            "num_anchors": num_anchors if num_anchors > 0 else 300,
            "best_val_loss": train_config.get("best_val_loss", np.nan),
            "final_val_sim": train_config.get("final_val_sim", np.nan),
            "epochs_trained": train_config.get("epochs_trained", 0),
            **eval_results,
        }
        axis_results.append(row)
        all_results.append(row)

        console.print(
            f"  val_loss={row['best_val_loss']:.4f} | "
            f"gap: BR={row['br_tt_gap']:+.1f}% PT={row['pt_tt_gap']:+.1f}% "
            f"AO={row['ao_tt_gap']:+.1f}% | mean={row['mean_tt_gap']:+.1f}%"
        )

    return axis_results


def plot_ablation(df: pd.DataFrame, output_dir: Path):
    """Generate ablation plots for each axis."""
    axes_in_data = df["axis"].unique()

    fig, axes_list = plt.subplots(1, len(axes_in_data), figsize=(6 * len(axes_in_data), 5))
    if len(axes_in_data) == 1:
        axes_list = [axes_list]

    for i, axis_name in enumerate(axes_in_data):
        ax = axes_list[i]
        cfg = ABLATION_CONFIGS[axis_name]
        sub = df[df["axis"] == axis_name].sort_values("value")

        x = sub["value"].values
        x_labels = [str(v) for v in x]

        # Plot per-country gaps
        for country, color, marker in [("BR", "#2196F3", "o"), ("PT", "#4CAF50", "s"), ("AO", "#FF9800", "^")]:
            ax.plot(range(len(x)), sub[f"{country.lower()}_tt_gap"].values,
                    color=color, marker=marker, label=country, linewidth=2, markersize=8)

        # Mean gap
        ax.plot(range(len(x)), sub["mean_tt_gap"].values,
                color="red", marker="D", label="Mean", linewidth=2.5, markersize=9, linestyle="--")

        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel(cfg["label"], fontsize=12)
        ax.set_ylabel("Text-Text Gap vs EN (%)", fontsize=11)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Ablation: {cfg['label']}", fontsize=13, fontweight="bold")

    plt.suptitle("FinEduBench-Luso: Ablation Study", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "ablation_study.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"\n  Ablation plot salvo: {path}")

    # Also plot val_loss curves
    fig, axes_list = plt.subplots(1, len(axes_in_data), figsize=(6 * len(axes_in_data), 5))
    if len(axes_in_data) == 1:
        axes_list = [axes_list]

    for i, axis_name in enumerate(axes_in_data):
        ax = axes_list[i]
        cfg = ABLATION_CONFIGS[axis_name]
        sub = df[df["axis"] == axis_name].sort_values("value")

        x = sub["value"].values
        ax.bar(range(len(x)), sub["best_val_loss"].values, color="#673AB7", alpha=0.8)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([str(v) for v in x])
        ax.set_xlabel(cfg["label"], fontsize=12)
        ax.set_ylabel("Best Validation Loss", fontsize=11)
        ax.set_title(f"Val Loss: {cfg['label']}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate bars
        for j, v in enumerate(sub["best_val_loss"].values):
            ax.annotate(f"{v:.3f}", xy=(j, v), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=9)

    plt.suptitle("FinEduBench-Luso: Validation Loss by Config", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "ablation_val_loss.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Val loss plot salvo: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", type=str, default="all",
                        choices=["all", "rank", "cl_weight", "anchors"])
    args = parser.parse_args()

    console.print("\n[bold magenta]===== FinEduBench-Luso: Ablation Study =====[/bold magenta]")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    axes_to_run = list(ABLATION_CONFIGS.keys()) if args.axis == "all" else [args.axis]

    all_results: list[dict] = []

    for axis_name in axes_to_run:
        run_ablation_axis(axis_name, all_results)

    if not all_results:
        console.print("[red]Nenhum resultado obtido.[/red]")
        return

    df = pd.DataFrame(all_results)

    # Summary table
    table = Table(title="Ablation Study: Resumo", show_lines=True)
    table.add_column("Config", style="bold cyan")
    table.add_column("Rank", justify="right")
    table.add_column("CL Weight", justify="right")
    table.add_column("Anchors", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("BR Gap", justify="right")
    table.add_column("PT Gap", justify="right")
    table.add_column("AO Gap", justify="right")
    table.add_column("Mean Gap", justify="right", style="bold")

    for _, row in df.iterrows():
        table.add_row(
            f"{row['axis']}={row['value']}",
            str(row["rank"]),
            f"{row['cl_weight']:.1f}",
            str(row["num_anchors"]),
            f"{row['best_val_loss']:.4f}",
            f"{row['br_tt_gap']:+.1f}%",
            f"{row['pt_tt_gap']:+.1f}%",
            f"{row['ao_tt_gap']:+.1f}%",
            f"{row['mean_tt_gap']:+.1f}%",
        )

    console.print(table)

    # Save CSV
    csv_path = _RESULTS_DIR / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"\n  CSV salvo: {csv_path}")

    # Plots
    console.print("\n[bold green]Gerando visualizacoes...[/bold green]")
    plot_ablation(df, _RESULTS_DIR)

    # Best config
    best = df.loc[df["mean_tt_gap"].abs().idxmin()]
    console.print(
        f"\n[bold magenta]Melhor configuracao (menor |gap|):[/bold magenta]\n"
        f"  rank={int(best['rank'])}, cl_weight={best['cl_weight']:.1f}, "
        f"anchors={int(best['num_anchors'])}\n"
        f"  Mean gap: {best['mean_tt_gap']:+.1f}%  |  Val loss: {best['best_val_loss']:.4f}"
    )


if __name__ == "__main__":
    main()

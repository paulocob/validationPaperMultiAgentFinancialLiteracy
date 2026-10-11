"""
Multi-model benchmark: compara degradacao de embeddings entre
variantes do portugues para diferentes modelos multilingues.

Modelos avaliados:
  1. OpenCLIP ViT-B-32 (baseline monolingue)
  2. OpenCLIP ViT-B-32 + LoRA+CL (nosso fine-tune)
  3. XLM-RoBERTa-Base + ViT-B-32 (CLIP multilingue)
  4. NLLB-CLIP-Base (Meta, 200+ linguas)
  5. multilingual-e5-large (text-only, sentence-transformers)

Uso:
    python -m embeddings.model_benchmark
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from rich.console import Console
from rich.table import Table
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import load_all_countries, QAPair

console = Console()

ENGLISH_REFERENCE_PAIRS = [
    {"question": "What is the 50-30-20 budget rule?",
     "answer": "The 50-30-20 rule divides your after-tax income: 50% for needs, 30% for wants, and 20% for savings and debt repayment.",
     "image_description": "A pie chart divided into three colored slices: 50% blue (Needs), 30% green (Wants), 20% orange (Savings)."},
    {"question": "How does a savings account work?",
     "answer": "A savings account earns interest on your deposited money. The bank pays you a percentage (APY) for keeping your money with them.",
     "image_description": "Bar chart comparing interest rates of different savings products: regular savings, high-yield savings, CDs."},
    {"question": "What is a credit score and how to improve it?",
     "answer": "A credit score is a number (300-850) that represents your creditworthiness. Pay bills on time, keep credit utilization low, and avoid opening too many accounts.",
     "image_description": "Speedometer-style gauge showing credit score ranges from poor (red) to excellent (green)."},
    {"question": "How do index funds work?",
     "answer": "Index funds track a market index like the S&P 500. They offer diversification, low fees, and historically strong returns for long-term investors.",
     "image_description": "Line chart showing S&P 500 growth over 20 years with annotation showing compound growth effect."},
    {"question": "What is compound interest?",
     "answer": "Compound interest is interest earned on both the principal and previously accumulated interest. It accelerates wealth growth over time.",
     "image_description": "Two growth curves: simple interest (linear) vs compound interest (exponential), showing divergence over 30 years."},
    {"question": "How to build an emergency fund?",
     "answer": "Save 3-6 months of living expenses in a liquid, low-risk account. Start small and contribute consistently each month.",
     "image_description": "Thermometer showing savings milestones: 1 month (critical), 3 months (minimum), 6 months (ideal)."},
    {"question": "What is inflation and how does it affect my money?",
     "answer": "Inflation is the rate at which prices increase over time, reducing purchasing power. A dollar today buys less than a dollar ten years ago.",
     "image_description": "Infographic showing what $100 could buy in 2000 vs 2010 vs 2020, with shrinking shopping baskets."},
    {"question": "How do credit cards work and what are the risks?",
     "answer": "Credit cards allow you to borrow money for purchases up to a credit limit. If you don't pay the full balance, interest accrues at high APR rates.",
     "image_description": "Flowchart: purchase with credit card -> pay full balance (no interest) vs pay minimum (interest accumulates)."},
    {"question": "What is diversification in investing?",
     "answer": "Diversification means spreading investments across different asset classes to reduce risk. Don't put all your eggs in one basket.",
     "image_description": "Two portfolio pie charts: concentrated (high risk) vs diversified (lower risk), with volatility comparison."},
    {"question": "How does a mortgage work?",
     "answer": "A mortgage is a loan to buy property. You pay monthly installments of principal and interest over 15-30 years, with the property as collateral.",
     "image_description": "Amortization chart showing how payments shift from mostly interest to mostly principal over 30 years."},
]


# ---------------------------------------------------------------------------
# Model loaders — each returns (encode_fn, supports_image_text, model_name)
# encode_fn(texts) -> np.ndarray of normalized embeddings
# ---------------------------------------------------------------------------

def _get_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_openclip_base():
    """OpenCLIP ViT-B-32 (laion2b, monolingue)."""
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

    return encode, True, "OpenCLIP ViT-B-32"


def load_openclip_lora_cl():
    """OpenCLIP ViT-B-32 + LoRA+CL (nosso fine-tune)."""
    import open_clip
    import json
    from embeddings.clip_lora_finetune import apply_lora_simple, load_lora_weights

    lora_path = _PROJECT_ROOT / "embeddings" / "lora_output_cl" / "best_lora.pt"
    if not lora_path.exists():
        return None, False, "LoRA+CL (not found)"

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

    return encode, True, "OpenCLIP + LoRA+CL"


def load_xlm_clip():
    """XLM-RoBERTa-Base + ViT-B-32 (multilingue nativo)."""
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

    return encode, True, "XLM-R + ViT-B-32"


def load_nllb_clip():
    """NLLB-CLIP-Base (Meta, 200+ linguas)."""
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "nllb-clip-base", pretrained="v1"
    )
    tokenizer = open_clip.get_tokenizer("nllb-clip-base")
    device = _get_device()
    model = model.float().to(device).eval()

    def encode(texts: list[str]) -> np.ndarray:
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()

    return encode, True, "NLLB-CLIP-Base"


def load_e5_multilingual():
    """multilingual-e5-large (Microsoft, text-only)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("intfloat/multilingual-e5-large")

    def encode(texts: list[str]) -> np.ndarray:
        # e5 expects "query: " or "passage: " prefix
        prefixed = [f"query: {t}" for t in texts]
        emb = model.encode(prefixed, normalize_embeddings=True)
        return emb

    return encode, False, "multilingual-e5-large"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_sims(qa_pairs: list[QAPair], encode_fn, supports_ti: bool) -> pd.DataFrame:
    """Compute similarities for all QA pairs."""
    records = []
    for qa in qa_pairs:
        emb = encode_fn([qa.question, qa.answer])
        tt_sim = float(np.dot(emb[0], emb[1]))

        ti_sim = np.nan
        if supports_ti and qa.image_description:
            emb_ti = encode_fn([qa.question, qa.image_description])
            ti_sim = float(np.dot(emb_ti[0], emb_ti[1]))

        records.append({
            "country": qa.country, "topic": qa.topic,
            "text_text_sim": tt_sim, "text_image_sim": ti_sim,
        })
    return pd.DataFrame(records)


def compute_en_sims(encode_fn, supports_ti: bool) -> pd.DataFrame:
    records = []
    for pair in ENGLISH_REFERENCE_PAIRS:
        emb_tt = encode_fn([pair["question"], pair["answer"]])
        tt_sim = float(np.dot(emb_tt[0], emb_tt[1]))

        ti_sim = np.nan
        if supports_ti:
            emb_ti = encode_fn([pair["question"], pair["image_description"]])
            ti_sim = float(np.dot(emb_ti[0], emb_ti[1]))

        records.append({"country": "EN", "text_text_sim": tt_sim, "text_image_sim": ti_sim})
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for country in ["EN", "BR", "PT", "AO"]:
        sub = df[df["country"] == country]
        ti = sub["text_image_sim"].dropna()
        result[country] = {
            "tt_mean": sub["text_text_sim"].mean(),
            "tt_std": sub["text_text_sim"].std(),
            "ti_mean": ti.mean() if len(ti) > 0 else np.nan,
            "ti_std": ti.std() if len(ti) > 0 else np.nan,
            "n": len(sub),
        }
    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_comparison(all_results: dict[str, dict], output_dir: Path):
    """Bar chart comparing degradation gaps across models."""
    models = list(all_results.keys())
    countries = ["BR", "PT", "AO"]

    # --- Text-Text gap chart ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    x = np.arange(len(countries))
    width = 0.8 / len(models)

    for i, model_name in enumerate(models):
        s = all_results[model_name]
        en_tt = s["EN"]["tt_mean"]
        gaps = []
        for c in countries:
            gap = (en_tt - s[c]["tt_mean"]) / en_tt * 100 if en_tt != 0 else 0
            gaps.append(gap)
        offset = (i - len(models) / 2 + 0.5) * width
        bars = axes[0].bar(x + offset, gaps, width, label=model_name)
        for bar, val in zip(bars, gaps):
            axes[0].annotate(f"{val:+.1f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    axes[0].set_xlabel("Country")
    axes[0].set_ylabel("Degradation vs EN (%)")
    axes[0].set_title("Text-Text Gap (lower = better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(countries)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8, loc="upper right")

    # --- Text-Image gap chart ---
    # Only models that support T-I
    ti_models = [m for m in models if not np.isnan(all_results[m]["EN"].get("ti_mean", np.nan))]
    width_ti = 0.8 / max(len(ti_models), 1)

    for i, model_name in enumerate(ti_models):
        s = all_results[model_name]
        en_ti = s["EN"]["ti_mean"]
        gaps = []
        for c in countries:
            ti_mean = s[c]["ti_mean"]
            gap = (en_ti - ti_mean) / en_ti * 100 if (not np.isnan(en_ti) and en_ti != 0) else 0
            gaps.append(gap)
        offset = (i - len(ti_models) / 2 + 0.5) * width_ti
        bars = axes[1].bar(x + offset, gaps, width_ti, label=model_name)
        for bar, val in zip(bars, gaps):
            axes[1].annotate(f"{val:+.1f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    axes[1].set_xlabel("Country")
    axes[1].set_ylabel("Degradation vs EN (%)")
    axes[1].set_title("Text-Image Gap (lower = better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(countries)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8, loc="upper right")

    plt.suptitle("FinEduBench-Luso: Multi-Model Degradation Benchmark", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "benchmark_degradation_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Chart salvo: {path.name}")

    # --- Absolute similarity heatmap ---
    rows = []
    for model_name in models:
        s = all_results[model_name]
        for c in ["EN", "BR", "PT", "AO"]:
            rows.append({"Model": model_name, "Country": c,
                         "Text-Text": s[c]["tt_mean"],
                         "Text-Image": s[c].get("ti_mean", np.nan)})

    heat_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pivot_tt = heat_df.pivot(index="Model", columns="Country", values="Text-Text")[["EN", "BR", "PT", "AO"]]
    sns.heatmap(pivot_tt, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0], vmin=0.3, vmax=0.95)
    axes[0].set_title("Text-Text Similarity")

    pivot_ti = heat_df.pivot(index="Model", columns="Country", values="Text-Image")[["EN", "BR", "PT", "AO"]]
    sns.heatmap(pivot_ti, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1], vmin=0.3, vmax=0.95)
    axes[1].set_title("Text-Image Similarity")

    plt.suptitle("FinEduBench-Luso: Absolute Similarity by Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "benchmark_similarity_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Heatmap salvo: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_LOADERS = [
    ("OpenCLIP ViT-B-32", load_openclip_base),
    ("OpenCLIP + LoRA+CL", load_openclip_lora_cl),
    ("XLM-R + ViT-B-32", load_xlm_clip),
    ("NLLB-CLIP-Base", load_nllb_clip),
    ("multilingual-e5-large", load_e5_multilingual),
]


def main():
    console.print("\n[bold magenta]===== FinEduBench-Luso: Multi-Model Benchmark =====[/bold magenta]\n")

    output_dir = _PROJECT_ROOT / "embeddings" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_pairs = load_all_countries(_PROJECT_ROOT / "dataset")
    console.print(f"  {len(qa_pairs)} QA pairs carregados.\n")

    all_summaries: dict[str, dict] = {}
    csv_rows = []

    for model_name, loader_fn in MODEL_LOADERS:
        console.print(f"[bold green]Avaliando: {model_name}...[/bold green]")
        try:
            encode_fn, supports_ti, display_name = loader_fn()
            if encode_fn is None:
                console.print(f"  [yellow]Pulando {model_name} (nao disponivel)[/yellow]\n")
                continue
        except Exception as e:
            console.print(f"  [red]Erro ao carregar {model_name}: {e}[/red]\n")
            continue

        df_pt = compute_sims(qa_pairs, encode_fn, supports_ti)
        df_en = compute_en_sims(encode_fn, supports_ti)
        df_all = pd.concat([df_en, df_pt], ignore_index=True)

        s = summarize(df_all)
        all_summaries[model_name] = s

        # Per-model stats
        en_tt = s["EN"]["tt_mean"]
        en_ti = s["EN"]["ti_mean"]
        for c in ["EN", "BR", "PT", "AO"]:
            tt_gap = (en_tt - s[c]["tt_mean"]) / en_tt * 100 if c != "EN" else 0
            ti_gap = (en_ti - s[c]["ti_mean"]) / en_ti * 100 if (c != "EN" and not np.isnan(en_ti)) else 0
            csv_rows.append({
                "model": model_name, "country": c,
                "tt_mean": s[c]["tt_mean"], "tt_std": s[c]["tt_std"],
                "ti_mean": s[c]["ti_mean"], "ti_std": s[c]["ti_std"],
                "tt_gap_pct": tt_gap, "ti_gap_pct": ti_gap,
            })

        console.print(f"  {model_name}: EN T-T={s['EN']['tt_mean']:.4f} | "
                       f"BR={s['BR']['tt_mean']:.4f} PT={s['PT']['tt_mean']:.4f} AO={s['AO']['tt_mean']:.4f}")

        # Kruskal-Wallis
        luso = df_all[df_all["country"].isin(["BR", "PT", "AO"])]
        groups = [g["text_text_sim"].values for _, g in luso.groupby("country")]
        h, p = stats.kruskal(*groups)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        console.print(f"  KW text-text: H={h:.4f} p={p:.6f} {sig}")

        if supports_ti:
            luso_ti = luso[luso["text_image_sim"].notna()]
            groups_ti = [g["text_image_sim"].values for _, g in luso_ti.groupby("country")]
            h_ti, p_ti = stats.kruskal(*groups_ti)
            sig_ti = "***" if p_ti < 0.001 else "**" if p_ti < 0.01 else "*" if p_ti < 0.05 else "ns"
            console.print(f"  KW text-image: H={h_ti:.4f} p={p_ti:.6f} {sig_ti}")

        console.print()

        # Free memory
        del encode_fn
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # --- Summary Tables ---
    if not all_summaries:
        console.print("[red]Nenhum modelo avaliado com sucesso.[/red]")
        return

    # Text-Text table
    tt_table = Table(title="Text-Text Similarity (mean ± std)", show_lines=True)
    tt_table.add_column("Model", style="bold cyan")
    for c in ["EN", "BR", "PT", "AO"]:
        tt_table.add_column(c, justify="right")
    tt_table.add_column("Mean Gap", justify="right", style="bold")

    for model_name, s in all_summaries.items():
        en_tt = s["EN"]["tt_mean"]
        gaps = []
        row = [model_name]
        for c in ["EN", "BR", "PT", "AO"]:
            row.append(f"{s[c]['tt_mean']:.4f}±{s[c]['tt_std']:.3f}")
            if c != "EN":
                gaps.append((en_tt - s[c]["tt_mean"]) / en_tt * 100)
        row.append(f"{np.mean(gaps):+.1f}%")
        tt_table.add_row(*row)

    console.print(tt_table)

    # Text-Image table
    ti_table = Table(title="Text-Image Similarity (mean ± std)", show_lines=True)
    ti_table.add_column("Model", style="bold cyan")
    for c in ["EN", "BR", "PT", "AO"]:
        ti_table.add_column(c, justify="right")
    ti_table.add_column("Mean Gap", justify="right", style="bold")

    for model_name, s in all_summaries.items():
        en_ti = s["EN"]["ti_mean"]
        if np.isnan(en_ti):
            ti_table.add_row(model_name, "—", "—", "—", "—", "text-only")
            continue
        gaps = []
        row = [model_name]
        for c in ["EN", "BR", "PT", "AO"]:
            ti_val = s[c]["ti_mean"]
            ti_sd = s[c]["ti_std"]
            if np.isnan(ti_val):
                row.append("—")
            else:
                row.append(f"{ti_val:.4f}±{ti_sd:.3f}")
                if c != "EN":
                    gaps.append((en_ti - ti_val) / en_ti * 100)
        row.append(f"{np.mean(gaps):+.1f}%" if gaps else "—")
        ti_table.add_row(*row)

    console.print(ti_table)

    # --- Degradation Table ---
    deg_table = Table(title="Degradacao Text-Text vs English (%)", show_lines=True)
    deg_table.add_column("Model", style="bold cyan")
    for c in ["BR", "PT", "AO"]:
        deg_table.add_column(c, justify="right")
    deg_table.add_column("Media", justify="right", style="bold")

    for model_name, s in all_summaries.items():
        en_tt = s["EN"]["tt_mean"]
        gaps = []
        row = [model_name]
        for c in ["BR", "PT", "AO"]:
            gap = (en_tt - s[c]["tt_mean"]) / en_tt * 100
            gaps.append(gap)
            row.append(f"{gap:+.1f}%")
        row.append(f"{np.mean(gaps):+.1f}%")
        deg_table.add_row(*row)

    console.print(deg_table)

    # Plots
    console.print("\n[bold green]Gerando visualizacoes...[/bold green]")
    plot_comparison(all_summaries, output_dir)

    # Save CSV
    pd.DataFrame(csv_rows).to_csv(output_dir / "benchmark_results.csv", index=False)
    console.print(f"\n[bold magenta]Resultados salvos em {output_dir}[/bold magenta]\n")


if __name__ == "__main__":
    main()

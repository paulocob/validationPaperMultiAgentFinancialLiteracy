"""
Analise comparativa: Baseline vs LoRA vs LoRA+Cross-Lingual.

Calcula similaridades text-text e text-image para cada condicao
e gera tabela de degradacao relativa ao ingles.

Uso:
    python -m embeddings.clip_compare
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
from rich.console import Console
from rich.table import Table
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import load_all_countries, QAPair
from embeddings.clip_lora_finetune import apply_lora_simple, load_lora_weights

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


def load_model(lora_path: Path | None = None) -> tuple:
    """Load OpenCLIP model, optionally with LoRA weights."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.float()

    if lora_path is not None:
        import json
        config_path = lora_path.parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        rank = config.get("rank", 4)
        alpha = config.get("alpha", 8.0)
        apply_lora_simple(model, rank=rank, alpha=alpha)
        load_lora_weights(model, lora_path)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def encode_texts(texts: list[str], model, tokenizer, device: str) -> np.ndarray:
    tokens = tokenizer(texts)[:, :77].to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()


def compute_all_sims(qa_pairs: list[QAPair], model, tokenizer, device: str) -> pd.DataFrame:
    """Compute text-text and text-image similarities for QA pairs."""
    records = []
    for qa in qa_pairs:
        emb = encode_texts([qa.question, qa.answer], model, tokenizer, device)
        tt_sim = float(np.dot(emb[0], emb[1]))

        ti_sim = np.nan
        if qa.image_description:
            emb_ti = encode_texts([qa.question, qa.image_description], model, tokenizer, device)
            ti_sim = float(np.dot(emb_ti[0], emb_ti[1]))

        records.append({
            "country": qa.country, "topic": qa.topic,
            "text_text_sim": tt_sim, "text_image_sim": ti_sim,
        })
    return pd.DataFrame(records)


def compute_en_sims(model, tokenizer, device: str) -> pd.DataFrame:
    records = []
    for pair in ENGLISH_REFERENCE_PAIRS:
        emb_tt = encode_texts([pair["question"], pair["answer"]], model, tokenizer, device)
        tt_sim = float(np.dot(emb_tt[0], emb_tt[1]))
        emb_ti = encode_texts([pair["question"], pair["image_description"]], model, tokenizer, device)
        ti_sim = float(np.dot(emb_ti[0], emb_ti[1]))
        records.append({"country": "EN", "topic": "ref",
                        "text_text_sim": tt_sim, "text_image_sim": ti_sim})
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return per-country mean similarities."""
    summary = {}
    for country in ["EN", "BR", "PT", "AO"]:
        sub = df[df["country"] == country]
        ti = sub["text_image_sim"].dropna()
        summary[country] = {
            "tt_mean": sub["text_text_sim"].mean(),
            "ti_mean": ti.mean() if len(ti) > 0 else 0.0,
            "n": len(sub),
        }
    return summary


def main():
    console.print("\n[bold magenta]===== Analise Comparativa: Baseline vs LoRA vs LoRA+CL =====[/bold magenta]\n")

    # Load dataset
    qa_pairs = load_all_countries(_PROJECT_ROOT / "dataset")
    console.print(f"  {len(qa_pairs)} QA pairs carregados.\n")

    lora_path = _PROJECT_ROOT / "embeddings" / "lora_output" / "best_lora.pt"
    lora_cl_path = _PROJECT_ROOT / "embeddings" / "lora_output_cl" / "best_lora.pt"

    conditions = {"Baseline": None}
    if lora_path.exists():
        conditions["LoRA"] = lora_path
    if lora_cl_path.exists():
        conditions["LoRA+CL"] = lora_cl_path

    all_summaries = {}
    all_dfs = {}

    for name, path in conditions.items():
        console.print(f"[bold green]Avaliando: {name}...[/bold green]")
        model, tokenizer, device = load_model(path)

        df_pt = compute_all_sims(qa_pairs, model, tokenizer, device)
        df_en = compute_en_sims(model, tokenizer, device)
        df_all = pd.concat([df_en, df_pt], ignore_index=True)

        all_summaries[name] = summarize(df_all)
        all_dfs[name] = df_all
        console.print(f"  {name} concluido.\n")

        # Free memory
        del model
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    # --- Results Table ---
    table = Table(title="Comparacao de Similaridade Media", show_lines=True)
    table.add_column("Pais", style="bold cyan")
    for cond in conditions:
        table.add_column(f"{cond}\nT-T", justify="right")
        table.add_column(f"{cond}\nT-I", justify="right")

    for country in ["EN", "BR", "PT", "AO"]:
        label = {"EN": "English (ref)", "BR": "Brasil", "PT": "Portugal", "AO": "Angola"}[country]
        row = [label]
        for cond in conditions:
            s = all_summaries[cond][country]
            row.append(f"{s['tt_mean']:.4f}")
            row.append(f"{s['ti_mean']:.4f}")
        table.add_row(*row)

    console.print(table)

    # --- Degradation Table ---
    deg_table = Table(title="Degradacao vs English Baseline (%)", show_lines=True)
    deg_table.add_column("Pais", style="bold cyan")
    for cond in conditions:
        deg_table.add_column(f"{cond}\nT-T gap", justify="right")
        deg_table.add_column(f"{cond}\nT-I gap", justify="right")

    for country in ["BR", "PT", "AO"]:
        label = {"BR": "Brasil", "PT": "Portugal", "AO": "Angola"}[country]
        row = [label]
        for cond in conditions:
            en = all_summaries[cond]["EN"]
            c = all_summaries[cond][country]
            tt_gap = (en["tt_mean"] - c["tt_mean"]) / en["tt_mean"] * 100
            ti_gap = (en["ti_mean"] - c["ti_mean"]) / en["ti_mean"] * 100
            row.append(f"{tt_gap:+.1f}%")
            row.append(f"{ti_gap:+.1f}%")
        deg_table.add_row(*row)

    console.print(deg_table)

    # --- Statistical tests per condition ---
    console.print("\n[bold]Kruskal-Wallis (BR vs PT vs AO) por condicao:[/bold]")
    for cond in conditions:
        df = all_dfs[cond]
        luso = df[df["country"].isin(["BR", "PT", "AO"])]

        groups_tt = [g["text_text_sim"].values for _, g in luso.groupby("country")]
        h_tt, p_tt = stats.kruskal(*groups_tt)

        luso_ti = luso[luso["text_image_sim"].notna()]
        groups_ti = [g["text_image_sim"].values for _, g in luso_ti.groupby("country")]
        h_ti, p_ti = stats.kruskal(*groups_ti)

        sig_tt = "***" if p_tt < 0.001 else "**" if p_tt < 0.01 else "*" if p_tt < 0.05 else "ns"
        sig_ti = "***" if p_ti < 0.001 else "**" if p_ti < 0.01 else "*" if p_ti < 0.05 else "ns"

        console.print(f"  {cond}: T-T H={h_tt:.4f} p={p_tt:.6f} {sig_tt} | T-I H={h_ti:.4f} p={p_ti:.6f} {sig_ti}")

    # Save results
    output_dir = _PROJECT_ROOT / "embeddings" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cond in conditions:
        for country in ["EN", "BR", "PT", "AO"]:
            s = all_summaries[cond][country]
            en = all_summaries[cond]["EN"]
            tt_gap = (en["tt_mean"] - s["tt_mean"]) / en["tt_mean"] * 100 if country != "EN" else 0
            ti_gap = (en["ti_mean"] - s["ti_mean"]) / en["ti_mean"] * 100 if country != "EN" else 0
            rows.append({
                "condition": cond, "country": country,
                "tt_mean": s["tt_mean"], "ti_mean": s["ti_mean"],
                "tt_gap_pct": tt_gap, "ti_gap_pct": ti_gap,
            })

    pd.DataFrame(rows).to_csv(output_dir / "comparative_results.csv", index=False)
    console.print(f"\n[bold magenta]Resultados salvos em {output_dir / 'comparative_results.csv'}[/bold magenta]\n")


if __name__ == "__main__":
    main()

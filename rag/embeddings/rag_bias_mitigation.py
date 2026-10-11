"""
RAG Bias Mitigation: Testing strategies to reduce inter-country generation bias.

Previous experiment showed GPT-4o-mini produces lower quality answers for Angola
(ROUGE-L 0.320) vs Brazil (0.394) even with equitable retrieval (LoRA+CL).
The bias is in the LLM, not the retriever.

Strategies tested:
  1. BASELINE - current prompt (for comparison)
  2. COUNTRY-AWARE PROMPT - explicit country context with local terminology
  3. FEW-SHOT - 2 example QA pairs from same country as demonstrations
  4. CULTURAL NOTES - include cultural_notes field in generation context

Usage:
    export OPENAI_API_KEY="sk-..."
    python3 -m embeddings.rag_bias_mitigation
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from rouge_score import rouge_scorer
from scipy.stats import kruskal
from rich.console import Console
from rich.table import Table

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import load_all_countries, QAPair

# Reuse functions from rag_generation
from embeddings.rag_generation import (
    build_corpus,
    encode_batch,
    retrieve_top_k,
    compute_rouge,
    load_openclip_lora_cl,
    TOP_K,
    LLM_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
)

console = Console()

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

BASELINE_SYSTEM = (
    "Voce e um assistente especializado em educacao financeira para paises lusofonos "
    "(Brasil, Portugal, Angola). Responda de forma clara, precisa e culturalmente "
    "adaptada ao contexto do usuario. Responda em portugues."
)

COUNTRY_DETAILS = {
    "BR": {
        "name": "Brasil",
        "terms": "Brasil, incluindo termos como FGTS, CDI, Tesouro Direto, Bolsa Familia",
    },
    "PT": {
        "name": "Portugal",
        "terms": "Portugal, incluindo termos como PPR, TAEG, IRS, subsidio de Natal",
    },
    "AO": {
        "name": "Angola",
        "terms": "Angola, incluindo termos como kwanza, BNA, kinguilas, gasosa",
    },
}

PROMPT_WITH_CONTEXT = """Com base nos documentos recuperados abaixo, responda a pergunta.

Documentos recuperados:
{context}

Pergunta: {question}

Resposta:"""

PROMPT_WITH_CONTEXT_AND_NOTES = """Com base nos documentos recuperados abaixo, responda a pergunta.

Documentos recuperados:
{context}

Nota cultural: {cultural_notes}

Pergunta: {question}

Resposta:"""

PROMPT_FEWSHOT = """Com base nos documentos recuperados abaixo, responda a pergunta.

Documentos recuperados:
{context}

Aqui estao exemplos de perguntas e respostas sobre educacao financeira para este pais:

Exemplo 1:
Pergunta: {ex1_q}
Resposta: {ex1_a}

Exemplo 2:
Pergunta: {ex2_q}
Resposta: {ex2_a}

Agora responda:
Pergunta: {question}

Resposta:"""


# ---------------------------------------------------------------------------
# LLM call with custom system prompt
# ---------------------------------------------------------------------------

def call_llm_custom(prompt: str, system_prompt: str, client, max_retries: int = 3) -> str:
    """Call GPT-4o-mini with custom system prompt and retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                console.print(f"    [yellow]API error (attempt {attempt+1}): {e}. Retrying in {wait}s...[/yellow]")
                time.sleep(wait)
            else:
                console.print(f"    [red]API error after {max_retries} attempts: {e}[/red]")
                return ""


def make_context_str(retrieved_docs: list[dict]) -> str:
    """Format retrieved documents as numbered context string."""
    parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        parts.append(f"[{i}] {doc['text']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Strategy generators
# ---------------------------------------------------------------------------

def generate_baseline(qa: QAPair, retrieved_docs: list[dict], client) -> str:
    """Strategy 1: Baseline - same as original experiment."""
    context = make_context_str(retrieved_docs)
    prompt = PROMPT_WITH_CONTEXT.format(context=context, question=qa.question)
    return call_llm_custom(prompt, BASELINE_SYSTEM, client)


def generate_country_aware(qa: QAPair, retrieved_docs: list[dict], client) -> str:
    """Strategy 2: Country-aware prompt with local terminology."""
    info = COUNTRY_DETAILS[qa.country]
    system_prompt = (
        f"Voce e um assistente especializado em educacao financeira para paises lusofonos. "
        f"O usuario e de {info['name']}. Use terminologia, exemplos e referencias culturais "
        f"especificas de {info['terms']}. "
        f"Responda de forma clara, precisa e culturalmente adaptada ao contexto de {info['name']}. "
        f"Responda em portugues."
    )
    context = make_context_str(retrieved_docs)
    prompt = PROMPT_WITH_CONTEXT.format(context=context, question=qa.question)
    return call_llm_custom(prompt, system_prompt, client)


def generate_fewshot(qa: QAPair, retrieved_docs: list[dict], client,
                     examples_by_country: dict[str, list[QAPair]]) -> str:
    """Strategy 3: Few-shot with 2 same-country examples from different topics."""
    # Pick 2 examples from the same country but different topic
    candidates = [
        ex for ex in examples_by_country[qa.country]
        if ex.topic != qa.topic and ex.id != qa.id
    ]
    if len(candidates) < 2:
        # Fallback: allow same topic but different id
        candidates = [ex for ex in examples_by_country[qa.country] if ex.id != qa.id]

    selected = random.sample(candidates, min(2, len(candidates)))
    if len(selected) < 2:
        # Ultimate fallback: just use baseline
        return generate_baseline(qa, retrieved_docs, client)

    context = make_context_str(retrieved_docs)
    prompt = PROMPT_FEWSHOT.format(
        context=context,
        ex1_q=selected[0].question,
        ex1_a=selected[0].answer,
        ex2_q=selected[1].question,
        ex2_a=selected[1].answer,
        question=qa.question,
    )
    return call_llm_custom(prompt, BASELINE_SYSTEM, client)


def generate_cultural_notes(qa: QAPair, retrieved_docs: list[dict], client) -> str:
    """Strategy 4: Cultural notes included in context."""
    context = make_context_str(retrieved_docs)
    notes = qa.cultural_notes or "Nenhuma nota cultural disponivel."
    prompt = PROMPT_WITH_CONTEXT_AND_NOTES.format(
        context=context,
        cultural_notes=notes,
        question=qa.question,
    )
    return call_llm_custom(prompt, BASELINE_SYSTEM, client)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]OPENAI_API_KEY nao configurada.[/red]")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    console.print("\n[bold magenta]===== FinEduBench-Luso: RAG Bias Mitigation Experiment =====[/bold magenta]\n")

    output_dir = _PROJECT_ROOT / "embeddings" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = _PROJECT_ROOT.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    all_qa = load_all_countries(_PROJECT_ROOT / "dataset")
    console.print(f"  Total QA pairs: {len(all_qa)}")

    # Stratified sample: 50 per country = 150 total
    SEED = 42
    PER_COUNTRY = 50
    random.seed(SEED)

    by_country: dict[str, list[QAPair]] = {}
    for qa in all_qa:
        by_country.setdefault(qa.country, []).append(qa)

    sample: list[QAPair] = []
    for c in ["BR", "PT", "AO"]:
        pool = by_country[c].copy()
        random.shuffle(pool)
        sample.extend(pool[:PER_COUNTRY])
        console.print(f"    {c}: {min(PER_COUNTRY, len(pool))} sampled (from {len(pool)} total)")

    random.shuffle(sample)
    console.print(f"  Sample size: {len(sample)}\n")

    # Build few-shot example pool (all QA pairs NOT in sample, per country)
    sample_ids = {qa.id for qa in sample}
    examples_by_country: dict[str, list[QAPair]] = {}
    for qa in all_qa:
        if qa.id not in sample_ids:
            examples_by_country.setdefault(qa.country, []).append(qa)
    for c in ["BR", "PT", "AO"]:
        n = len(examples_by_country.get(c, []))
        console.print(f"    Few-shot pool {c}: {n} examples")
    console.print()

    # Load retriever: LoRA+CL only
    console.print("[bold green]Loading retriever: LoRA+CL...[/bold green]")
    encode_fn, retriever_name = load_openclip_lora_cl()
    if encode_fn is None:
        console.print("[red]LoRA+CL model not found. Aborting.[/red]")
        sys.exit(1)

    # Build corpus from ALL qa pairs (not just sample)
    corpus = build_corpus(all_qa)
    console.print(f"  Corpus: {len(corpus)} documents")

    # Encode corpus
    console.print("  Encoding corpus...")
    corpus_texts = [d["text"] for d in corpus]
    corpus_embs = encode_batch(corpus_texts, encode_fn)

    # Encode queries
    console.print("  Encoding queries...")
    query_texts = [qa.question for qa in sample]
    query_embs = encode_batch(query_texts, encode_fn)

    # Free retriever
    del encode_fn
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    console.print()

    # Pre-retrieve for all samples (shared across strategies)
    console.print("[bold green]Retrieving documents for all samples...[/bold green]")
    all_retrieved = []
    for i in range(len(sample)):
        retrieved = retrieve_top_k(query_embs[i:i+1], corpus_embs, corpus, TOP_K)
        all_retrieved.append(retrieved)
    console.print(f"  Done: {len(all_retrieved)} queries\n")

    # Define strategies
    strategies = {
        "Baseline": lambda qa, docs: generate_baseline(qa, docs, client),
        "Country-Aware": lambda qa, docs: generate_country_aware(qa, docs, client),
        "Few-Shot": lambda qa, docs: generate_fewshot(qa, docs, client, examples_by_country),
        "Cultural-Notes": lambda qa, docs: generate_cultural_notes(qa, docs, client),
    }

    # Run all strategies
    all_records = []
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    for strat_name, gen_fn in strategies.items():
        console.print(f"[bold cyan]Strategy: {strat_name}[/bold cyan]")
        t0 = time.time()

        for i, qa in enumerate(sample):
            retrieved = all_retrieved[i]
            answer = gen_fn(qa, retrieved)

            # Compute ROUGE
            scores = scorer.score(qa.answer, answer)
            record = {
                "strategy": strat_name,
                "qa_id": qa.id,
                "country": qa.country,
                "topic": qa.topic,
                "difficulty": qa.difficulty,
                "rouge1_f": scores["rouge1"].fmeasure,
                "rouge2_f": scores["rouge2"].fmeasure,
                "rougeL_f": scores["rougeL"].fmeasure,
                "prediction": answer,
            }
            all_records.append(record)

            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(sample) - i - 1) / rate if rate > 0 else 0
                console.print(
                    f"    {i+1}/{len(sample)} | "
                    f"{elapsed:.0f}s elapsed | ~{eta:.0f}s remaining"
                )

        elapsed = time.time() - t0
        console.print(f"  Completed: {elapsed:.1f}s\n")

    # Build DataFrame
    df = pd.DataFrame(all_records)

    # Save results (without prediction text)
    csv_path = output_dir / "rag_bias_mitigation_results.csv"
    df.drop(columns=["prediction"]).to_csv(csv_path, index=False)
    console.print(f"[bold green]Results saved: {csv_path}[/bold green]\n")

    # ---------------------------------------------------------------------------
    # Analysis & Summary
    # ---------------------------------------------------------------------------
    _print_summary(df)

    # ---------------------------------------------------------------------------
    # Statistical tests
    # ---------------------------------------------------------------------------
    console.print("\n[bold magenta]Kruskal-Wallis Tests (ROUGE-L by country, per strategy):[/bold magenta]")
    stat_table = Table(title="Kruskal-Wallis: ROUGE-L by Country", show_lines=True)
    stat_table.add_column("Strategy", style="bold")
    stat_table.add_column("H-statistic", justify="right")
    stat_table.add_column("p-value", justify="right")
    stat_table.add_column("Significant?", justify="center")

    for strat in df["strategy"].unique():
        sub = df[df["strategy"] == strat]
        groups = [sub[sub["country"] == c]["rougeL_f"].values for c in ["BR", "PT", "AO"]]
        if all(len(g) > 0 for g in groups):
            H, p = kruskal(*groups)
            sig = "Yes *" if p < 0.05 else "No"
            stat_table.add_row(strat, f"{H:.4f}", f"{p:.4f}", sig)
        else:
            stat_table.add_row(strat, "N/A", "N/A", "N/A")
    console.print(stat_table)

    # Inter-country spread (max - min mean ROUGE-L across countries)
    console.print("\n[bold magenta]Inter-Country Spread (max - min mean ROUGE-L):[/bold magenta]")
    spread_table = Table(title="Inter-Country ROUGE-L Spread", show_lines=True)
    spread_table.add_column("Strategy", style="bold")
    spread_table.add_column("BR", justify="right")
    spread_table.add_column("PT", justify="right")
    spread_table.add_column("AO", justify="right")
    spread_table.add_column("Spread", justify="right", style="bold red")
    spread_table.add_column("Variance", justify="right")

    for strat in df["strategy"].unique():
        sub = df[df["strategy"] == strat]
        means = {}
        for c in ["BR", "PT", "AO"]:
            means[c] = sub[sub["country"] == c]["rougeL_f"].mean()
        vals = list(means.values())
        spread = max(vals) - min(vals)
        var = np.var(vals)
        spread_table.add_row(
            strat,
            f"{means['BR']:.4f}",
            f"{means['PT']:.4f}",
            f"{means['AO']:.4f}",
            f"{spread:.4f}",
            f"{var:.6f}",
        )
    console.print(spread_table)

    # ---------------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------------
    _plot_results(df, output_dir, figures_dir)

    console.print(f"\n[bold magenta]Done! Results in {output_dir}[/bold magenta]\n")


def _print_summary(df: pd.DataFrame):
    """Print per-strategy, per-country summary tables."""
    # Overall per-strategy
    table = Table(title="Overall ROUGE Scores by Strategy", show_lines=True)
    table.add_column("Strategy", style="bold cyan")
    table.add_column("ROUGE-1", justify="right")
    table.add_column("ROUGE-2", justify="right")
    table.add_column("ROUGE-L", justify="right")
    table.add_column("N", justify="right")

    for strat in df["strategy"].unique():
        sub = df[df["strategy"] == strat]
        table.add_row(
            strat,
            f"{sub['rouge1_f'].mean():.4f}",
            f"{sub['rouge2_f'].mean():.4f}",
            f"{sub['rougeL_f'].mean():.4f}",
            str(len(sub)),
        )
    console.print(table)

    # Per-strategy per-country
    for strat in df["strategy"].unique():
        sub = df[df["strategy"] == strat]
        table = Table(title=f"ROUGE-L by Country: {strat}", show_lines=True)
        table.add_column("Country", style="bold")
        table.add_column("ROUGE-1", justify="right")
        table.add_column("ROUGE-2", justify="right")
        table.add_column("ROUGE-L", justify="right")
        table.add_column("N", justify="right")

        for c in ["BR", "PT", "AO"]:
            csub = sub[sub["country"] == c]
            table.add_row(
                c,
                f"{csub['rouge1_f'].mean():.4f}",
                f"{csub['rouge2_f'].mean():.4f}",
                f"{csub['rougeL_f'].mean():.4f}",
                str(len(csub)),
            )
        console.print(table)


def _plot_results(df: pd.DataFrame, output_dir: Path, figures_dir: Path):
    """Generate comparison visualization."""
    strategies = df["strategy"].unique().tolist()
    countries = ["BR", "PT", "AO"]
    country_labels = {"BR": "Brazil", "PT": "Portugal", "AO": "Angola"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: Grouped bar chart (strategy x country) ---
    ax = axes[0]
    x = np.arange(len(strategies))
    width = 0.25
    colors = {"BR": "#2ecc71", "PT": "#3498db", "AO": "#e74c3c"}

    for j, c in enumerate(countries):
        vals = []
        errs = []
        for strat in strategies:
            sub = df[(df["strategy"] == strat) & (df["country"] == c)]
            vals.append(sub["rougeL_f"].mean())
            errs.append(sub["rougeL_f"].std() / np.sqrt(len(sub)))  # SE
        bars = ax.bar(x + (j - 1) * width, vals, width, yerr=errs, label=country_labels[c],
                      color=colors[c], capsize=3, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_ylabel("ROUGE-L (F1)")
    ax.set_title("ROUGE-L by Strategy and Country")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, df["rougeL_f"].max() * 1.35)

    # --- Panel 2: Inter-country spread comparison ---
    ax = axes[1]
    spreads = []
    for strat in strategies:
        sub = df[df["strategy"] == strat]
        means = [sub[sub["country"] == c]["rougeL_f"].mean() for c in countries]
        spreads.append(max(means) - min(means))

    bar_colors = ["#95a5a6", "#f39c12", "#9b59b6", "#1abc9c"]
    bars = ax.bar(strategies, spreads, color=bar_colors[:len(strategies)],
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, spreads):
        ax.annotate(f"{val:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("ROUGE-L Spread (max - min)")
    ax.set_title("Inter-Country Bias (lower = fairer)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(spreads) * 1.5)
    ax.axhline(y=spreads[0], color="red", linestyle="--", alpha=0.5, label="Baseline spread")
    ax.legend(fontsize=8)

    # --- Panel 3: Box plot of ROUGE-L by strategy and country ---
    ax = axes[2]
    # Prepare data for boxplot
    plot_data = []
    for strat in strategies:
        for c in countries:
            sub = df[(df["strategy"] == strat) & (df["country"] == c)]
            for val in sub["rougeL_f"].values:
                plot_data.append({"Strategy": strat, "Country": country_labels[c], "ROUGE-L": val})
    plot_df = pd.DataFrame(plot_data)

    box_colors = {country_labels[c]: colors[c] for c in countries}
    sns.boxplot(data=plot_df, x="Strategy", y="ROUGE-L", hue="Country",
                palette=box_colors, ax=ax, linewidth=0.8, fliersize=3)
    ax.set_title("ROUGE-L Distribution by Strategy")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("FinEduBench-Luso: RAG Bias Mitigation Strategies",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save
    plot_path = output_dir / "rag_bias_mitigation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    console.print(f"  Plot saved: {plot_path}")

    # Copy to figures directory
    figures_path = figures_dir / "rag_bias_mitigation.png"
    shutil.copy2(plot_path, figures_path)
    console.print(f"  Plot copied to: {figures_path}")

    plt.close()


if __name__ == "__main__":
    main()

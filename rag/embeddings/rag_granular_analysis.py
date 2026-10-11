"""
RAG Granular Analysis: desagregacao por topico, dificuldade e testes estatisticos.

Lê os resultados do rag_evaluation.py (CSV) e do dataset (para obter difficulty),
produz:
  1. MRR e Hit@5 por topico (6 topicos x 4 modelos x 3 paises)
  2. MRR e Hit@5 por dificuldade (basic/intermediate/advanced)
  3. Testes Kruskal-Wallis inter-pais para cada modelo
  4. Visualizacoes: heatmaps e barplots

Uso:
    python -m embeddings.rag_granular_analysis
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from rich.console import Console
from rich.table import Table

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import load_all_countries

console = Console()

TOPICS = ["budget", "savings", "debt", "investment", "credit", "cultural"]
TOPIC_LABELS = {
    "budget": "Orcamento",
    "savings": "Poupanca",
    "debt": "Divida",
    "investment": "Investimento",
    "credit": "Credito",
    "cultural": "Cultural",
}
DIFFICULTIES = ["basic", "intermediate", "advanced"]
COUNTRIES = ["BR", "PT", "AO"]


def load_data() -> pd.DataFrame:
    """Load RAG results CSV and merge with difficulty from dataset."""
    results_path = _PROJECT_ROOT / "embeddings" / "results" / "rag_retrieval_results.csv"
    df = pd.read_csv(results_path)

    # Load QA pairs to get difficulty
    qa_pairs = load_all_countries(_PROJECT_ROOT / "dataset")
    diff_map = {qa.id: qa.difficulty for qa in qa_pairs}
    df["difficulty"] = df["qa_id"].map(diff_map)

    console.print(f"  {len(df)} registros carregados ({df['model'].nunique()} modelos).")
    console.print(f"  Distribuicao dificuldade: {df.groupby('difficulty')['qa_id'].nunique().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# 1. Analise por topico
# ---------------------------------------------------------------------------

def analysis_by_topic(df: pd.DataFrame) -> pd.DataFrame:
    """MRR and Hit@5 by topic, model, country."""
    rows = []
    for model in df["model"].unique():
        for topic in TOPICS:
            sub = df[(df["model"] == model) & (df["topic"] == topic)]
            row = {
                "model": model,
                "topic": topic,
                "topic_label": TOPIC_LABELS[topic],
                "n": len(sub),
                "mrr": sub["mrr"].mean(),
                "hit@5": sub["hit@5"].mean(),
                "hit@1": sub["hit@1"].mean(),
                "recall@5": sub["recall@5"].mean(),
            }
            for c in COUNTRIES:
                csub = sub[sub["country"] == c]
                row[f"mrr_{c}"] = csub["mrr"].mean() if len(csub) > 0 else np.nan
                row[f"hit@5_{c}"] = csub["hit@5"].mean() if len(csub) > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def print_topic_table(topic_df: pd.DataFrame):
    """Print topic analysis as rich table."""
    for model in topic_df["model"].unique():
        sub = topic_df[topic_df["model"] == model]
        table = Table(title=f"Por Topico: {model}", show_lines=True)
        table.add_column("Topico", style="bold cyan")
        table.add_column("N", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("Hit@5", justify="right")
        table.add_column("MRR_BR", justify="right")
        table.add_column("MRR_PT", justify="right")
        table.add_column("MRR_AO", justify="right")
        table.add_column("Gap Max", justify="right")

        for _, row in sub.iterrows():
            mrrs = [row["mrr_BR"], row["mrr_PT"], row["mrr_AO"]]
            mrrs_valid = [m for m in mrrs if not np.isnan(m)]
            gap = max(mrrs_valid) - min(mrrs_valid) if len(mrrs_valid) > 1 else 0.0
            table.add_row(
                row["topic_label"],
                str(int(row["n"])),
                f"{row['mrr']:.3f}",
                f"{row['hit@5']:.3f}",
                f"{row['mrr_BR']:.3f}" if not np.isnan(row["mrr_BR"]) else "-",
                f"{row['mrr_PT']:.3f}" if not np.isnan(row["mrr_PT"]) else "-",
                f"{row['mrr_AO']:.3f}" if not np.isnan(row["mrr_AO"]) else "-",
                f"{gap:.3f}",
            )
        console.print(table)


# ---------------------------------------------------------------------------
# 2. Analise por dificuldade
# ---------------------------------------------------------------------------

def analysis_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """MRR and Hit@5 by difficulty, model, country."""
    rows = []
    for model in df["model"].unique():
        for diff in DIFFICULTIES:
            sub = df[(df["model"] == model) & (df["difficulty"] == diff)]
            row = {
                "model": model,
                "difficulty": diff,
                "n": len(sub),
                "mrr": sub["mrr"].mean(),
                "hit@5": sub["hit@5"].mean(),
                "hit@1": sub["hit@1"].mean(),
                "recall@5": sub["recall@5"].mean(),
            }
            for c in COUNTRIES:
                csub = sub[sub["country"] == c]
                row[f"mrr_{c}"] = csub["mrr"].mean() if len(csub) > 0 else np.nan
                row[f"hit@5_{c}"] = csub["hit@5"].mean() if len(csub) > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def print_difficulty_table(diff_df: pd.DataFrame):
    """Print difficulty analysis as rich table."""
    for model in diff_df["model"].unique():
        sub = diff_df[diff_df["model"] == model]
        table = Table(title=f"Por Dificuldade: {model}", show_lines=True)
        table.add_column("Nivel", style="bold cyan")
        table.add_column("N", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("Hit@5", justify="right")
        table.add_column("MRR_BR", justify="right")
        table.add_column("MRR_PT", justify="right")
        table.add_column("MRR_AO", justify="right")

        for _, row in sub.iterrows():
            table.add_row(
                row["difficulty"].capitalize(),
                str(int(row["n"])),
                f"{row['mrr']:.3f}",
                f"{row['hit@5']:.3f}",
                f"{row['mrr_BR']:.3f}" if not np.isnan(row["mrr_BR"]) else "-",
                f"{row['mrr_PT']:.3f}" if not np.isnan(row["mrr_PT"]) else "-",
                f"{row['mrr_AO']:.3f}" if not np.isnan(row["mrr_AO"]) else "-",
            )
        console.print(table)


# ---------------------------------------------------------------------------
# 3. Testes estatisticos (Kruskal-Wallis)
# ---------------------------------------------------------------------------

def statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Kruskal-Wallis test on MRR across countries for each model."""
    rows = []
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        groups = [sub[sub["country"] == c]["mrr"].values for c in COUNTRIES]

        # Kruskal-Wallis H-test
        if all(len(g) > 0 for g in groups):
            h_stat, p_value = stats.kruskal(*groups)
        else:
            h_stat, p_value = np.nan, np.nan

        # Effect size: epsilon-squared = H / (N-1)
        n_total = sum(len(g) for g in groups)
        eps_sq = h_stat / (n_total - 1) if n_total > 1 else np.nan

        # Pairwise Mann-Whitney U
        pairs = [("BR", "PT"), ("BR", "AO"), ("PT", "AO")]
        pairwise = {}
        for c1, c2 in pairs:
            g1 = sub[sub["country"] == c1]["mrr"].values
            g2 = sub[sub["country"] == c2]["mrr"].values
            if len(g1) > 0 and len(g2) > 0:
                u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                pairwise[f"U_{c1}_{c2}"] = u_stat
                pairwise[f"p_{c1}_{c2}"] = u_p
            else:
                pairwise[f"U_{c1}_{c2}"] = np.nan
                pairwise[f"p_{c1}_{c2}"] = np.nan

        rows.append({
            "model": model,
            "H_statistic": h_stat,
            "p_value": p_value,
            "epsilon_squared": eps_sq,
            "significant": "Yes" if p_value < 0.05 else "No",
            **pairwise,
        })
    return pd.DataFrame(rows)


def print_stats_table(stats_df: pd.DataFrame):
    """Print Kruskal-Wallis results."""
    table = Table(title="Kruskal-Wallis: MRR inter-pais", show_lines=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("H", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("eps^2", justify="right")
    table.add_column("Sig?", justify="center")
    table.add_column("p(BR-PT)", justify="right")
    table.add_column("p(BR-AO)", justify="right")
    table.add_column("p(PT-AO)", justify="right")

    for _, row in stats_df.iterrows():
        table.add_row(
            row["model"],
            f"{row['H_statistic']:.2f}",
            f"{row['p_value']:.4f}",
            f"{row['epsilon_squared']:.4f}",
            "[red]Yes[/red]" if row["significant"] == "Yes" else "[green]No[/green]",
            f"{row['p_BR_PT']:.4f}",
            f"{row['p_BR_AO']:.4f}",
            f"{row['p_PT_AO']:.4f}",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# 4. Visualizacoes
# ---------------------------------------------------------------------------

def plot_topic_heatmap(topic_df: pd.DataFrame, output_dir: Path):
    """Heatmap of MRR by topic for each model."""
    models = topic_df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for i, model in enumerate(models):
        sub = topic_df[topic_df["model"] == model]
        pivot = pd.DataFrame(index=TOPICS)
        for c in COUNTRIES:
            pivot[c] = sub.set_index("topic")[f"mrr_{c}"].reindex(TOPICS)

        # Use topic labels
        pivot.index = [TOPIC_LABELS[t] for t in pivot.index]

        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                    vmin=0, vmax=1, ax=axes[i], cbar=i == len(models) - 1,
                    linewidths=0.5)
        axes[i].set_title(model, fontsize=10, fontweight="bold")
        axes[i].set_ylabel("" if i > 0 else "Topico")

    plt.suptitle("MRR por Topico e Pais", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "rag_topic_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Chart salvo: {path.name}")


def plot_difficulty_bars(diff_df: pd.DataFrame, output_dir: Path):
    """Bar plot: MRR by difficulty level for each model."""
    models = diff_df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    colors = {"BR": "#2196F3", "PT": "#4CAF50", "AO": "#FF9800"}

    for i, model in enumerate(models):
        ax = axes[i]
        sub = diff_df[diff_df["model"] == model]
        x = np.arange(len(DIFFICULTIES))
        width = 0.25

        for j, c in enumerate(COUNTRIES):
            vals = sub.set_index("difficulty").reindex(DIFFICULTIES)[f"mrr_{c}"].values
            offset = (j - 1) * width
            bars = ax.bar(x + offset, vals, width, label=c, color=colors[c])
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.annotate(f"{val:.2f}",
                                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in DIFFICULTIES])
        ax.set_ylabel("MRR" if i == 0 else "")
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("MRR por Nivel de Dificuldade e Pais", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "rag_difficulty_bars.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Chart salvo: {path.name}")


def plot_topic_model_comparison(topic_df: pd.DataFrame, output_dir: Path):
    """Grouped bar: overall MRR by topic, comparing models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    models = topic_df["model"].unique()
    x = np.arange(len(TOPICS))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        sub = topic_df[topic_df["model"] == model]
        vals = sub.set_index("topic").reindex(TOPICS)["mrr"].values
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([TOPIC_LABELS[t] for t in TOPICS])
    ax.set_ylabel("MRR")
    ax.set_title("MRR por Topico: Comparacao entre Modelos", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = output_dir / "rag_topic_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Chart salvo: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    console.print("\n[bold magenta]===== FinEduBench-Luso: RAG Granular Analysis =====[/bold magenta]\n")

    output_dir = _PROJECT_ROOT / "embeddings" / "results"

    # Load data
    df = load_data()
    console.print()

    # 1. Topic analysis
    console.print("[bold]1. Analise por Topico[/bold]\n")
    topic_df = analysis_by_topic(df)
    print_topic_table(topic_df)

    # 2. Difficulty analysis
    console.print("\n[bold]2. Analise por Dificuldade[/bold]\n")
    diff_df = analysis_by_difficulty(df)
    print_difficulty_table(diff_df)

    # 3. Statistical tests
    console.print("\n[bold]3. Testes Estatisticos[/bold]\n")
    stats_df = statistical_tests(df)
    print_stats_table(stats_df)

    # Also test per-topic significance for LoRA+CL
    console.print("\n[bold]  Kruskal-Wallis por topico (LoRA+CL):[/bold]")
    lora_df = df[df["model"] == "OpenCLIP + LoRA+CL"]
    topic_stats_rows = []
    for topic in TOPICS:
        sub = lora_df[lora_df["topic"] == topic]
        groups = [sub[sub["country"] == c]["mrr"].values for c in COUNTRIES]
        if all(len(g) > 0 for g in groups):
            h, p = stats.kruskal(*groups)
            topic_stats_rows.append({
                "topic": TOPIC_LABELS[topic], "H": h, "p": p,
                "sig": "Yes" if p < 0.05 else "No",
            })

    t = Table(show_lines=True)
    t.add_column("Topico", style="bold")
    t.add_column("H", justify="right")
    t.add_column("p-value", justify="right")
    t.add_column("Sig?", justify="center")
    for r in topic_stats_rows:
        t.add_row(r["topic"], f"{r['H']:.2f}", f"{r['p']:.4f}",
                  "[red]Yes[/red]" if r["sig"] == "Yes" else "[green]No[/green]")
    console.print(t)

    # 4. Visualizations
    console.print("\n[bold green]Gerando visualizacoes...[/bold green]")
    plot_topic_heatmap(topic_df, output_dir)
    plot_difficulty_bars(diff_df, output_dir)
    plot_topic_model_comparison(topic_df, output_dir)

    # 5. Save CSVs
    topic_df.to_csv(output_dir / "rag_topic_analysis.csv", index=False)
    diff_df.to_csv(output_dir / "rag_difficulty_analysis.csv", index=False)
    stats_df.to_csv(output_dir / "rag_statistical_tests.csv", index=False)

    console.print(f"\n[bold magenta]Resultados salvos em {output_dir}[/bold magenta]\n")


if __name__ == "__main__":
    main()

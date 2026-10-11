"""
Lusofono-CLIP-Bench: analise de degradacao de embeddings CLIP
entre variantes do portugues (pt-BR, pt-PT, pt-AO).

Compara similaridade de embeddings de texto e imagem (via descricao)
entre paises lusófonos e contra um baseline em ingles.
Gera visualizacoes e testes estatisticos.

Uso:
    python -m embeddings.clip_analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.table import Table
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import load_all_countries, QAPair

console = Console()

# ---------------------------------------------------------------------------
# Pares de referencia em ingles (para calcular degradacao relativa)
# ---------------------------------------------------------------------------

ENGLISH_REFERENCE_PAIRS: list[dict[str, str]] = [
    {
        "question": "What is the 50-30-20 budget rule?",
        "answer": "The 50-30-20 rule divides your after-tax income: 50% for needs, 30% for wants, and 20% for savings and debt repayment.",
        "image_description": "A pie chart divided into three colored slices: 50% blue (Needs), 30% green (Wants), 20% orange (Savings).",
    },
    {
        "question": "How does a savings account work?",
        "answer": "A savings account earns interest on your deposited money. The bank pays you a percentage (APY) for keeping your money with them.",
        "image_description": "Bar chart comparing interest rates of different savings products: regular savings, high-yield savings, CDs.",
    },
    {
        "question": "What is a credit score and how to improve it?",
        "answer": "A credit score is a number (300-850) that represents your creditworthiness. Pay bills on time, keep credit utilization low, and avoid opening too many accounts.",
        "image_description": "Speedometer-style gauge showing credit score ranges from poor (red) to excellent (green).",
    },
    {
        "question": "How do index funds work?",
        "answer": "Index funds track a market index like the S&P 500. They offer diversification, low fees, and historically strong returns for long-term investors.",
        "image_description": "Line chart showing S&P 500 growth over 20 years with annotation showing compound growth effect.",
    },
    {
        "question": "What is compound interest?",
        "answer": "Compound interest is interest earned on both the principal and previously accumulated interest. It accelerates wealth growth over time.",
        "image_description": "Two growth curves: simple interest (linear) vs compound interest (exponential), showing divergence over 30 years.",
    },
    {
        "question": "How to build an emergency fund?",
        "answer": "Save 3-6 months of living expenses in a liquid, low-risk account. Start small and contribute consistently each month.",
        "image_description": "Thermometer showing savings milestones: 1 month (critical), 3 months (minimum), 6 months (ideal).",
    },
    {
        "question": "What is inflation and how does it affect my money?",
        "answer": "Inflation is the rate at which prices increase over time, reducing purchasing power. A dollar today buys less than a dollar ten years ago.",
        "image_description": "Infographic showing what $100 could buy in 2000 vs 2010 vs 2020, with shrinking shopping baskets.",
    },
    {
        "question": "How do credit cards work and what are the risks?",
        "answer": "Credit cards allow you to borrow money for purchases up to a credit limit. If you don't pay the full balance, interest accrues at high APR rates.",
        "image_description": "Flowchart: purchase with credit card -> pay full balance (no interest) vs pay minimum (interest accumulates).",
    },
    {
        "question": "What is diversification in investing?",
        "answer": "Diversification means spreading investments across different asset classes to reduce risk. Don't put all your eggs in one basket.",
        "image_description": "Two portfolio pie charts: concentrated (high risk) vs diversified (lower risk), with volatility comparison.",
    },
    {
        "question": "How does a mortgage work?",
        "answer": "A mortgage is a loan to buy property. You pay monthly installments of principal and interest over 15-30 years, with the property as collateral.",
        "image_description": "Amortization chart showing how payments shift from mostly interest to mostly principal over 30 years.",
    },
]


def load_clip_model() -> tuple:
    """
    Carrega modelo OpenCLIP (ViT-B-32) para analise de embeddings.

    Returns:
        Tupla (model, preprocess, tokenizer)
    """
    import open_clip

    console.print("  Carregando OpenCLIP ViT-B-32...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    console.print("  Modelo carregado com sucesso.")
    return model, preprocess, tokenizer


def encode_texts(
    texts: list[str],
    model,
    tokenizer,
) -> np.ndarray:
    """Codifica textos usando OpenCLIP e retorna embeddings normalizados."""
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    tokens = tokenizer(texts)
    # Trunca para o limite do modelo (77 tokens)
    tokens = tokens[:, :77]
    tokens = tokens.to(device)

    with torch.no_grad():
        embeddings = model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return embeddings.cpu().numpy()


def compute_similarities(
    qa_pairs: list[QAPair],
    model,
    tokenizer,
) -> pd.DataFrame:
    """
    Calcula similaridades text-text e text-image para cada QA pair.

    Returns:
        DataFrame com colunas: id, country, topic, text_text_sim, text_image_sim
    """
    records: list[dict] = []

    for qa in qa_pairs:
        # Text-text: question <-> answer
        embeddings_tt = encode_texts([qa.question, qa.answer], model, tokenizer)
        tt_sim = float(np.dot(embeddings_tt[0], embeddings_tt[1]))

        # Text-image: question <-> image_description (proxy)
        ti_sim = 0.0
        if qa.image_description:
            embeddings_ti = encode_texts(
                [qa.question, qa.image_description], model, tokenizer
            )
            ti_sim = float(np.dot(embeddings_ti[0], embeddings_ti[1]))

        records.append({
            "id": qa.id,
            "country": qa.country,
            "topic": qa.topic,
            "difficulty": qa.difficulty,
            "has_image": qa.image_description is not None,
            "text_text_sim": tt_sim,
            "text_image_sim": ti_sim if qa.image_description else np.nan,
        })

    return pd.DataFrame(records)


def compute_english_baseline(model, tokenizer) -> pd.DataFrame:
    """Calcula similaridades para os pares de referencia em ingles."""
    records: list[dict] = []

    for i, pair in enumerate(ENGLISH_REFERENCE_PAIRS):
        embeddings_tt = encode_texts(
            [pair["question"], pair["answer"]], model, tokenizer
        )
        tt_sim = float(np.dot(embeddings_tt[0], embeddings_tt[1]))

        embeddings_ti = encode_texts(
            [pair["question"], pair["image_description"]], model, tokenizer
        )
        ti_sim = float(np.dot(embeddings_ti[0], embeddings_ti[1]))

        records.append({
            "id": f"EN-REF-{i+1:03d}",
            "country": "EN",
            "topic": "reference",
            "difficulty": "reference",
            "has_image": True,
            "text_text_sim": tt_sim,
            "text_image_sim": ti_sim,
        })

    return pd.DataFrame(records)


def plot_boxplot_by_country(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Gera box plot de distribuicao de similaridade por pais."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Text-text similarity
    data_tt = df[["country", "text_text_sim"]].copy()
    sns.boxplot(data=data_tt, x="country", y="text_text_sim", ax=axes[0],
                order=["EN", "BR", "PT", "AO"], palette="Set2")
    axes[0].set_title("Text-Text Similarity by Country", fontsize=13)
    axes[0].set_xlabel("Country / Variant")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].axhline(y=data_tt[data_tt["country"] == "EN"]["text_text_sim"].mean(),
                     color="red", linestyle="--", alpha=0.5, label="EN baseline")
    axes[0].legend()

    # Text-image similarity
    data_ti = df[df["text_image_sim"].notna()][["country", "text_image_sim"]].copy()
    sns.boxplot(data=data_ti, x="country", y="text_image_sim", ax=axes[1],
                order=["EN", "BR", "PT", "AO"], palette="Set2")
    axes[1].set_title("Text-Image Similarity by Country", fontsize=13)
    axes[1].set_xlabel("Country / Variant")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].axhline(y=data_ti[data_ti["country"] == "EN"]["text_image_sim"].mean(),
                     color="red", linestyle="--", alpha=0.5, label="EN baseline")
    axes[1].legend()

    plt.tight_layout()
    path = output_dir / "boxplot_similarity_by_country.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Box plot salvo: {path.name}")


def plot_bar_country_modality(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Gera bar chart de similaridade media por pais x modalidade."""
    countries = ["EN", "BR", "PT", "AO"]
    tt_means = []
    ti_means = []

    for c in countries:
        subset = df[df["country"] == c]
        tt_means.append(subset["text_text_sim"].mean())
        ti_subset = subset[subset["text_image_sim"].notna()]
        ti_means.append(ti_subset["text_image_sim"].mean() if len(ti_subset) > 0 else 0)

    x = np.arange(len(countries))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, tt_means, width, label="Text-Text", color="#2196F3")
    bars2 = ax.bar(x + width / 2, ti_means, width, label="Text-Image", color="#FF9800")

    ax.set_xlabel("Country / Variant")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Average Similarity: Country x Modality")
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.legend()

    # Valores sobre as barras
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    path = output_dir / "bar_similarity_country_modality.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Bar chart salvo: {path.name}")


def plot_degradation_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Gera heatmap de degradacao relativa ao baseline em ingles."""
    en_tt_mean = df[df["country"] == "EN"]["text_text_sim"].mean()
    en_ti_mean = df[df["country"] == "EN"]["text_image_sim"].dropna().mean()

    countries = ["BR", "PT", "AO"]
    degradation_data: dict[str, list[float]] = {"Text-Text": [], "Text-Image": []}

    for c in countries:
        subset = df[df["country"] == c]
        tt_mean = subset["text_text_sim"].mean()
        ti_subset = subset[subset["text_image_sim"].notna()]
        ti_mean = ti_subset["text_image_sim"].mean() if len(ti_subset) > 0 else 0

        # Degradacao como porcentagem de queda relativa ao ingles
        degradation_data["Text-Text"].append(
            ((en_tt_mean - tt_mean) / en_tt_mean * 100) if en_tt_mean != 0 else 0
        )
        degradation_data["Text-Image"].append(
            ((en_ti_mean - ti_mean) / en_ti_mean * 100) if en_ti_mean != 0 else 0
        )

    deg_df = pd.DataFrame(degradation_data, index=countries)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        deg_df,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Degradation (%)"},
        linewidths=0.5,
    )
    ax.set_title("Embedding Similarity Degradation vs English Baseline (%)")
    ax.set_ylabel("Portuguese Variant")
    ax.set_xlabel("Modality Comparison")

    plt.tight_layout()
    path = output_dir / "heatmap_degradation_vs_english.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Heatmap salvo: {path.name}")


def run_statistical_tests(df: pd.DataFrame) -> dict[str, Any]:
    """
    Executa teste de Kruskal-Wallis para comparar distribuicoes
    de similaridade entre paises lusófonos.

    Returns:
        Dicionario com resultados dos testes.
    """
    from typing import Any

    results: dict[str, Any] = {}
    luso_df = df[df["country"].isin(["BR", "PT", "AO"])]

    # Text-text
    groups_tt = [
        group["text_text_sim"].values
        for _, group in luso_df.groupby("country")
    ]
    if len(groups_tt) >= 2 and all(len(g) > 0 for g in groups_tt):
        stat_tt, p_tt = stats.kruskal(*groups_tt)
        results["text_text"] = {"statistic": stat_tt, "p_value": p_tt}
    else:
        results["text_text"] = {"statistic": np.nan, "p_value": np.nan}

    # Text-image (apenas pares com imagem)
    luso_ti = luso_df[luso_df["text_image_sim"].notna()]
    groups_ti = [
        group["text_image_sim"].values
        for _, group in luso_ti.groupby("country")
    ]
    if len(groups_ti) >= 2 and all(len(g) > 0 for g in groups_ti):
        stat_ti, p_ti = stats.kruskal(*groups_ti)
        results["text_image"] = {"statistic": stat_ti, "p_value": p_ti}
    else:
        results["text_image"] = {"statistic": np.nan, "p_value": np.nan}

    return results


def print_summary_table(
    df: pd.DataFrame,
    stat_results: dict,
) -> None:
    """Imprime tabela resumo com Rich."""
    table = Table(title="Lusofono-CLIP-Bench: Resumo de Similaridade", show_lines=True)
    table.add_column("Pais", style="bold cyan")
    table.add_column("N", justify="right")
    table.add_column("Text-Text (mean)", justify="right")
    table.add_column("Text-Text (std)", justify="right")
    table.add_column("Text-Image (mean)", justify="right")
    table.add_column("Text-Image (std)", justify="right")

    for country in ["EN", "BR", "PT", "AO"]:
        subset = df[df["country"] == country]
        n = len(subset)
        tt_mean = subset["text_text_sim"].mean()
        tt_std = subset["text_text_sim"].std()
        ti_subset = subset[subset["text_image_sim"].notna()]
        ti_mean = ti_subset["text_image_sim"].mean() if len(ti_subset) > 0 else 0
        ti_std = ti_subset["text_image_sim"].std() if len(ti_subset) > 0 else 0

        label = {"EN": "English (ref)", "BR": "Brasil", "PT": "Portugal", "AO": "Angola"}
        table.add_row(
            label.get(country, country),
            str(n),
            f"{tt_mean:.4f}",
            f"{tt_std:.4f}",
            f"{ti_mean:.4f}",
            f"{ti_std:.4f}",
        )

    console.print(table)

    # Testes estatisticos
    console.print("\n[bold]Testes Estatisticos (Kruskal-Wallis entre BR, PT, AO):[/bold]")
    for modality, res in stat_results.items():
        sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "ns"
        console.print(
            f"  {modality}: H={res['statistic']:.4f}, p={res['p_value']:.6f} {sig}"
        )


def main() -> None:
    """Executa a analise completa de embeddings CLIP."""
    console.print("\n[bold magenta]===== Lusofono-CLIP-Bench =====[/bold magenta]\n")

    output_dir = _PROJECT_ROOT / "embeddings" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar dados
    console.print("[bold green]1. Carregando dataset...[/bold green]")
    qa_pairs = load_all_countries(_PROJECT_ROOT / "dataset")
    if not qa_pairs:
        console.print("[red]Dataset nao encontrado. Execute: python -m dataset.generator[/red]")
        sys.exit(1)
    console.print(f"  {len(qa_pairs)} QA pairs carregados.")

    # 2. Carregar modelo CLIP
    console.print("\n[bold green]2. Carregando modelo CLIP...[/bold green]")
    model, _preprocess, tokenizer = load_clip_model()

    # 3. Calcular similaridades para portugues
    console.print("\n[bold green]3. Calculando similaridades (portugues)...[/bold green]")
    pt_results = compute_similarities(qa_pairs, model, tokenizer)
    console.print(f"  {len(pt_results)} pares processados.")

    # 4. Calcular baseline ingles
    console.print("\n[bold green]4. Calculando baseline ingles...[/bold green]")
    en_results = compute_english_baseline(model, tokenizer)
    console.print(f"  {len(en_results)} pares de referencia processados.")

    # 5. Combinar resultados
    all_results = pd.concat([en_results, pt_results], ignore_index=True)

    # 6. Salvar dados brutos
    all_results.to_csv(output_dir / "clip_similarities.csv", index=False)
    console.print(f"\n  Dados brutos salvos em {output_dir / 'clip_similarities.csv'}")

    # 7. Gerar visualizacoes
    console.print("\n[bold green]5. Gerando visualizacoes...[/bold green]")
    plot_boxplot_by_country(all_results, output_dir)
    plot_bar_country_modality(all_results, output_dir)
    plot_degradation_heatmap(all_results, output_dir)

    # 8. Testes estatisticos
    console.print("\n[bold green]6. Executando testes estatisticos...[/bold green]")
    stat_results = run_statistical_tests(all_results)

    # 9. Resumo
    console.print("\n[bold green]7. Resumo final:[/bold green]")
    print_summary_table(all_results, stat_results)

    console.print(f"\n[bold magenta]Resultados salvos em: {output_dir}/[/bold magenta]\n")


if __name__ == "__main__":
    main()

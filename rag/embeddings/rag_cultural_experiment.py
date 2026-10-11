"""
Experimento: Impacto de Cultural Notes no RAG.

Compara retrieval com corpus BASE vs BASE+CULTURAL:
- BASE: corpus atual (answers + image_descriptions)
- BASE+CULTURAL: corpus estendido incluindo cultural_notes como documentos adicionais

Todos os 999 pares possuem cultural_notes (100% cobertura).

Uso:
    python -m embeddings.rag_cultural_experiment
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
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import load_all_countries, QAPair

console = Console()

K_VALUES = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------

def build_corpus_base(qa_pairs: list[QAPair]) -> list[dict]:
    """Corpus atual: answers + image_descriptions."""
    docs = []
    for qa in qa_pairs:
        docs.append({
            "id": f"{qa.id}_answer",
            "qa_id": qa.id,
            "text": qa.answer,
            "type": "answer",
            "country": qa.country,
            "topic": qa.topic,
        })
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


def build_corpus_with_cultural(qa_pairs: list[QAPair]) -> list[dict]:
    """Corpus estendido: answers + image_descriptions + cultural_notes."""
    docs = build_corpus_base(qa_pairs)
    cultural_count = 0
    for qa in qa_pairs:
        if qa.cultural_notes:
            docs.append({
                "id": f"{qa.id}_cultural",
                "qa_id": qa.id,
                "text": qa.cultural_notes,
                "type": "cultural_notes",
                "country": qa.country,
                "topic": qa.topic,
            })
            cultural_count += 1
    return docs, cultural_count


def build_corpus_full_multimodal(qa_pairs: list[QAPair]) -> list[dict]:
    """Corpus FULL: answers + image_descriptions + cultural_notes + audio_transcripts."""
    docs, _ = build_corpus_with_cultural(qa_pairs)
    audio_count = 0
    for qa in qa_pairs:
        if qa.audio_transcript:
            docs.append({
                "id": f"{qa.id}_audio",
                "qa_id": qa.id,
                "text": qa.audio_transcript,
                "type": "audio_transcript",
                "country": qa.country,
                "topic": qa.topic,
            })
            audio_count += 1
    return docs, audio_count


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


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
    return encode, "LoRA+CL"


def load_e5_multilingual():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    def encode(texts: list[str]) -> np.ndarray:
        prefixed = [f"query: {t}" for t in texts]
        return model.encode(prefixed, normalize_embeddings=True)
    return encode, "e5-multilingual"


# ---------------------------------------------------------------------------
# Batch encoding & Retrieval evaluation
# ---------------------------------------------------------------------------

def encode_batch(texts: list[str], encode_fn, batch_size: int = 64) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = encode_fn(batch)
        all_embs.append(emb)
    return np.vstack(all_embs)


def evaluate_retrieval(
    qa_pairs: list[QAPair],
    corpus: list[dict],
    encode_fn,
    k_values: list[int] = K_VALUES,
    batch_size: int = 64,
) -> pd.DataFrame:
    corpus_texts = [d["text"] for d in corpus]
    corpus_embs = encode_batch(corpus_texts, encode_fn, batch_size)

    qa_to_corpus = {}
    for idx, doc in enumerate(corpus):
        qa_id = doc["qa_id"]
        if qa_id not in qa_to_corpus:
            qa_to_corpus[qa_id] = []
        qa_to_corpus[qa_id].append(idx)

    queries = [qa.question for qa in qa_pairs]
    query_embs = encode_batch(queries, encode_fn, batch_size)

    max_k = max(k_values)
    records = []

    for i, qa in enumerate(qa_pairs):
        q_emb = query_embs[i:i+1]
        sims = (q_emb @ corpus_embs.T).flatten()
        top_indices = np.argsort(sims)[::-1][:max_k]
        gt_indices = set(qa_to_corpus.get(qa.id, []))

        row = {
            "qa_id": qa.id,
            "country": qa.country,
            "topic": qa.topic,
            "difficulty": qa.difficulty,
        }

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

        # Track retrieved types
        retrieved_types = [corpus[idx]["type"] for idx in top_indices[:5]]
        row["retrieved_types"] = ",".join(retrieved_types)

        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Comparison & Analysis
# ---------------------------------------------------------------------------

def compare_corpora(
    df_base: pd.DataFrame,
    df_ext: pd.DataFrame,
    model_name: str,
    corpus_label: str,
) -> list[dict]:
    """Compare BASE vs extended corpus results."""
    console.print(f"\n[bold cyan]Comparacao: BASE vs {corpus_label} ({model_name})[/bold cyan]")

    table = Table(title=f"Retrieval: BASE vs {corpus_label} ({model_name})", show_lines=True)
    table.add_column("Metrica", style="bold")
    table.add_column("BASE", justify="right")
    table.add_column(corpus_label, justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Delta %", justify="right")

    rows = []
    for metric_name, col in [("MRR", "mrr")] + [(f"Hit@{k}", f"hit@{k}") for k in K_VALUES]:
        v_base = df_base[col].mean()
        v_ext = df_ext[col].mean()
        delta = v_ext - v_base
        pct = (delta / v_base * 100) if v_base > 0 else 0
        style = "[green]" if delta > 0 else "[red]" if delta < 0 else ""
        table.add_row(metric_name, f"{v_base:.4f}", f"{v_ext:.4f}",
                      f"{style}{delta:+.4f}", f"{style}{pct:+.2f}%")
        rows.append({
            "model": model_name, "corpus": corpus_label, "subset": "all",
            "metric": metric_name, "base": v_base, "extended": v_ext,
            "delta": delta, "delta_pct": pct,
        })
    console.print(table)

    # Per-country
    table = Table(title=f"MRR por Pais: BASE vs {corpus_label} ({model_name})", show_lines=True)
    table.add_column("Pais", style="bold")
    table.add_column("BASE", justify="right")
    table.add_column(corpus_label, justify="right")
    table.add_column("Delta", justify="right")

    for c in ["BR", "PT", "AO"]:
        v_base = df_base[df_base["country"] == c]["mrr"].mean()
        v_ext = df_ext[df_ext["country"] == c]["mrr"].mean()
        delta = v_ext - v_base
        style = "[green]" if delta > 0 else "[red]" if delta < 0 else ""
        table.add_row(c, f"{v_base:.4f}", f"{v_ext:.4f}", f"{style}{delta:+.4f}")
        rows.append({
            "model": model_name, "corpus": corpus_label, "subset": f"country_{c}",
            "metric": "MRR", "base": v_base, "extended": v_ext,
            "delta": delta, "delta_pct": (delta / v_base * 100) if v_base > 0 else 0,
        })
    console.print(table)

    # Per-topic
    table = Table(title=f"MRR por Topico: BASE vs {corpus_label} ({model_name})", show_lines=True)
    table.add_column("Topico", style="bold")
    table.add_column("BASE", justify="right")
    table.add_column(corpus_label, justify="right")
    table.add_column("Delta", justify="right")

    for t in sorted(df_base["topic"].unique()):
        v_base = df_base[df_base["topic"] == t]["mrr"].mean()
        v_ext = df_ext[df_ext["topic"] == t]["mrr"].mean()
        delta = v_ext - v_base
        style = "[green]" if delta > 0 else "[red]" if delta < 0 else ""
        table.add_row(t, f"{v_base:.4f}", f"{v_ext:.4f}", f"{style}{delta:+.4f}")
        rows.append({
            "model": model_name, "corpus": corpus_label, "subset": f"topic_{t}",
            "metric": "MRR", "base": v_base, "extended": v_ext,
            "delta": delta, "delta_pct": (delta / v_base * 100) if v_base > 0 else 0,
        })
    console.print(table)

    # Wilcoxon test
    stat_w, p_w = stats.wilcoxon(df_base["mrr"].values, df_ext["mrr"].values, alternative="two-sided")
    console.print(f"\n  Wilcoxon signed-rank: W={stat_w:.2f}, p={p_w:.6f}")
    sig = "SIGNIFICATIVA" if p_w < 0.05 else "NAO significativa"
    color = "green" if p_w < 0.05 else "yellow"
    console.print(f"  [{color}]Diferenca {sig} (p {'<' if p_w < 0.05 else '>='} 0.05)[/{color}]")

    rows.append({
        "model": model_name, "corpus": corpus_label, "subset": "wilcoxon",
        "metric": "W_statistic", "base": stat_w, "extended": p_w,
        "delta": 0, "delta_pct": 0,
    })

    return rows


def analyze_retrieved_types(df: pd.DataFrame, model_name: str, corpus_label: str):
    """Count how often cultural_notes appear in top-5."""
    total = len(df)
    cultural_in_top5 = sum(1 for _, row in df.iterrows() if "cultural_notes" in row["retrieved_types"])
    cultural_rank1 = sum(1 for _, row in df.iterrows() if row["retrieved_types"].startswith("cultural_notes"))

    console.print(f"\n[bold]Tipos recuperados ({model_name}, {corpus_label}):[/bold]")
    console.print(f"  cultural_notes no top-5: {cultural_in_top5}/{total} ({100*cultural_in_top5/total:.1f}%)")
    console.print(f"  cultural_notes em rank 1: {cultural_rank1}/{total} ({100*cultural_rank1/total:.1f}%)")

    # Count by type
    type_counts = {}
    for _, row in df.iterrows():
        for t in row["retrieved_types"].split(","):
            type_counts[t] = type_counts.get(t, 0) + 1
    total_slots = total * 5
    console.print(f"  Distribuicao no top-5 ({total_slots} slots):")
    for t in sorted(type_counts, key=type_counts.get, reverse=True):
        console.print(f"    {t}: {type_counts[t]} ({100*type_counts[t]/total_slots:.1f}%)")


def plot_results(all_rows: pd.DataFrame, output_dir: Path):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Plot 1: Overall metrics BASE vs CULTURAL vs FULL (LoRA+CL) ---
    ax = axes[0, 0]
    lora_data = all_rows[(all_rows["model"] == "LoRA+CL") & (all_rows["subset"] == "all")]
    metrics = ["MRR", "Hit@1", "Hit@3", "Hit@5", "Hit@10"]
    x = np.arange(len(metrics))
    width = 0.25

    corpora = lora_data["corpus"].unique()
    colors = {"BASE+CULTURAL": "#4ECDC4", "FULL": "#45B7D1"}
    # Base values from BASE+CULTURAL (base column)
    base_vals = lora_data[lora_data["corpus"] == corpora[0]][["metric", "base"]].set_index("metric")

    bars_base = ax.bar(x - width, [base_vals.loc[m, "base"] for m in metrics], width,
                       label="BASE", color="#FF6B6B", edgecolor="black", linewidth=0.5)
    for i, corpus_name in enumerate(corpora):
        sub = lora_data[lora_data["corpus"] == corpus_name][["metric", "extended"]].set_index("metric")
        vals = [sub.loc[m, "extended"] for m in metrics]
        bars = ax.bar(x + width * i, vals, width, label=corpus_name,
                      color=colors.get(corpus_name, "#95A5A6"), edgecolor="black", linewidth=0.5)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    for bar in bars_base:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("LoRA+CL: BASE vs CULTURAL vs FULL")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Plot 2: Per-country MRR (LoRA+CL, BASE+CULTURAL) ---
    ax = axes[0, 1]
    countries = ["BR", "PT", "AO"]
    country_data = all_rows[
        (all_rows["model"] == "LoRA+CL") &
        (all_rows["subset"].str.startswith("country_")) &
        (all_rows["corpus"] == "BASE+CULTURAL")
    ]
    x = np.arange(len(countries))
    width = 0.35
    base_vals = [country_data[country_data["subset"] == f"country_{c}"]["base"].values[0] for c in countries]
    ext_vals = [country_data[country_data["subset"] == f"country_{c}"]["extended"].values[0] for c in countries]

    bars1 = ax.bar(x - width/2, base_vals, width, label="BASE", color="#FF6B6B", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, ext_vals, width, label="BASE+CULTURAL", color="#4ECDC4", edgecolor="black", linewidth=0.5)
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.annotate(f"{bar.get_height():.4f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.set_ylabel("MRR")
    ax.set_title("MRR por Pais: LoRA+CL")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # --- Plot 3: Per-topic MRR delta (LoRA+CL, BASE+CULTURAL) ---
    ax = axes[1, 0]
    topic_data = all_rows[
        (all_rows["model"] == "LoRA+CL") &
        (all_rows["subset"].str.startswith("topic_")) &
        (all_rows["corpus"] == "BASE+CULTURAL")
    ]
    topics = sorted(topic_data["subset"].str.replace("topic_", "").values)
    deltas = [topic_data[topic_data["subset"] == f"topic_{t}"]["delta_pct"].values[0] for t in topics]
    colors_bar = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]
    bars = ax.bar(topics, deltas, color=colors_bar, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, deltas):
        ax.annotate(f"{val:+.2f}%",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3 if val >= 0 else -12), textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Delta MRR (%)")
    ax.set_title("Impacto por Topico: BASE+CULTURAL (LoRA+CL)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Plot 4: Model comparison (both models, both corpora) ---
    ax = axes[1, 1]
    overall = all_rows[(all_rows["subset"] == "all") & (all_rows["metric"] == "MRR")]
    models = overall["model"].unique()
    corpora_names = overall["corpus"].unique()
    x = np.arange(len(models))
    width = 0.25
    corpus_colors = {"BASE+CULTURAL": "#4ECDC4", "FULL": "#45B7D1"}

    # Base bars
    base_vals = []
    for m in models:
        sub = overall[(overall["model"] == m)]
        base_vals.append(sub.iloc[0]["base"])
    bars_b = ax.bar(x - width, base_vals, width, label="BASE", color="#FF6B6B", edgecolor="black", linewidth=0.5)

    for i, cn in enumerate(corpora_names):
        ext_vals = []
        for m in models:
            sub = overall[(overall["model"] == m) & (overall["corpus"] == cn)]
            ext_vals.append(sub.iloc[0]["extended"] if len(sub) > 0 else 0)
        bars = ax.bar(x + width * i, ext_vals, width, label=cn,
                      color=corpus_colors.get(cn, "#95A5A6"), edgecolor="black", linewidth=0.5)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.4f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    for bar in bars_b:
        ax.annotate(f"{bar.get_height():.4f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("MRR")
    ax.set_title("MRR: Comparacao entre Modelos e Corpora")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("FinEduBench-Luso: Impacto de Cultural Notes e Corpus Multimodal Completo",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "rag_cultural_impact.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Chart salvo: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    console.print("\n[bold magenta]===== Experimento: Cultural Notes + Corpus Multimodal Completo =====[/bold magenta]\n")

    output_dir = _PROJECT_ROOT / "embeddings" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_pairs = load_all_countries(_PROJECT_ROOT / "dataset")
    console.print(f"  {len(qa_pairs)} QA pairs carregados.")

    # Build corpora
    corpus_base = build_corpus_base(qa_pairs)
    corpus_cultural, n_cultural = build_corpus_with_cultural(qa_pairs)
    corpus_full, n_audio = build_corpus_full_multimodal(qa_pairs)

    console.print(f"  Corpus BASE: {len(corpus_base)} documentos (answers + image_desc)")
    console.print(f"  Corpus BASE+CULTURAL: {len(corpus_cultural)} documentos (+{n_cultural} cultural_notes)")
    console.print(f"  Corpus FULL: {len(corpus_full)} documentos (+{n_audio} audio_transcripts)")

    # Cultural notes stats
    cn_lens = [len(q.cultural_notes) for q in qa_pairs if q.cultural_notes]
    console.print(f"\n  Cultural notes: {len(cn_lens)}/999 pares (100%)")
    console.print(f"  Comprimento medio: {np.mean(cn_lens):.0f} chars")

    model_loaders = [
        ("LoRA+CL", load_openclip_lora_cl),
        ("e5-multilingual", load_e5_multilingual),
    ]

    all_rows = []

    for model_name, loader_fn in model_loaders:
        console.print(f"\n[bold green]{'='*20} {model_name} {'='*20}[/bold green]")
        encode_fn, _ = loader_fn()
        if encode_fn is None:
            console.print(f"  [yellow]Pulando {model_name}[/yellow]")
            continue

        # Evaluate BASE
        console.print(f"  [bold]Avaliando corpus BASE...[/bold]")
        t0 = time.time()
        df_base = evaluate_retrieval(qa_pairs, corpus_base, encode_fn)
        console.print(f"    MRR={df_base['mrr'].mean():.4f} | Hit@5={df_base['hit@5'].mean():.3f} | {time.time()-t0:.1f}s")

        # Evaluate BASE+CULTURAL
        console.print(f"  [bold]Avaliando corpus BASE+CULTURAL...[/bold]")
        t0 = time.time()
        df_cultural = evaluate_retrieval(qa_pairs, corpus_cultural, encode_fn)
        console.print(f"    MRR={df_cultural['mrr'].mean():.4f} | Hit@5={df_cultural['hit@5'].mean():.3f} | {time.time()-t0:.1f}s")

        # Evaluate FULL (all modalities)
        console.print(f"  [bold]Avaliando corpus FULL (todos campos)...[/bold]")
        t0 = time.time()
        df_full = evaluate_retrieval(qa_pairs, corpus_full, encode_fn)
        console.print(f"    MRR={df_full['mrr'].mean():.4f} | Hit@5={df_full['hit@5'].mean():.3f} | {time.time()-t0:.1f}s")

        # Compare BASE vs CULTURAL
        rows = compare_corpora(df_base, df_cultural, model_name, "BASE+CULTURAL")
        all_rows.extend(rows)

        # Compare BASE vs FULL
        rows = compare_corpora(df_base, df_full, model_name, "FULL")
        all_rows.extend(rows)

        # Analyze retrieved types for CULTURAL corpus
        analyze_retrieved_types(df_cultural, model_name, "BASE+CULTURAL")
        analyze_retrieved_types(df_full, model_name, "FULL")

        # Equity analysis: check if cultural notes reduce inter-country variance
        console.print(f"\n[bold]Analise de equidade inter-pais ({model_name}):[/bold]")
        for label, df in [("BASE", df_base), ("BASE+CULTURAL", df_cultural), ("FULL", df_full)]:
            mrrs = [df[df["country"] == c]["mrr"].mean() for c in ["BR", "PT", "AO"]]
            variance = np.var(mrrs)
            spread = max(mrrs) - min(mrrs)
            # Kruskal-Wallis
            groups = [df[df["country"] == c]["mrr"].values for c in ["BR", "PT", "AO"]]
            h_stat, p_val = stats.kruskal(*groups)
            console.print(f"  {label:16s} | var={variance:.6f} | spread={spread:.4f} | KW: H={h_stat:.2f}, p={p_val:.4f}")

        del encode_fn
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Save and plot
    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(output_dir / "rag_cultural_comparison.csv", index=False)
    console.print(f"\n  Resultados salvos em rag_cultural_comparison.csv")

    console.print("\n[bold green]Gerando visualizacoes...[/bold green]")
    plot_results(df_results, output_dir)

    console.print(f"\n[bold magenta]Experimento concluido![/bold magenta]\n")


if __name__ == "__main__":
    main()

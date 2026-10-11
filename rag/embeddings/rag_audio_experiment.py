"""
Experimento: Impacto de Audio Transcripts no RAG.

Compara retrieval com corpus TEXT-ONLY vs TEXT+AUDIO.
- TEXT-ONLY: corpus atual (answers + image_descriptions)
- TEXT+AUDIO: corpus estendido incluindo audio_transcript como documentos adicionais

Avalia impacto no MRR, Hit@k, Precision@k para os 57 QA pairs que possuem audio,
e para o dataset completo (999 pares).

Uso:
    python -m embeddings.rag_audio_experiment
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
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

def build_corpus_text_only(qa_pairs: list[QAPair]) -> list[dict]:
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


def build_corpus_with_audio(qa_pairs: list[QAPair]) -> list[dict]:
    """Corpus estendido: answers + image_descriptions + audio_transcripts."""
    docs = build_corpus_text_only(qa_pairs)
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
# Batch encoding
# ---------------------------------------------------------------------------

def encode_batch(texts: list[str], encode_fn, batch_size: int = 64) -> np.ndarray:
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
    """Evaluate retrieval for each question against the corpus."""
    corpus_texts = [d["text"] for d in corpus]
    corpus_embs = encode_batch(corpus_texts, encode_fn, batch_size)

    # qa_id -> corpus indices (ground truth)
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
            "has_audio": qa.audio_transcript is not None and len(qa.audio_transcript) > 0,
        }

        # MRR
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

        # Track which document types were retrieved
        retrieved_types = []
        for idx in top_indices[:5]:
            retrieved_types.append(corpus[idx]["type"])
        row["retrieved_types"] = ",".join(retrieved_types)

        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def audio_distribution_analysis(qa_pairs: list[QAPair], output_dir: Path):
    """Analyze and visualize audio transcript distribution."""
    audio_pairs = [q for q in qa_pairs if q.audio_transcript]

    console.print(f"\n[bold cyan]Distribuicao de Audio Transcripts[/bold cyan]")
    console.print(f"  Total QA pairs: {len(qa_pairs)}")
    console.print(f"  Com audio_transcript: {len(audio_pairs)} ({100*len(audio_pairs)/len(qa_pairs):.1f}%)")

    # By country
    table = Table(title="Audio Transcripts por Pais", show_lines=True)
    table.add_column("Pais", style="bold")
    table.add_column("Com Audio", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("%", justify="right")

    for c in ["BR", "PT", "AO"]:
        total = len([q for q in qa_pairs if q.country == c])
        audio = len([q for q in audio_pairs if q.country == c])
        table.add_row(c, str(audio), str(total), f"{100*audio/total:.1f}%")
    console.print(table)

    # By topic
    table = Table(title="Audio Transcripts por Topico", show_lines=True)
    table.add_column("Topico", style="bold")
    table.add_column("Com Audio", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("%", justify="right")

    topics = sorted(set(q.topic for q in qa_pairs))
    for t in topics:
        total = len([q for q in qa_pairs if q.topic == t])
        audio = len([q for q in audio_pairs if q.topic == t])
        table.add_row(t, str(audio), str(total), f"{100*audio/total:.1f}%")
    console.print(table)

    # Average transcript length
    lengths = [len(q.audio_transcript) for q in audio_pairs]
    answer_lengths = [len(q.answer) for q in audio_pairs]
    console.print(f"\n  Comprimento medio audio_transcript: {np.mean(lengths):.0f} chars")
    console.print(f"  Comprimento medio answer (mesmos pares): {np.mean(answer_lengths):.0f} chars")
    console.print(f"  Ratio audio/answer: {np.mean(lengths)/np.mean(answer_lengths):.2f}x")

    # Plot distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # By country
    ax = axes[0]
    countries = ["BR", "PT", "AO"]
    audio_counts = [len([q for q in audio_pairs if q.country == c]) for c in countries]
    total_counts = [len([q for q in qa_pairs if q.country == c]) for c in countries]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(countries, audio_counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val, total in zip(bars, audio_counts, total_counts):
        ax.annotate(f"{val}\n({100*val/total:.0f}%)",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    ax.set_ylabel("Numero de QA Pairs")
    ax.set_title("Audio Transcripts por Pais")
    ax.grid(True, alpha=0.3, axis="y")

    # By topic
    ax = axes[1]
    topic_counts = [len([q for q in audio_pairs if q.topic == t]) for t in topics]
    topic_colors = plt.cm.Set2(np.linspace(0, 1, len(topics)))
    bars = ax.bar(topics, topic_counts, color=topic_colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, topic_counts):
        ax.annotate(str(val), xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    ax.set_ylabel("Numero de QA Pairs")
    ax.set_title("Audio Transcripts por Topico")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    # Length comparison
    ax = axes[2]
    ax.hist(lengths, bins=15, alpha=0.7, color="#4ECDC4", label="Audio Transcript", edgecolor="black")
    ax.hist(answer_lengths, bins=15, alpha=0.5, color="#FF6B6B", label="Answer (texto)", edgecolor="black")
    ax.set_xlabel("Comprimento (caracteres)")
    ax.set_ylabel("Frequencia")
    ax.set_title("Distribuicao de Comprimento")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("FinEduBench-Luso: Audio Transcripts (57/999 pares)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "rag_audio_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Chart salvo: {path.name}")


def compare_retrieval_results(
    df_text: pd.DataFrame,
    df_audio: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Compare TEXT-ONLY vs TEXT+AUDIO retrieval results."""
    console.print(f"\n[bold cyan]Comparacao: TEXT-ONLY vs TEXT+AUDIO ({model_name})[/bold cyan]")

    # Overall metrics
    table = Table(title=f"Retrieval: TEXT vs TEXT+AUDIO ({model_name})", show_lines=True)
    table.add_column("Metrica", style="bold")
    table.add_column("TEXT-ONLY", justify="right")
    table.add_column("TEXT+AUDIO", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Delta %", justify="right")

    comparison_rows = []
    for metric_name, col in [("MRR", "mrr")] + [(f"Hit@{k}", f"hit@{k}") for k in K_VALUES]:
        v_text = df_text[col].mean()
        v_audio = df_audio[col].mean()
        delta = v_audio - v_text
        pct = (delta / v_text * 100) if v_text > 0 else 0
        style = "[green]" if delta > 0 else "[red]" if delta < 0 else ""
        table.add_row(
            metric_name,
            f"{v_text:.4f}",
            f"{v_audio:.4f}",
            f"{style}{delta:+.4f}",
            f"{style}{pct:+.2f}%",
        )
        comparison_rows.append({
            "model": model_name,
            "subset": "all",
            "metric": metric_name,
            "text_only": v_text,
            "text_audio": v_audio,
            "delta": delta,
            "delta_pct": pct,
        })
    console.print(table)

    # Subset: only QA pairs WITH audio
    audio_text = df_text[df_text["has_audio"]]
    audio_audio = df_audio[df_audio["has_audio"]]

    table = Table(title=f"Subset Audio-Only (N={len(audio_text)}): {model_name}", show_lines=True)
    table.add_column("Metrica", style="bold")
    table.add_column("TEXT-ONLY", justify="right")
    table.add_column("TEXT+AUDIO", justify="right")
    table.add_column("Delta", justify="right")

    for metric_name, col in [("MRR", "mrr")] + [(f"Hit@{k}", f"hit@{k}") for k in K_VALUES]:
        v_text = audio_text[col].mean()
        v_audio = audio_audio[col].mean()
        delta = v_audio - v_text
        pct = (delta / v_text * 100) if v_text > 0 else 0
        style = "[green]" if delta > 0 else "[red]" if delta < 0 else ""
        table.add_row(
            metric_name,
            f"{v_text:.4f}",
            f"{v_audio:.4f}",
            f"{style}{delta:+.4f}",
        )
        comparison_rows.append({
            "model": model_name,
            "subset": "audio_only",
            "metric": metric_name,
            "text_only": v_text,
            "text_audio": v_audio,
            "delta": delta,
            "delta_pct": pct,
        })
    console.print(table)

    # Per-country comparison
    table = Table(title=f"MRR por Pais: TEXT vs TEXT+AUDIO ({model_name})", show_lines=True)
    table.add_column("Pais", style="bold")
    table.add_column("TEXT-ONLY", justify="right")
    table.add_column("TEXT+AUDIO", justify="right")
    table.add_column("Delta", justify="right")

    for c in ["BR", "PT", "AO"]:
        v_text = df_text[df_text["country"] == c]["mrr"].mean()
        v_audio = df_audio[df_audio["country"] == c]["mrr"].mean()
        delta = v_audio - v_text
        style = "[green]" if delta > 0 else "[red]" if delta < 0 else ""
        table.add_row(c, f"{v_text:.4f}", f"{v_audio:.4f}", f"{style}{delta:+.4f}")
        comparison_rows.append({
            "model": model_name,
            "subset": f"country_{c}",
            "metric": "MRR",
            "text_only": v_text,
            "text_audio": v_audio,
            "delta": delta,
            "delta_pct": (delta / v_text * 100) if v_text > 0 else 0,
        })

    console.print(table)

    # Statistical test: Wilcoxon signed-rank on MRR (paired)
    console.print(f"\n[bold]Teste estatistico (Wilcoxon signed-rank): MRR TEXT vs TEXT+AUDIO[/bold]")
    stat_w, p_w = stats.wilcoxon(df_text["mrr"].values, df_audio["mrr"].values, alternative="two-sided")
    console.print(f"  W={stat_w:.2f}, p={p_w:.6f}")
    if p_w < 0.05:
        console.print("  [green]Diferenca SIGNIFICATIVA (p < 0.05)[/green]")
    else:
        console.print("  [yellow]Diferenca NAO significativa (p >= 0.05)[/yellow]")

    comparison_rows.append({
        "model": model_name,
        "subset": "statistical_test",
        "metric": "Wilcoxon_MRR",
        "text_only": stat_w,
        "text_audio": p_w,
        "delta": 0,
        "delta_pct": 0,
    })

    return pd.DataFrame(comparison_rows)


def analyze_audio_retrieval_types(df_audio: pd.DataFrame, model_name: str):
    """Analyze which document types are retrieved when audio is in corpus."""
    console.print(f"\n[bold cyan]Tipos de documentos recuperados ({model_name})[/bold cyan]")

    # Count how often audio_transcript appears in top-5
    audio_in_top5 = 0
    audio_rank_1 = 0
    total = len(df_audio)

    for _, row in df_audio.iterrows():
        types = row["retrieved_types"].split(",")
        if "audio_transcript" in types:
            audio_in_top5 += 1
        if types[0] == "audio_transcript":
            audio_rank_1 += 1

    console.print(f"  Audio transcript no top-5: {audio_in_top5}/{total} ({100*audio_in_top5/total:.1f}%)")
    console.print(f"  Audio transcript em rank 1: {audio_rank_1}/{total} ({100*audio_rank_1/total:.1f}%)")

    # For audio subset only
    audio_sub = df_audio[df_audio["has_audio"]]
    audio_in_top5_sub = 0
    for _, row in audio_sub.iterrows():
        types = row["retrieved_types"].split(",")
        if "audio_transcript" in types:
            audio_in_top5_sub += 1
    console.print(f"  Audio no top-5 (subset audio): {audio_in_top5_sub}/{len(audio_sub)} ({100*audio_in_top5_sub/len(audio_sub):.1f}%)")


def plot_comparison(all_comparisons: pd.DataFrame, output_dir: Path):
    """Plot TEXT vs TEXT+AUDIO comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Overall metrics comparison
    ax = axes[0]
    overall = all_comparisons[
        (all_comparisons["subset"] == "all") &
        (all_comparisons["model"] == "LoRA+CL")
    ]
    metrics = overall["metric"].values
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2, overall["text_only"].values, width,
                   label="TEXT-ONLY", color="#FF6B6B", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, overall["text_audio"].values, width,
                   label="TEXT+AUDIO", color="#4ECDC4", edgecolor="black", linewidth=0.5)
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Overall: TEXT vs TEXT+AUDIO (LoRA+CL)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Plot 2: Per-country MRR
    ax = axes[1]
    countries = ["BR", "PT", "AO"]
    country_data = all_comparisons[
        (all_comparisons["subset"].str.startswith("country_")) &
        (all_comparisons["model"] == "LoRA+CL")
    ]
    x = np.arange(len(countries))
    text_vals = [country_data[country_data["subset"] == f"country_{c}"]["text_only"].values[0] for c in countries]
    audio_vals = [country_data[country_data["subset"] == f"country_{c}"]["text_audio"].values[0] for c in countries]

    bars1 = ax.bar(x - width/2, text_vals, width, label="TEXT-ONLY", color="#FF6B6B", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, audio_vals, width, label="TEXT+AUDIO", color="#4ECDC4", edgecolor="black", linewidth=0.5)
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.annotate(f"{bar.get_height():.4f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.set_ylabel("MRR")
    ax.set_title("MRR por Pais (LoRA+CL)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Audio subset improvement
    ax = axes[2]
    audio_sub = all_comparisons[
        (all_comparisons["subset"] == "audio_only") &
        (all_comparisons["metric"].isin(["MRR", "Hit@1", "Hit@5"]))
    ]
    models_in_data = audio_sub["model"].unique()
    x = np.arange(3)  # MRR, Hit@1, Hit@5
    width_m = 0.8 / len(models_in_data)
    model_colors = {"LoRA+CL": "#4ECDC4", "e5-multilingual": "#45B7D1"}

    for j, model in enumerate(models_in_data):
        msub = audio_sub[audio_sub["model"] == model]
        deltas = msub["delta_pct"].values
        offset = (j - len(models_in_data)/2 + 0.5) * width_m
        bars = ax.bar(x + offset, deltas, width_m, label=model,
                      color=model_colors.get(model, "#95A5A6"), edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, deltas):
            ax.annotate(f"{val:+.1f}%",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["MRR", "Hit@1", "Hit@5"])
    ax.set_ylabel("Variacao (%)")
    ax.set_title("Impacto no Subset Audio (N=57)")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("FinEduBench-Luso: Impacto de Audio Transcripts no RAG", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "rag_audio_impact.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Chart salvo: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    console.print("\n[bold magenta]===== Experimento: Audio Transcripts no RAG =====[/bold magenta]\n")

    output_dir = _PROJECT_ROOT / "embeddings" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    qa_pairs = load_all_countries(_PROJECT_ROOT / "dataset")
    console.print(f"  {len(qa_pairs)} QA pairs carregados.")

    # Build both corpora
    corpus_text = build_corpus_text_only(qa_pairs)
    corpus_audio, n_audio = build_corpus_with_audio(qa_pairs)
    console.print(f"  Corpus TEXT-ONLY: {len(corpus_text)} documentos")
    console.print(f"  Corpus TEXT+AUDIO: {len(corpus_audio)} documentos (+{n_audio} audio transcripts)")

    # Distribution analysis
    audio_distribution_analysis(qa_pairs, output_dir)

    # Evaluate with both models
    model_loaders = [
        ("LoRA+CL", load_openclip_lora_cl),
        ("e5-multilingual", load_e5_multilingual),
    ]

    all_comparisons = []

    for model_name, loader_fn in model_loaders:
        console.print(f"\n[bold green]========== {model_name} ==========[/bold green]")
        encode_fn, _ = loader_fn()
        if encode_fn is None:
            console.print(f"  [yellow]Pulando {model_name}[/yellow]")
            continue

        # Evaluate TEXT-ONLY
        console.print(f"  [bold]Avaliando corpus TEXT-ONLY...[/bold]")
        t0 = time.time()
        df_text = evaluate_retrieval(qa_pairs, corpus_text, encode_fn)
        df_text["corpus"] = "text_only"
        console.print(f"    MRR={df_text['mrr'].mean():.4f} | Hit@5={df_text['hit@5'].mean():.3f} | {time.time()-t0:.1f}s")

        # Evaluate TEXT+AUDIO
        console.print(f"  [bold]Avaliando corpus TEXT+AUDIO...[/bold]")
        t0 = time.time()
        df_audio = evaluate_retrieval(qa_pairs, corpus_audio, encode_fn)
        df_audio["corpus"] = "text_audio"
        console.print(f"    MRR={df_audio['mrr'].mean():.4f} | Hit@5={df_audio['hit@5'].mean():.3f} | {time.time()-t0:.1f}s")

        # Compare
        comp_df = compare_retrieval_results(df_text, df_audio, model_name, output_dir)
        all_comparisons.append(comp_df)

        # Analyze retrieved types
        analyze_audio_retrieval_types(df_audio, model_name)

        # Free memory
        del encode_fn
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Combine all comparisons
    all_comp = pd.concat(all_comparisons, ignore_index=True)

    # Plot
    console.print("\n[bold green]Gerando visualizacoes...[/bold green]")
    plot_comparison(all_comp, output_dir)

    # Save results
    all_comp.to_csv(output_dir / "rag_audio_comparison.csv", index=False)
    console.print(f"  Resultados salvos em rag_audio_comparison.csv")

    # Save detailed per-QA results
    # (merge text and audio results for each model)
    console.print(f"\n[bold magenta]Experimento concluido![/bold magenta]\n")


if __name__ == "__main__":
    main()

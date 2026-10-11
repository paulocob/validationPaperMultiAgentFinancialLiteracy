"""
Experimento: Reranking com Cultural Notes e Audio Transcripts.

Estrategia otima descoberta em experimentos anteriores:
- Adicionar cultural_notes diretamente ao corpus PREJUDICA o MRR (-2.6%)
  porque notas curtas (140 chars) competem com respostas longas (672 chars)
- Mas Hit@5 melhora e equidade inter-pais melhora (KW p de 0.093 para 0.139)

Hipotese: Retrieve do corpus BASE primeiro, depois RERANK combinando
similaridade original com bonus de cultural_notes/audio_transcript.

Pipeline:
  1. Retrieve top-K (K=20) do corpus BASE (answers + image_descriptions, 1659 docs)
  2. Para cada doc recuperado, calcula score reranked:
     reranked = alpha * original_sim + (1-alpha) * max(cultural_sim, audio_sim)
  3. Testa multiplos valores de alpha: [0.7, 0.8, 0.9, 1.0]
     (alpha=1.0 = sem reranking, baseline)
  4. Avalia MRR, Hit@1, Hit@3, Hit@5, Hit@10
  5. Mede equidade inter-pais (Kruskal-Wallis, variancia, spread)

Uso:
    python3 -m embeddings.rag_reranker_experiment
"""
from __future__ import annotations

import json
import shutil
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
TOP_K_RETRIEVE = 20  # Initial retrieval depth
ALPHA_VALUES = [0.7, 0.8, 0.9, 1.0]  # 1.0 = no reranking (baseline)

# Caminho relativo ao projeto: o original apontava para uma pasta local do autor.
FIGURES_DIR = _PROJECT_ROOT / "embeddings" / "results" / "figures"


# ---------------------------------------------------------------------------
# Corpus builder (BASE only: answers + image_descriptions)
# ---------------------------------------------------------------------------

def build_corpus_base(qa_pairs: list[QAPair]) -> list[dict]:
    """Corpus BASE: answers + image_descriptions."""
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


# ---------------------------------------------------------------------------
# Build auxiliary context maps (cultural_notes, audio_transcripts)
# ---------------------------------------------------------------------------

def build_context_maps(qa_pairs: list[QAPair]) -> tuple[dict[str, str], dict[str, str]]:
    """Build qa_id -> cultural_notes and qa_id -> audio_transcript mappings."""
    cultural_map: dict[str, str] = {}
    audio_map: dict[str, str] = {}
    for qa in qa_pairs:
        if qa.cultural_notes:
            cultural_map[qa.id] = qa.cultural_notes
        if qa.audio_transcript:
            audio_map[qa.id] = qa.audio_transcript
    return cultural_map, audio_map


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
# Reranker evaluation
# ---------------------------------------------------------------------------

def evaluate_reranker(
    qa_pairs: list[QAPair],
    corpus: list[dict],
    encode_fn,
    cultural_map: dict[str, str],
    audio_map: dict[str, str],
    alpha: float,
    top_k_retrieve: int = TOP_K_RETRIEVE,
    k_values: list[int] = K_VALUES,
    batch_size: int = 64,
    corpus_embs: np.ndarray | None = None,
    query_embs: np.ndarray | None = None,
    cultural_embs: dict[str, np.ndarray] | None = None,
    audio_embs: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict, dict]:
    """
    Retrieve top-K from BASE corpus, then rerank using cultural/audio context.

    Returns:
        df: DataFrame with per-query metrics
        corpus_embs: cached corpus embeddings
        query_embs: cached query embeddings
        cultural_embs: cached cultural note embeddings
        audio_embs: cached audio transcript embeddings
    """
    # Encode corpus (cache)
    if corpus_embs is None:
        console.print("    Encoding corpus...")
        corpus_texts = [d["text"] for d in corpus]
        corpus_embs = encode_batch(corpus_texts, encode_fn, batch_size)

    # Build qa_id -> corpus indices mapping
    qa_to_corpus: dict[str, list[int]] = {}
    for idx, doc in enumerate(corpus):
        qa_id = doc["qa_id"]
        if qa_id not in qa_to_corpus:
            qa_to_corpus[qa_id] = []
        qa_to_corpus[qa_id].append(idx)

    # Encode queries (cache)
    if query_embs is None:
        console.print("    Encoding queries...")
        queries = [qa.question for qa in qa_pairs]
        query_embs = encode_batch(queries, encode_fn, batch_size)

    # Encode cultural notes (cache)
    if cultural_embs is None:
        console.print("    Encoding cultural notes...")
        cultural_embs = {}
        # Batch encode all cultural notes
        cultural_ids = list(cultural_map.keys())
        cultural_texts = [cultural_map[qid] for qid in cultural_ids]
        if cultural_texts:
            embs = encode_batch(cultural_texts, encode_fn, batch_size)
            for j, qid in enumerate(cultural_ids):
                cultural_embs[qid] = embs[j]

    # Encode audio transcripts (cache)
    if audio_embs is None:
        console.print("    Encoding audio transcripts...")
        audio_embs = {}
        audio_ids = list(audio_map.keys())
        audio_texts = [audio_map[qid] for qid in audio_ids]
        if audio_texts:
            embs = encode_batch(audio_texts, encode_fn, batch_size)
            for j, qid in enumerate(audio_ids):
                audio_embs[qid] = embs[j]

    max_k = max(k_values)
    records = []

    for i, qa in enumerate(qa_pairs):
        q_emb = query_embs[i]  # (d,)
        sims = (q_emb @ corpus_embs.T)  # (n_docs,)

        # Step 1: Retrieve top-K from BASE corpus
        top_indices = np.argsort(sims)[::-1][:top_k_retrieve]

        # Step 2: Rerank using cultural/audio context
        if alpha < 1.0:
            reranked_scores = []
            for idx in top_indices:
                original_sim = sims[idx]
                doc_qa_id = corpus[idx]["qa_id"]

                # Compute similarity with cultural notes
                cultural_sim = 0.0
                if doc_qa_id in cultural_embs:
                    c_emb = cultural_embs[doc_qa_id]
                    cultural_sim = float(q_emb @ c_emb)

                # Compute similarity with audio transcript
                audio_sim = 0.0
                if doc_qa_id in audio_embs:
                    a_emb = audio_embs[doc_qa_id]
                    audio_sim = float(q_emb @ a_emb)

                # Reranked score: alpha * original + (1-alpha) * max(cultural, audio)
                context_sim = max(cultural_sim, audio_sim)
                reranked = alpha * original_sim + (1.0 - alpha) * context_sim
                reranked_scores.append(reranked)

            # Re-sort by reranked scores
            reranked_order = np.argsort(reranked_scores)[::-1]
            top_indices = top_indices[reranked_order]

        # Ground truth: corpus indices for this QA pair
        gt_indices = set(qa_to_corpus.get(qa.id, []))

        row = {
            "qa_id": qa.id,
            "country": qa.country,
            "topic": qa.topic,
            "difficulty": qa.difficulty,
        }

        # MRR: reciprocal rank of first correct hit
        rr = 0.0
        for rank, idx in enumerate(top_indices[:max_k]):
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

    return pd.DataFrame(records), corpus_embs, query_embs, cultural_embs, audio_embs


# ---------------------------------------------------------------------------
# Equity analysis
# ---------------------------------------------------------------------------

def compute_equity(df: pd.DataFrame) -> dict:
    """Compute inter-country equity metrics."""
    countries = ["BR", "PT", "AO"]
    mrrs = {c: df[df["country"] == c]["mrr"].values for c in countries}
    means = {c: mrrs[c].mean() for c in countries}

    variance = np.var(list(means.values()))
    spread = max(means.values()) - min(means.values())

    # Kruskal-Wallis test
    h_stat, p_val = stats.kruskal(*[mrrs[c] for c in countries])

    return {
        "mrr_br": means["BR"],
        "mrr_pt": means["PT"],
        "mrr_ao": means["AO"],
        "variance": variance,
        "spread": spread,
        "kw_h": h_stat,
        "kw_p": p_val,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_reranker_results(results_df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive comparison plots for reranker experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = results_df["model"].unique()

    # --- Plot 1: MRR by alpha for each model ---
    ax = axes[0, 0]
    model_colors = {"LoRA+CL": "#E53935", "e5-multilingual": "#1E88E5"}
    for model_name in models:
        sub = results_df[results_df["model"] == model_name]
        alphas = sub["alpha"].values
        mrrs = sub["mrr"].values
        color = model_colors.get(model_name, "#666666")
        ax.plot(alphas, mrrs, marker="o", linewidth=2.5, markersize=8,
                label=model_name, color=color)
        for a, m in zip(alphas, mrrs):
            ax.annotate(f"{m:.4f}", xy=(a, m), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8)

    ax.set_xlabel("Alpha (1.0 = no reranking)", fontsize=11)
    ax.set_ylabel("MRR", fontsize=11)
    ax.set_title("MRR vs Alpha (Reranking Strength)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(ALPHA_VALUES)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="baseline")

    # --- Plot 2: Hit@k comparison (best alpha vs baseline) for each model ---
    ax = axes[0, 1]
    width = 0.15
    x = np.arange(len(K_VALUES))

    bar_idx = 0
    for model_name in models:
        sub = results_df[results_df["model"] == model_name]
        baseline = sub[sub["alpha"] == 1.0].iloc[0]
        # Find best alpha (highest MRR, excluding 1.0)
        non_baseline = sub[sub["alpha"] < 1.0]
        if len(non_baseline) > 0:
            best_row = non_baseline.loc[non_baseline["mrr"].idxmax()]
            best_alpha = best_row["alpha"]
        else:
            best_row = baseline
            best_alpha = 1.0

        base_color = model_colors.get(model_name, "#666666")
        # Baseline bars
        hits_base = [baseline[f"hit@{k}"] for k in K_VALUES]
        offset = (bar_idx - len(models) + 0.5) * width
        bars1 = ax.bar(x + offset, hits_base, width, label=f"{model_name} (a=1.0)",
                       color=base_color, alpha=0.4, edgecolor="black", linewidth=0.5)
        bar_idx += 1

        # Best alpha bars
        hits_best = [best_row[f"hit@{k}"] for k in K_VALUES]
        offset = (bar_idx - len(models) + 0.5) * width
        bars2 = ax.bar(x + offset, hits_best, width,
                       label=f"{model_name} (a={best_alpha})",
                       color=base_color, alpha=0.9, edgecolor="black", linewidth=0.5)
        for bar in bars2:
            ax.annotate(f"{bar.get_height():.3f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
        bar_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels([f"Hit@{k}" for k in K_VALUES])
    ax.set_ylabel("Score")
    ax.set_title("Hit@k: Baseline vs Best Reranking", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Plot 3: Equity - KW p-value by alpha ---
    ax = axes[1, 0]
    for model_name in models:
        sub = results_df[results_df["model"] == model_name]
        color = model_colors.get(model_name, "#666666")
        ax.plot(sub["alpha"], sub["kw_p"], marker="s", linewidth=2.5, markersize=8,
                label=f"{model_name} (KW p)", color=color)
        for a, p in zip(sub["alpha"], sub["kw_p"]):
            ax.annotate(f"{p:.4f}", xy=(a, p), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8)

    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.7, label="p=0.05 threshold")
    ax.set_xlabel("Alpha (1.0 = no reranking)", fontsize=11)
    ax.set_ylabel("Kruskal-Wallis p-value", fontsize=11)
    ax.set_title("Inter-Country Equity (higher p = more equitable)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xticks(ALPHA_VALUES)
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Per-country MRR at best alpha ---
    ax = axes[1, 1]
    countries = ["BR", "PT", "AO"]
    country_colors = {"BR": "#2196F3", "PT": "#4CAF50", "AO": "#FF9800"}
    x = np.arange(len(countries))
    width = 0.8 / (len(models) * 2)

    bar_idx = 0
    for model_name in models:
        sub = results_df[results_df["model"] == model_name]
        baseline = sub[sub["alpha"] == 1.0].iloc[0]
        non_baseline = sub[sub["alpha"] < 1.0]
        if len(non_baseline) > 0:
            best_row = non_baseline.loc[non_baseline["mrr"].idxmax()]
            best_alpha = best_row["alpha"]
        else:
            best_row = baseline
            best_alpha = 1.0

        # Baseline
        base_mrrs = [baseline[f"mrr_{c.lower()}"] for c in countries]
        offset = (bar_idx - len(models)) * width
        ax.bar(x + offset, base_mrrs, width, label=f"{model_name} (a=1.0)",
               alpha=0.4, edgecolor="black", linewidth=0.5,
               color=[country_colors[c] for c in countries])
        bar_idx += 1

        # Best
        best_mrrs = [best_row[f"mrr_{c.lower()}"] for c in countries]
        offset = (bar_idx - len(models)) * width
        bars = ax.bar(x + offset, best_mrrs, width,
                      label=f"{model_name} (a={best_alpha})",
                      alpha=0.9, edgecolor="black", linewidth=0.5,
                      color=[country_colors[c] for c in countries])
        for bar in bars:
            ax.annotate(f"{bar.get_height():.4f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
        bar_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.set_ylabel("MRR")
    ax.set_title("Per-Country MRR: Baseline vs Best Reranking", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("FinEduBench-Luso: Reranker Experiment (Cultural Notes + Audio Transcripts)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "rag_reranker_experiment.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Plot salvo: {path}")
    return path


def plot_delta_analysis(results_df: pd.DataFrame, output_dir: Path):
    """Plot delta (improvement) analysis: how much reranking helps per topic/difficulty."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Use the model with best improvement
    for model_name in results_df["model"].unique():
        sub = results_df[results_df["model"] == model_name]
        baseline = sub[sub["alpha"] == 1.0].iloc[0]
        non_baseline = sub[sub["alpha"] < 1.0]
        if len(non_baseline) == 0:
            continue
        best_row = non_baseline.loc[non_baseline["mrr"].idxmax()]
        best_alpha = best_row["alpha"]

        # Plot 1: MRR delta by topic
        ax = axes[0]
        topics = sorted(baseline.keys())
        topic_names = []
        deltas = []
        for key in sorted(baseline.keys()):
            if key.startswith("mrr_topic_"):
                topic = key.replace("mrr_topic_", "")
                topic_names.append(topic)
                delta = best_row[key] - baseline[key]
                deltas.append(delta * 100)  # Convert to percentage points

        if topic_names:
            colors_bar = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]
            bars = ax.bar(topic_names, deltas, color=colors_bar, edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, deltas):
                ax.annotate(f"{val:+.2f}pp",
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 3 if val >= 0 else -12),
                            textcoords="offset points", ha="center", fontsize=9)
            ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
            ax.set_ylabel("Delta MRR (percentage points)")
            ax.set_title(f"MRR Change by Topic ({model_name}, a={best_alpha})",
                         fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, axis="y")

        # Plot 2: MRR delta by difficulty
        ax = axes[1]
        diff_names = []
        diff_deltas = []
        for key in sorted(baseline.keys()):
            if key.startswith("mrr_diff_"):
                diff = key.replace("mrr_diff_", "")
                diff_names.append(diff)
                delta = best_row[key] - baseline[key]
                diff_deltas.append(delta * 100)

        if diff_names:
            colors_bar = ["#4CAF50" if d >= 0 else "#F44336" for d in diff_deltas]
            bars = ax.bar(diff_names, diff_deltas, color=colors_bar, edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, diff_deltas):
                ax.annotate(f"{val:+.2f}pp",
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 3 if val >= 0 else -12),
                            textcoords="offset points", ha="center", fontsize=9)
            ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
            ax.set_ylabel("Delta MRR (percentage points)")
            ax.set_title(f"MRR Change by Difficulty ({model_name}, a={best_alpha})",
                         fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

        break  # Only plot first model with improvements

    plt.suptitle("FinEduBench-Luso: Reranking Impact Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "rag_reranker_delta_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Plot salvo: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    console.print("\n[bold magenta]===== Experimento: Reranking com Cultural Notes + Audio Transcripts =====[/bold magenta]\n")

    output_dir = _PROJECT_ROOT / "embeddings" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    qa_pairs = load_all_countries(_PROJECT_ROOT / "dataset")
    console.print(f"  {len(qa_pairs)} QA pairs carregados.")

    # Build corpus BASE
    corpus = build_corpus_base(qa_pairs)
    console.print(f"  Corpus BASE: {len(corpus)} documentos (answers + image_descriptions)")

    # Build context maps
    cultural_map, audio_map = build_context_maps(qa_pairs)
    console.print(f"  Cultural notes: {len(cultural_map)} pares")
    console.print(f"  Audio transcripts: {len(audio_map)} pares")

    # Stats on context lengths
    if cultural_map:
        cn_lens = [len(v) for v in cultural_map.values()]
        console.print(f"  Cultural notes avg length: {np.mean(cn_lens):.0f} chars")
    if audio_map:
        at_lens = [len(v) for v in audio_map.values()]
        console.print(f"  Audio transcripts avg length: {np.mean(at_lens):.0f} chars")

    model_loaders = [
        ("LoRA+CL", load_openclip_lora_cl),
        ("e5-multilingual", load_e5_multilingual),
    ]

    all_summary_rows = []

    for model_label, loader_fn in model_loaders:
        console.print(f"\n[bold green]{'='*20} {model_label} {'='*20}[/bold green]")
        encode_fn, _ = loader_fn()
        if encode_fn is None:
            console.print(f"  [yellow]Pulando {model_label} (nao encontrado)[/yellow]")
            continue

        # Cache embeddings across alpha values
        corpus_embs = None
        query_embs = None
        cultural_embs_cache = None
        audio_embs_cache = None

        model_results = []

        for alpha in ALPHA_VALUES:
            label = f"alpha={alpha}" + (" (baseline)" if alpha == 1.0 else "")
            console.print(f"\n  [bold cyan]Testando {label}...[/bold cyan]")
            t0 = time.time()

            df, corpus_embs, query_embs, cultural_embs_cache, audio_embs_cache = evaluate_reranker(
                qa_pairs=qa_pairs,
                corpus=corpus,
                encode_fn=encode_fn,
                cultural_map=cultural_map,
                audio_map=audio_map,
                alpha=alpha,
                top_k_retrieve=TOP_K_RETRIEVE,
                corpus_embs=corpus_embs,
                query_embs=query_embs,
                cultural_embs=cultural_embs_cache,
                audio_embs=audio_embs_cache,
            )
            elapsed = time.time() - t0

            # Overall metrics
            mrr = df["mrr"].mean()
            hit1 = df["hit@1"].mean()
            hit3 = df["hit@3"].mean()
            hit5 = df["hit@5"].mean()
            hit10 = df["hit@10"].mean()

            console.print(
                f"    MRR={mrr:.4f} | Hit@1={hit1:.3f} Hit@3={hit3:.3f} "
                f"Hit@5={hit5:.3f} Hit@10={hit10:.3f} | {elapsed:.1f}s"
            )

            # Equity
            equity = compute_equity(df)
            console.print(
                f"    Equity: var={equity['variance']:.6f} spread={equity['spread']:.4f} "
                f"KW: H={equity['kw_h']:.2f} p={equity['kw_p']:.4f}"
            )

            # Summary row
            row = {
                "model": model_label,
                "alpha": alpha,
                "mrr": mrr,
                "hit@1": hit1,
                "hit@3": hit3,
                "hit@5": hit5,
                "hit@10": hit10,
                "time_s": elapsed,
                "mrr_br": equity["mrr_br"],
                "mrr_pt": equity["mrr_pt"],
                "mrr_ao": equity["mrr_ao"],
                "variance": equity["variance"],
                "spread": equity["spread"],
                "kw_h": equity["kw_h"],
                "kw_p": equity["kw_p"],
            }

            # Per-topic MRR
            for t in sorted(df["topic"].unique()):
                row[f"mrr_topic_{t}"] = df[df["topic"] == t]["mrr"].mean()

            # Per-difficulty MRR
            for d in sorted(df["difficulty"].unique()):
                row[f"mrr_diff_{d}"] = df[df["difficulty"] == d]["mrr"].mean()

            model_results.append(row)
            all_summary_rows.append(row)

        # Print comparison table for this model
        console.print(f"\n[bold]Resumo {model_label}:[/bold]")
        table = Table(title=f"Reranker Results: {model_label}", show_lines=True)
        table.add_column("Alpha", style="bold")
        table.add_column("MRR", justify="right")
        table.add_column("Hit@1", justify="right")
        table.add_column("Hit@3", justify="right")
        table.add_column("Hit@5", justify="right")
        table.add_column("Hit@10", justify="right")
        table.add_column("KW p", justify="right")
        table.add_column("Spread", justify="right")

        baseline_mrr = None
        for r in model_results:
            if r["alpha"] == 1.0:
                baseline_mrr = r["mrr"]

        for r in model_results:
            alpha_str = f"{r['alpha']:.1f}"
            if r["alpha"] == 1.0:
                alpha_str += " (base)"
            mrr_str = f"{r['mrr']:.4f}"
            if baseline_mrr and r["alpha"] < 1.0:
                delta = r["mrr"] - baseline_mrr
                pct = (delta / baseline_mrr * 100) if baseline_mrr > 0 else 0
                color = "green" if delta > 0 else "red" if delta < 0 else ""
                mrr_str += f" [{color}]({pct:+.2f}%)[/{color}]"

            table.add_row(
                alpha_str,
                mrr_str,
                f"{r['hit@1']:.3f}",
                f"{r['hit@3']:.3f}",
                f"{r['hit@5']:.3f}",
                f"{r['hit@10']:.3f}",
                f"{r['kw_p']:.4f}",
                f"{r['spread']:.4f}",
            )
        console.print(table)

        # Per-country table
        country_table = Table(title=f"MRR por Pais: {model_label}", show_lines=True)
        country_table.add_column("Alpha", style="bold")
        country_table.add_column("BR", justify="right")
        country_table.add_column("PT", justify="right")
        country_table.add_column("AO", justify="right")
        country_table.add_column("Overall", justify="right", style="bold")

        for r in model_results:
            country_table.add_row(
                f"{r['alpha']:.1f}",
                f"{r['mrr_br']:.4f}",
                f"{r['mrr_pt']:.4f}",
                f"{r['mrr_ao']:.4f}",
                f"{r['mrr']:.4f}",
            )
        console.print(country_table)

        # Free memory
        del encode_fn, corpus_embs, query_embs, cultural_embs_cache, audio_embs_cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # --- Save results ---
    results_df = pd.DataFrame(all_summary_rows)
    csv_path = output_dir / "rag_reranker_results.csv"
    results_df.to_csv(csv_path, index=False)
    console.print(f"\n  Resultados salvos em {csv_path}")

    # --- Generate plots ---
    console.print("\n[bold green]Gerando visualizacoes...[/bold green]")
    plot1 = plot_reranker_results(results_df, output_dir)
    plot2 = plot_delta_analysis(results_df, output_dir)

    # Copy to figures directory
    for plot_path in [plot1, plot2]:
        if plot_path and plot_path.exists():
            dest = FIGURES_DIR / plot_path.name
            shutil.copy2(plot_path, dest)
            console.print(f"  Copiado para {dest}")

    # --- Final summary ---
    console.print("\n[bold magenta]===== Resumo Final =====[/bold magenta]")
    for model_name in results_df["model"].unique():
        sub = results_df[results_df["model"] == model_name]
        baseline = sub[sub["alpha"] == 1.0]
        if len(baseline) == 0:
            continue
        baseline = baseline.iloc[0]

        non_baseline = sub[sub["alpha"] < 1.0]
        if len(non_baseline) > 0:
            best = non_baseline.loc[non_baseline["mrr"].idxmax()]
            best_alpha = best["alpha"]
            delta_mrr = best["mrr"] - baseline["mrr"]
            delta_pct = (delta_mrr / baseline["mrr"] * 100) if baseline["mrr"] > 0 else 0
            delta_hit5 = best["hit@5"] - baseline["hit@5"]

            console.print(f"\n  [bold]{model_name}:[/bold]")
            console.print(f"    Baseline (a=1.0): MRR={baseline['mrr']:.4f} Hit@5={baseline['hit@5']:.3f} KW_p={baseline['kw_p']:.4f}")
            console.print(f"    Best (a={best_alpha}): MRR={best['mrr']:.4f} Hit@5={best['hit@5']:.3f} KW_p={best['kw_p']:.4f}")

            color = "green" if delta_mrr > 0 else "red"
            console.print(f"    [{color}]Delta MRR: {delta_mrr:+.4f} ({delta_pct:+.2f}%)[/{color}]")
            color = "green" if delta_hit5 > 0 else "red"
            console.print(f"    [{color}]Delta Hit@5: {delta_hit5:+.4f}[/{color}]")

            # Equity improvement
            eq_delta = best["kw_p"] - baseline["kw_p"]
            color = "green" if eq_delta > 0 else "red"
            console.print(f"    [{color}]Delta KW_p: {eq_delta:+.4f} (higher = more equitable)[/{color}]")

    console.print(f"\n[bold magenta]Experimento concluido![/bold magenta]\n")


if __name__ == "__main__":
    main()

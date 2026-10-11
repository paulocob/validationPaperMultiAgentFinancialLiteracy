"""
RAG Generation with Reranker: End-to-End Evaluation.

Compares 3 conditions:
  1. Zero-shot (no context)
  2. RAG baseline (direct corpus retrieval, alpha=1.0)
  3. RAG + Reranker (alpha=0.9, cultural_notes + audio_transcripts)

Pipeline:
  1. Load LoRA+CL retriever (best model)
  2. Build BASE corpus (answers + image_descriptions)
  3. Build cultural_notes and audio_transcript maps for reranking
  4. Stratified sample of 150 QA pairs (50/country, seed=42)
  5. For each sample:
     a. Retrieve top-20 from BASE corpus
     b. Rerank with alpha=0.9 (or alpha=1.0 for baseline)
     c. Take top-5 as context
     d. Generate with GPT-4o-mini
  6. Evaluate ROUGE-1, ROUGE-2, ROUGE-L
  7. Compare per-country, compute equity (Kruskal-Wallis)

Uso:
    export OPENAI_API_KEY="sk-..."
    python3 -m embeddings.rag_generation_reranker
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
from scipy import stats
from rich.console import Console
from rich.table import Table

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import load_all_countries, QAPair

console = Console()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_K_RETRIEVE = 20   # Initial retrieval depth
TOP_K_CONTEXT = 5     # Documents used as LLM context
ALPHA_RERANK = 0.9    # Optimal reranking alpha
ALPHA_BASELINE = 1.0  # No reranking
SAMPLE_SIZE = 150     # 50 per country
SEED = 42

LLM_MODEL = "gpt-4o-mini"
MAX_TOKENS = 512
TEMPERATURE = 0.3

SYSTEM_PROMPT = (
    "Voce e um assistente especializado em educacao financeira para paises lusofonos "
    "(Brasil, Portugal, Angola). Responda de forma clara, precisa e culturalmente "
    "adaptada ao contexto do usuario. Responda em portugues."
)

PROMPT_WITH_CONTEXT = """Com base nos documentos recuperados abaixo, responda a pergunta.

Documentos recuperados:
{context}

Pergunta: {question}

Resposta:"""

PROMPT_NO_CONTEXT = """Responda a seguinte pergunta sobre educacao financeira.

Pergunta: {question}

Resposta:"""

# Caminho relativo ao projeto: o original apontava para uma pasta local do autor.
FIGURES_DIR = _PROJECT_ROOT / "embeddings" / "results" / "figures"


# ---------------------------------------------------------------------------
# Model loaders (reused from existing code)
# ---------------------------------------------------------------------------

def _get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_openclip_lora_cl():
    import open_clip
    from embeddings.clip_lora_finetune import apply_lora_simple, load_lora_weights

    lora_path = _PROJECT_ROOT / "embeddings" / "lora_output_cl" / "best_lora.pt"
    if not lora_path.exists():
        console.print(f"[red]LoRA weights not found at {lora_path}[/red]")
        return None

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
    return encode


# ---------------------------------------------------------------------------
# Corpus & context maps
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
        })
        if qa.image_description:
            docs.append({
                "id": f"{qa.id}_image",
                "qa_id": qa.id,
                "text": qa.image_description,
                "type": "image_desc",
                "country": qa.country,
            })
    return docs


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


def encode_batch(texts: list[str], encode_fn, batch_size: int = 64) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = encode_fn(batch)
        all_embs.append(emb)
    return np.vstack(all_embs)


# ---------------------------------------------------------------------------
# Retrieval with optional reranking
# ---------------------------------------------------------------------------

def retrieve_and_rerank(
    query_emb: np.ndarray,
    corpus_embs: np.ndarray,
    corpus: list[dict],
    cultural_embs: dict[str, np.ndarray],
    audio_embs: dict[str, np.ndarray],
    alpha: float,
    top_k_retrieve: int = TOP_K_RETRIEVE,
    top_k_context: int = TOP_K_CONTEXT,
) -> list[dict]:
    """
    Retrieve top-K from BASE corpus, then optionally rerank.

    Args:
        alpha: 1.0 = no reranking (baseline), 0.9 = optimal reranking
    """
    sims = (query_emb @ corpus_embs.T).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k_retrieve]

    if alpha < 1.0:
        reranked_scores = []
        for idx in top_indices:
            original_sim = sims[idx]
            doc_qa_id = corpus[idx]["qa_id"]

            # Cultural notes similarity
            cultural_sim = 0.0
            if doc_qa_id in cultural_embs:
                cultural_sim = float(query_emb.flatten() @ cultural_embs[doc_qa_id])

            # Audio transcript similarity
            audio_sim = 0.0
            if doc_qa_id in audio_embs:
                audio_sim = float(query_emb.flatten() @ audio_embs[doc_qa_id])

            context_sim = max(cultural_sim, audio_sim)
            reranked = alpha * original_sim + (1.0 - alpha) * context_sim
            reranked_scores.append(reranked)

        reranked_order = np.argsort(reranked_scores)[::-1]
        top_indices = top_indices[reranked_order]

    return [corpus[i] for i in top_indices[:top_k_context]]


# ---------------------------------------------------------------------------
# LLM Generation
# ---------------------------------------------------------------------------

def call_llm(prompt: str, client, max_retries: int = 3) -> str:
    """Call GPT-4o-mini with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
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


def generate_with_context(question: str, retrieved_docs: list[dict], client) -> str:
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[{i}] {doc['text']}")
    context = "\n\n".join(context_parts)
    prompt = PROMPT_WITH_CONTEXT.format(context=context, question=question)
    return call_llm(prompt, client)


def generate_no_context(question: str, client) -> str:
    prompt = PROMPT_NO_CONTEXT.format(question=question)
    return call_llm(prompt, client)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_rouge(prediction: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate comparison plot: 3 conditions x per-country ROUGE-L."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    conditions = ["Zero-shot", "RAG (baseline)", "RAG + Reranker"]
    condition_colors = {
        "Zero-shot": "#FF6B6B",
        "RAG (baseline)": "#4ECDC4",
        "RAG + Reranker": "#45B7D1",
    }
    countries = ["BR", "PT", "AO"]
    country_labels = {"BR": "Brasil", "PT": "Portugal", "AO": "Angola"}

    # --- Plot 1: Overall ROUGE comparison (bars) ---
    ax = axes[0]
    metrics = ["rouge1_f", "rouge2_f", "rougeL_f"]
    metric_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    x = np.arange(len(metrics))
    width = 0.25

    for i, cond in enumerate(conditions):
        sub = df[df["condition"] == cond]
        vals = [sub[m].mean() for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=cond,
                      color=condition_colors[cond], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score (F1)")
    ax.set_title("Overall ROUGE Scores", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Plot 2: Per-country ROUGE-L ---
    ax = axes[1]
    x = np.arange(len(countries))
    width = 0.25

    for i, cond in enumerate(conditions):
        sub = df[df["condition"] == cond]
        vals = [sub[sub["country"] == c]["rougeL_f"].mean() for c in countries]
        bars = ax.bar(x + i * width, vals, width, label=cond,
                      color=condition_colors[cond], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([country_labels[c] for c in countries])
    ax.set_ylabel("ROUGE-L (F1)")
    ax.set_title("ROUGE-L by Country", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Plot 3: Equity analysis (spread + KW p-value) ---
    ax = axes[2]
    equity_data = []
    for cond in conditions:
        sub = df[df["condition"] == cond]
        country_means = [sub[sub["country"] == c]["rougeL_f"].mean() for c in countries]
        spread = max(country_means) - min(country_means)

        # Kruskal-Wallis test
        groups = [sub[sub["country"] == c]["rougeL_f"].values for c in countries]
        try:
            h_stat, p_val = stats.kruskal(*groups)
        except Exception:
            h_stat, p_val = 0.0, 1.0

        equity_data.append({
            "condition": cond,
            "spread": spread,
            "kw_p": p_val,
            "kw_h": h_stat,
        })

    # Bar plot for spread
    conds_short = ["Zero-shot", "RAG\n(baseline)", "RAG +\nReranker"]
    spreads = [e["spread"] for e in equity_data]
    kw_ps = [e["kw_p"] for e in equity_data]
    colors_bar = [condition_colors[c] for c in conditions]

    bars = ax.bar(conds_short, spreads, color=colors_bar, edgecolor="black", linewidth=0.5)
    for bar, sp, p in zip(bars, spreads, kw_ps):
        ax.annotate(f"spread={sp:.3f}\nKW p={p:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)

    ax.set_ylabel("Inter-country Spread (ROUGE-L)")
    ax.set_title("Equity: Inter-Country Spread\n(lower = more equitable)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(spreads) * 1.8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("FinEduBench-Luso: RAG Generation with Reranker (alpha=0.9)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = output_dir / "rag_generation_reranker.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Plot salvo: {path}")
    return path


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

    console.print("\n[bold magenta]===== RAG Generation with Reranker (alpha=0.9) =====[/bold magenta]\n")

    output_dir = _PROJECT_ROOT / "embeddings" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    all_qa = load_all_countries(_PROJECT_ROOT / "dataset")
    console.print(f"  Total QA pairs: {len(all_qa)}")

    # --- Stratified sample: 50 per country ---
    random.seed(SEED)
    by_country: dict[str, list[QAPair]] = {}
    for qa in all_qa:
        by_country.setdefault(qa.country, []).append(qa)

    per_country = SAMPLE_SIZE // 3  # 50
    sample_qa: list[QAPair] = []
    for c in ["BR", "PT", "AO"]:
        pool = by_country[c][:]
        random.shuffle(pool)
        sample_qa.extend(pool[:per_country])
    random.shuffle(sample_qa)

    console.print(f"  Stratified sample: {len(sample_qa)} pairs ({per_country}/country)")
    for c in ["BR", "PT", "AO"]:
        n = sum(1 for q in sample_qa if q.country == c)
        console.print(f"    {c}: {n}")

    # --- Build corpus from ALL QA pairs ---
    corpus = build_corpus_base(all_qa)
    console.print(f"  Corpus BASE: {len(corpus)} documents")

    # --- Build context maps ---
    cultural_map, audio_map = build_context_maps(all_qa)
    console.print(f"  Cultural notes: {len(cultural_map)} | Audio transcripts: {len(audio_map)}")

    # --- Load retriever ---
    console.print("\n[bold green]Loading LoRA+CL retriever...[/bold green]")
    encode_fn = load_openclip_lora_cl()
    if encode_fn is None:
        console.print("[red]LoRA+CL not found. Aborting.[/red]")
        sys.exit(1)

    # --- Encode corpus ---
    console.print("  Encoding corpus...")
    corpus_texts = [d["text"] for d in corpus]
    corpus_embs = encode_batch(corpus_texts, encode_fn)

    # --- Encode queries ---
    console.print("  Encoding queries...")
    query_texts = [qa.question for qa in sample_qa]
    query_embs = encode_batch(query_texts, encode_fn)

    # --- Encode cultural notes ---
    console.print("  Encoding cultural notes...")
    cultural_embs: dict[str, np.ndarray] = {}
    cultural_ids = list(cultural_map.keys())
    if cultural_ids:
        cultural_texts = [cultural_map[qid] for qid in cultural_ids]
        embs = encode_batch(cultural_texts, encode_fn)
        for j, qid in enumerate(cultural_ids):
            cultural_embs[qid] = embs[j]

    # --- Encode audio transcripts ---
    console.print("  Encoding audio transcripts...")
    audio_embs: dict[str, np.ndarray] = {}
    audio_ids = list(audio_map.keys())
    if audio_ids:
        audio_texts = [audio_map[qid] for qid in audio_ids]
        embs = encode_batch(audio_texts, encode_fn)
        for j, qid in enumerate(audio_ids):
            audio_embs[qid] = embs[j]

    # Free retriever model
    del encode_fn
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    console.print(f"\n  Embeddings ready. Starting generation (3 x {len(sample_qa)} = {3 * len(sample_qa)} API calls)...\n")

    # --- Build difficulty map ---
    diff_map = {qa.id: qa.difficulty for qa in all_qa}
    topic_map = {qa.id: qa.topic for qa in all_qa}

    # --- Generate answers for all 3 conditions ---
    all_records = []
    conditions_config = [
        ("Zero-shot", None),
        ("RAG (baseline)", ALPHA_BASELINE),
        ("RAG + Reranker", ALPHA_RERANK),
    ]

    for cond_name, alpha in conditions_config:
        console.print(f"[bold cyan]--- {cond_name} ---[/bold cyan]")
        t0 = time.time()

        for i, qa in enumerate(sample_qa):
            if cond_name == "Zero-shot":
                prediction = generate_no_context(qa.question, client)
            else:
                retrieved = retrieve_and_rerank(
                    query_emb=query_embs[i:i+1],
                    corpus_embs=corpus_embs,
                    corpus=corpus,
                    cultural_embs=cultural_embs,
                    audio_embs=audio_embs,
                    alpha=alpha,
                )
                prediction = generate_with_context(qa.question, retrieved, client)

            # Compute ROUGE
            rouge = compute_rouge(prediction, qa.answer)

            record = {
                "qa_id": qa.id,
                "country": qa.country,
                "topic": qa.topic,
                "difficulty": diff_map.get(qa.id, "unknown"),
                "condition": cond_name,
                "alpha": alpha if alpha is not None else "N/A",
                "prediction": prediction,
                "reference": qa.answer,
                **rouge,
            }
            all_records.append(record)

            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(sample_qa) - i - 1) / rate if rate > 0 else 0
                avg_rougeL = np.mean([r["rougeL_f"] for r in all_records if r["condition"] == cond_name])
                console.print(
                    f"  {i+1}/{len(sample_qa)} | {elapsed:.0f}s | ~{eta:.0f}s remaining | "
                    f"avg ROUGE-L={avg_rougeL:.4f}"
                )

        elapsed = time.time() - t0
        cond_records = [r for r in all_records if r["condition"] == cond_name]
        avg_rl = np.mean([r["rougeL_f"] for r in cond_records])
        console.print(f"  Completed: {elapsed:.1f}s | avg ROUGE-L={avg_rl:.4f}\n")

    # --- Build DataFrame ---
    df = pd.DataFrame(all_records)

    # --- Save CSV (without full text to keep manageable) ---
    df_save = df.drop(columns=["prediction", "reference"])
    csv_path = output_dir / "rag_generation_reranker_results.csv"
    df_save.to_csv(csv_path, index=False)
    console.print(f"\n  Results saved: {csv_path}")

    # --- Print summary tables ---
    _print_summary(df)

    # --- Generate plots ---
    console.print("\n[bold green]Generating plots...[/bold green]")
    plot_path = plot_results(df, output_dir)

    # Copy to figures directory
    if plot_path and plot_path.exists():
        dest = FIGURES_DIR / plot_path.name
        shutil.copy2(plot_path, dest)
        console.print(f"  Copied to {dest}")

    console.print(f"\n[bold magenta]Experiment complete![/bold magenta]\n")


def _print_summary(df: pd.DataFrame):
    """Print summary tables."""
    conditions = ["Zero-shot", "RAG (baseline)", "RAG + Reranker"]
    countries = ["BR", "PT", "AO"]

    # --- Overall comparison ---
    console.print()
    table = Table(title="Generation Quality: 3 Conditions", show_lines=True)
    table.add_column("Condition", style="bold cyan")
    table.add_column("ROUGE-1", justify="right")
    table.add_column("ROUGE-2", justify="right")
    table.add_column("ROUGE-L", justify="right")
    table.add_column("N", justify="right")

    for cond in conditions:
        sub = df[df["condition"] == cond]
        table.add_row(
            cond,
            f"{sub['rouge1_f'].mean():.4f}",
            f"{sub['rouge2_f'].mean():.4f}",
            f"{sub['rougeL_f'].mean():.4f}",
            str(len(sub)),
        )
    console.print(table)

    # --- Per-country ROUGE-L ---
    for cond in conditions:
        sub = df[df["condition"] == cond]
        table = Table(title=f"ROUGE-L by Country: {cond}", show_lines=True)
        table.add_column("Country", style="bold")
        table.add_column("ROUGE-1", justify="right")
        table.add_column("ROUGE-2", justify="right")
        table.add_column("ROUGE-L", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("N", justify="right")

        for c in countries:
            csub = sub[sub["country"] == c]
            if len(csub) > 0:
                table.add_row(
                    c,
                    f"{csub['rouge1_f'].mean():.4f}",
                    f"{csub['rouge2_f'].mean():.4f}",
                    f"{csub['rougeL_f'].mean():.4f}",
                    f"{csub['rougeL_f'].std():.4f}",
                    str(len(csub)),
                )
        console.print(table)

    # --- Equity analysis ---
    console.print()
    table = Table(title="Equity Analysis (ROUGE-L)", show_lines=True)
    table.add_column("Condition", style="bold cyan")
    table.add_column("Spread", justify="right")
    table.add_column("Variance", justify="right")
    table.add_column("KW H", justify="right")
    table.add_column("KW p", justify="right")
    table.add_column("Equitable?", justify="center")

    for cond in conditions:
        sub = df[df["condition"] == cond]
        country_means = [sub[sub["country"] == c]["rougeL_f"].mean() for c in countries]
        spread = max(country_means) - min(country_means)
        variance = np.var(country_means)

        groups = [sub[sub["country"] == c]["rougeL_f"].values for c in countries]
        try:
            h_stat, p_val = stats.kruskal(*groups)
        except Exception:
            h_stat, p_val = 0.0, 1.0

        equitable = "YES" if p_val > 0.05 else "NO"
        table.add_row(
            cond,
            f"{spread:.4f}",
            f"{variance:.6f}",
            f"{h_stat:.2f}",
            f"{p_val:.4f}",
            equitable,
        )
    console.print(table)

    # --- Delta analysis ---
    console.print()
    table = Table(title="Delta vs Zero-shot", show_lines=True)
    table.add_column("Condition", style="bold cyan")
    table.add_column("Delta ROUGE-L", justify="right")
    table.add_column("Delta %", justify="right")

    zs_rl = df[df["condition"] == "Zero-shot"]["rougeL_f"].mean()
    for cond in ["RAG (baseline)", "RAG + Reranker"]:
        cond_rl = df[df["condition"] == cond]["rougeL_f"].mean()
        delta = cond_rl - zs_rl
        pct = (delta / zs_rl * 100) if zs_rl > 0 else 0
        table.add_row(cond, f"{delta:+.4f}", f"{pct:+.2f}%")
    console.print(table)

    # --- Delta Reranker vs Baseline ---
    base_rl = df[df["condition"] == "RAG (baseline)"]["rougeL_f"].mean()
    rerank_rl = df[df["condition"] == "RAG + Reranker"]["rougeL_f"].mean()
    delta = rerank_rl - base_rl
    pct = (delta / base_rl * 100) if base_rl > 0 else 0
    console.print(f"\n  [bold]Reranker vs Baseline: Delta ROUGE-L = {delta:+.4f} ({pct:+.2f}%)[/bold]")

    # Per-country delta
    for c in countries:
        base_c = df[(df["condition"] == "RAG (baseline)") & (df["country"] == c)]["rougeL_f"].mean()
        rerank_c = df[(df["condition"] == "RAG + Reranker") & (df["country"] == c)]["rougeL_f"].mean()
        delta_c = rerank_c - base_c
        console.print(f"    {c}: {delta_c:+.4f} (baseline={base_c:.4f}, reranker={rerank_c:.4f})")


if __name__ == "__main__":
    main()

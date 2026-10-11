"""
RAG Generation End-to-End: Retrieval + LLM Generation + Evaluation.

Pipeline:
  1. Para cada pergunta, recupera top-k documentos (usando LoRA+CL e e5-multilingual)
  2. Monta prompt com contexto recuperado e envia ao GPT-4o-mini
  3. Compara resposta gerada vs ground truth usando ROUGE-L e BERTScore
  4. Tambem gera respostas SEM contexto (baseline zero-shot) para comparacao
  5. Desagrega por pais, topico e dificuldade

Uso:
    export OPENAI_API_KEY="sk-..."
    python -m embeddings.rag_generation
    python -m embeddings.rag_generation --sample 100   # amostra menor para teste
"""
from __future__ import annotations

import json
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from rouge_score import rouge_scorer
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
TOP_K = 5
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


# ---------------------------------------------------------------------------
# Model loaders (reuse from rag_evaluation)
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
# Corpus & Retrieval
# ---------------------------------------------------------------------------

def build_corpus(qa_pairs: list[QAPair]) -> list[dict]:
    docs = []
    for qa in qa_pairs:
        docs.append({
            "id": f"{qa.id}_answer",
            "qa_id": qa.id,
            "text": qa.answer,
            "type": "answer",
        })
        if qa.image_description:
            docs.append({
                "id": f"{qa.id}_image",
                "qa_id": qa.id,
                "text": qa.image_description,
                "type": "image_desc",
            })
    return docs


def encode_batch(texts: list[str], encode_fn, batch_size: int = 64) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = encode_fn(batch)
        all_embs.append(emb)
    return np.vstack(all_embs)


def retrieve_top_k(query_emb: np.ndarray, corpus_embs: np.ndarray,
                   corpus: list[dict], k: int = TOP_K) -> list[dict]:
    """Retrieve top-k documents for a single query."""
    sims = (query_emb @ corpus_embs.T).flatten()
    top_idx = np.argsort(sims)[::-1][:k]
    return [corpus[i] for i in top_idx]


# ---------------------------------------------------------------------------
# LLM Generation
# ---------------------------------------------------------------------------

def call_llm(prompt: str, client) -> str:
    """Call GPT-4o-mini via OpenAI API."""
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


def generate_with_context(question: str, retrieved_docs: list[dict], client) -> str:
    """Generate answer using retrieved context."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[{i}] {doc['text']}")
    context = "\n\n".join(context_parts)

    prompt = PROMPT_WITH_CONTEXT.format(context=context, question=question)
    return call_llm(prompt, client)


def generate_no_context(question: str, client) -> str:
    """Generate answer without retrieval (zero-shot baseline)."""
    prompt = PROMPT_NO_CONTEXT.format(question=question)
    return call_llm(prompt, client)


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_rouge(prediction: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def compute_bertscore_batch(predictions: list[str], references: list[str]) -> list[dict]:
    """Compute BERTScore for a batch of predictions."""
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        predictions, references,
        lang="pt",
        verbose=False,
        device="mps" if torch.backends.mps.is_available() else "cpu",
    )
    results = []
    for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist()):
        results.append({"bertscore_p": p, "bertscore_r": r, "bertscore_f1": f})
    return results


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Numero de QA pairs para avaliar (0 = todos)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]OPENAI_API_KEY nao configurada. Execute:[/red]")
        console.print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    console.print("\n[bold magenta]===== FinEduBench-Luso: RAG Generation End-to-End =====[/bold magenta]\n")

    output_dir = _PROJECT_ROOT / "embeddings" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    qa_pairs = load_all_countries(_PROJECT_ROOT / "dataset")
    console.print(f"  {len(qa_pairs)} QA pairs carregados.")

    # Sample if requested
    if args.sample > 0 and args.sample < len(qa_pairs):
        random.seed(args.seed)
        # Stratified sample: equal per country
        by_country = {}
        for qa in qa_pairs:
            by_country.setdefault(qa.country, []).append(qa)
        per_country = args.sample // 3
        qa_pairs = []
        for c in ["BR", "PT", "AO"]:
            random.shuffle(by_country[c])
            qa_pairs.extend(by_country[c][:per_country])
        random.shuffle(qa_pairs)
        console.print(f"  Amostra estratificada: {len(qa_pairs)} pares ({per_country}/pais)")

    # Build corpus
    corpus = build_corpus(qa_pairs)
    console.print(f"  {len(corpus)} documentos no corpus.\n")

    # Load retrieval models
    retrieval_configs = [
        ("LoRA+CL", load_openclip_lora_cl),
        ("e5-multilingual", load_e5_multilingual),
    ]

    # Build difficulty map
    all_qa = load_all_countries(_PROJECT_ROOT / "dataset")
    diff_map = {qa.id: qa.difficulty for qa in all_qa}

    all_records = []
    # Store prediction texts for BERTScore computation
    all_predictions: dict[str, list[str]] = {}  # key: "condition|retriever" -> list of texts
    all_references: dict[str, list[str]] = {}

    for retriever_name, loader_fn in retrieval_configs:
        console.print(f"[bold green]Carregando retriever: {retriever_name}...[/bold green]")
        encode_fn, _ = loader_fn()
        if encode_fn is None:
            console.print(f"  [yellow]Pulando {retriever_name}[/yellow]")
            continue

        # Encode corpus
        console.print("  Codificando corpus...")
        corpus_texts = [d["text"] for d in corpus]
        corpus_embs = encode_batch(corpus_texts, encode_fn)

        # Encode queries
        console.print("  Codificando perguntas...")
        query_texts = [qa.question for qa in qa_pairs]
        query_embs = encode_batch(query_texts, encode_fn)

        # Free retriever model
        del encode_fn
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # --- Generate answers ---
        console.print(f"  Gerando respostas com {LLM_MODEL}...")
        predictions_rag = []
        predictions_norag = []
        references = []
        retrieval_hits = []

        t0 = time.time()
        for i, qa in enumerate(qa_pairs):
            # Retrieve
            retrieved = retrieve_top_k(query_embs[i:i+1], corpus_embs, corpus, TOP_K)
            gt_ids = {f"{qa.id}_answer", f"{qa.id}_image"}
            hit = any(d["id"] in gt_ids for d in retrieved)
            retrieval_hits.append(hit)

            # Generate WITH context
            try:
                ans_rag = generate_with_context(qa.question, retrieved, client)
            except Exception as e:
                console.print(f"  [red]Erro API (RAG) em {qa.id}: {e}[/red]")
                ans_rag = ""

            # Generate WITHOUT context (only for first retriever to avoid duplicate cost)
            if retriever_name == "LoRA+CL":
                try:
                    ans_norag = generate_no_context(qa.question, client)
                except Exception as e:
                    console.print(f"  [red]Erro API (no-RAG) em {qa.id}: {e}[/red]")
                    ans_norag = ""
                predictions_norag.append(ans_norag)

            predictions_rag.append(ans_rag)
            references.append(qa.answer)

            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(qa_pairs) - i - 1) / rate if rate > 0 else 0
                console.print(
                    f"    {i+1}/{len(qa_pairs)} | "
                    f"{elapsed:.0f}s | ~{eta:.0f}s restantes | "
                    f"Hit@5={sum(retrieval_hits)/(i+1):.3f}"
                )

        elapsed = time.time() - t0
        console.print(f"  Geracao concluida: {elapsed:.1f}s ({len(qa_pairs)} pares)\n")

        # Store texts for BERTScore
        rag_key = f"RAG|{retriever_name}"
        all_predictions[rag_key] = predictions_rag
        all_references[rag_key] = references

        if retriever_name == "LoRA+CL" and predictions_norag:
            all_predictions["Zero-shot|none"] = predictions_norag
            all_references["Zero-shot|none"] = references

        # --- Compute ROUGE ---
        console.print(f"  Computando ROUGE ({retriever_name})...")
        for i, (pred, ref) in enumerate(zip(predictions_rag, references)):
            rouge = compute_rouge(pred, ref)
            record = {
                "qa_id": qa_pairs[i].id,
                "country": qa_pairs[i].country,
                "topic": qa_pairs[i].topic,
                "difficulty": diff_map.get(qa_pairs[i].id, "unknown"),
                "retriever": retriever_name,
                "condition": "RAG",
                "retrieval_hit": retrieval_hits[i],
                "prediction": pred,
                **rouge,
            }
            all_records.append(record)

        # No-RAG condition (only once)
        if retriever_name == "LoRA+CL" and predictions_norag:
            console.print("  Computando ROUGE (zero-shot)...")
            for i, (pred, ref) in enumerate(zip(predictions_norag, references)):
                rouge = compute_rouge(pred, ref)
                record = {
                    "qa_id": qa_pairs[i].id,
                    "country": qa_pairs[i].country,
                    "topic": qa_pairs[i].topic,
                    "difficulty": diff_map.get(qa_pairs[i].id, "unknown"),
                    "retriever": "none",
                    "condition": "Zero-shot",
                    "retrieval_hit": False,
                    "prediction": pred,
                    **rouge,
                }
                all_records.append(record)

    # --- Compute BERTScore (batch, per condition) ---
    df = pd.DataFrame(all_records)

    # Initialize BERTScore columns
    df["bertscore_p"] = np.nan
    df["bertscore_r"] = np.nan
    df["bertscore_f1"] = np.nan

    try:
        console.print("[bold green]Computando BERTScore...[/bold green]")
        for key, preds in all_predictions.items():
            cond, ret = key.split("|")
            refs = all_references[key]

            console.print(f"  BERTScore para {key} ({len(preds)} pares)...")
            bert_results = compute_bertscore_batch(preds, refs)

            mask = (df["condition"] == cond) & (df["retriever"] == ret)
            indices = df[mask].index.tolist()
            for idx, bs in zip(indices, bert_results):
                df.loc[idx, "bertscore_p"] = bs["bertscore_p"]
                df.loc[idx, "bertscore_r"] = bs["bertscore_r"]
                df.loc[idx, "bertscore_f1"] = bs["bertscore_f1"]

        console.print("  BERTScore concluido.\n")
    except Exception as e:
        console.print(f"  [yellow]BERTScore falhou ({e}). Continuando com ROUGE apenas.[/yellow]\n")

    # Save main results (drop prediction text to keep CSV manageable)
    df_save = df.drop(columns=["prediction"])
    csv_path = output_dir / "rag_generation_results.csv"
    df_save.to_csv(csv_path, index=False)
    console.print(f"\n  Resultados salvos em {csv_path.name}")

    # --- Summary Tables ---
    console.print("\n")
    _print_summary(df)

    # --- Plots ---
    console.print("\n[bold green]Gerando visualizacoes...[/bold green]")
    _plot_results(df, output_dir)

    console.print(f"\n[bold magenta]Concluido! Resultados em {output_dir}[/bold magenta]\n")


def _print_summary(df: pd.DataFrame):
    """Print summary tables."""
    # Overall comparison: Zero-shot vs RAG(LoRA+CL) vs RAG(e5)
    table = Table(title="Geracao End-to-End: Metricas de Qualidade", show_lines=True)
    table.add_column("Condicao", style="bold cyan")
    table.add_column("ROUGE-1", justify="right")
    table.add_column("ROUGE-2", justify="right")
    table.add_column("ROUGE-L", justify="right")
    table.add_column("BERTScore", justify="right")
    table.add_column("N", justify="right")

    for (cond, ret), group in df.groupby(["condition", "retriever"]):
        label = f"{cond}" if cond == "Zero-shot" else f"RAG ({ret})"
        bs_f1 = group["bertscore_f1"].mean() if "bertscore_f1" in group else 0.0
        table.add_row(
            label,
            f"{group['rouge1_f'].mean():.4f}",
            f"{group['rouge2_f'].mean():.4f}",
            f"{group['rougeL_f'].mean():.4f}",
            f"{bs_f1:.4f}",
            str(len(group)),
        )
    console.print(table)

    # Per-country breakdown for RAG conditions
    for ret in df[df["condition"] == "RAG"]["retriever"].unique():
        sub = df[(df["condition"] == "RAG") & (df["retriever"] == ret)]
        table = Table(title=f"ROUGE-L por Pais: RAG ({ret})", show_lines=True)
        table.add_column("Pais", style="bold")
        table.add_column("ROUGE-1", justify="right")
        table.add_column("ROUGE-L", justify="right")
        table.add_column("N", justify="right")

        for c in ["BR", "PT", "AO"]:
            csub = sub[sub["country"] == c]
            table.add_row(
                c,
                f"{csub['rouge1_f'].mean():.4f}",
                f"{csub['rougeL_f'].mean():.4f}",
                str(len(csub)),
            )
        console.print(table)

    # Zero-shot per-country
    zs = df[df["condition"] == "Zero-shot"]
    if len(zs) > 0:
        table = Table(title="ROUGE-L por Pais: Zero-shot", show_lines=True)
        table.add_column("Pais", style="bold")
        table.add_column("ROUGE-1", justify="right")
        table.add_column("ROUGE-L", justify="right")

        for c in ["BR", "PT", "AO"]:
            csub = zs[zs["country"] == c]
            table.add_row(
                c,
                f"{csub['rouge1_f'].mean():.4f}",
                f"{csub['rougeL_f'].mean():.4f}",
            )
        console.print(table)

    # By difficulty
    for (cond, ret), group in df.groupby(["condition", "retriever"]):
        label = f"{cond}" if cond == "Zero-shot" else f"RAG ({ret})"
        table = Table(title=f"ROUGE-L por Dificuldade: {label}", show_lines=True)
        table.add_column("Nivel", style="bold")
        table.add_column("ROUGE-L", justify="right")
        table.add_column("N", justify="right")

        for diff in ["basic", "intermediate", "advanced"]:
            dsub = group[group["difficulty"] == diff]
            if len(dsub) > 0:
                table.add_row(diff.capitalize(), f"{dsub['rougeL_f'].mean():.4f}", str(len(dsub)))
        console.print(table)


def _plot_results(df: pd.DataFrame, output_dir: Path):
    """Generate visualization plots."""
    # --- Plot 1: Overall ROUGE comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart: conditions comparison
    ax = axes[0]
    conditions = []
    rouge_l_means = []
    rouge_l_stds = []
    colors = []
    color_map = {"Zero-shot": "#FF6B6B", "RAG (LoRA+CL)": "#4ECDC4", "RAG (e5-multilingual)": "#45B7D1"}

    for (cond, ret), group in df.groupby(["condition", "retriever"]):
        label = f"{cond}" if cond == "Zero-shot" else f"RAG ({ret})"
        conditions.append(label)
        rouge_l_means.append(group["rougeL_f"].mean())
        rouge_l_stds.append(group["rougeL_f"].std())
        colors.append(color_map.get(label, "#95A5A6"))

    bars = ax.bar(conditions, rouge_l_means, yerr=rouge_l_stds, color=colors,
                  capsize=5, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, rouge_l_means):
        ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("ROUGE-L (F1)")
    ax.set_title("Qualidade da Geracao: Zero-shot vs RAG")
    ax.set_ylim(0, max(rouge_l_means) * 1.3)
    ax.grid(True, alpha=0.3, axis="y")

    # Bar chart: per-country for RAG(LoRA+CL) vs Zero-shot
    ax = axes[1]
    countries = ["BR", "PT", "AO"]
    x = np.arange(len(countries))
    width = 0.35

    zs = df[df["condition"] == "Zero-shot"]
    rag_lora = df[(df["condition"] == "RAG") & (df["retriever"] == "LoRA+CL")]

    if len(zs) > 0 and len(rag_lora) > 0:
        zs_vals = [zs[zs["country"] == c]["rougeL_f"].mean() for c in countries]
        rag_vals = [rag_lora[rag_lora["country"] == c]["rougeL_f"].mean() for c in countries]

        bars1 = ax.bar(x - width/2, zs_vals, width, label="Zero-shot", color="#FF6B6B")
        bars2 = ax.bar(x + width/2, rag_vals, width, label="RAG (LoRA+CL)", color="#4ECDC4")

        for bars in [bars1, bars2]:
            for bar in bars:
                ax.annotate(f"{bar.get_height():.3f}",
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(countries)
        ax.set_ylabel("ROUGE-L (F1)")
        ax.set_title("ROUGE-L por Pais")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("FinEduBench-Luso: RAG Generation Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "rag_generation_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  Chart salvo: {path.name}")

    # --- Plot 2: ROUGE by topic ---
    rag_lora = df[(df["condition"] == "RAG") & (df["retriever"] == "LoRA+CL")]
    if len(rag_lora) > 0:
        topics = sorted(rag_lora["topic"].unique())
        fig, ax = plt.subplots(figsize=(10, 6))

        topic_vals = [rag_lora[rag_lora["topic"] == t]["rougeL_f"].mean() for t in topics]
        topic_colors = plt.cm.Set2(np.linspace(0, 1, len(topics)))
        bars = ax.bar(topics, topic_vals, color=topic_colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, topic_vals):
            ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        ax.set_ylabel("ROUGE-L (F1)")
        ax.set_title("ROUGE-L por Topico: RAG (LoRA+CL)")
        ax.set_ylim(0, max(topic_vals) * 1.3)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = output_dir / "rag_generation_by_topic.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        console.print(f"  Chart salvo: {path.name}")

    # --- Plot 3: Improvement heatmap (RAG vs Zero-shot by country x topic) ---
    if len(zs) > 0 and len(rag_lora) > 0:
        topics = sorted(rag_lora["topic"].unique())
        improvement = pd.DataFrame(index=topics, columns=countries, dtype=float)

        for t in topics:
            for c in countries:
                zs_val = zs[(zs["topic"] == t) & (zs["country"] == c)]["rougeL_f"].mean()
                rag_val = rag_lora[(rag_lora["topic"] == t) & (rag_lora["country"] == c)]["rougeL_f"].mean()
                if zs_val > 0:
                    improvement.loc[t, c] = ((rag_val - zs_val) / zs_val) * 100
                else:
                    improvement.loc[t, c] = 0.0

        fig, ax = plt.subplots(figsize=(8, 6))
        import seaborn as sns
        sns.heatmap(improvement.astype(float), annot=True, fmt=".1f", cmap="RdYlGn",
                    center=0, ax=ax, linewidths=0.5)
        ax.set_title("Melhoria ROUGE-L (%) com RAG vs Zero-shot", fontweight="bold")
        ax.set_ylabel("Topico")
        ax.set_xlabel("Pais")

        plt.tight_layout()
        path = output_dir / "rag_generation_improvement.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        console.print(f"  Chart salvo: {path.name}")


if __name__ == "__main__":
    main()

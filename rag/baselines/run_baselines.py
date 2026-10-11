"""
Baselines RAG para o FinEduBench-Luso.

Executa tres sistemas baseline no benchmark:
1) Text-only RAG: indexa e recupera apenas texto
2) Multimodal RAG: indexa texto + descricoes de imagem
3) Random modality: selecao aleatoria de modalidade

Usa ChromaDB para indexacao e OpenCLIP para embeddings.

Uso:
    python -m baselines.run_baselines
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from rich.console import Console
from rich.table import Table

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import QAPair, load_all_countries

console = Console()

# Modalidades possiveis
ALL_MODALITIES = ["text", "image", "audio"]


def get_embedding_function():
    """
    Retorna funcao de embedding usando OpenCLIP para ChromaDB.

    Usa sentence-transformers como fallback mais leve.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        class STEmbeddingFunction:
            def __call__(self, input: list[str]) -> list[list[float]]:
                embeddings = model.encode(input, convert_to_numpy=True)
                return embeddings.tolist()

        return STEmbeddingFunction()
    except Exception as e:
        console.print(f"[yellow]Aviso: Nao foi possivel carregar sentence-transformers: {e}[/yellow]")
        console.print("[yellow]Usando embeddings default do ChromaDB.[/yellow]")
        return None


class TextOnlyRAG:
    """Baseline 1: RAG usando apenas texto."""

    def __init__(self, qa_pairs: list[QAPair], embed_fn: Any = None) -> None:
        """Inicializa o indice text-only."""
        self.name = "Text-Only RAG"
        self.client = chromadb.Client()

        collection_kwargs: dict[str, Any] = {"name": "text_only"}
        if embed_fn:
            collection_kwargs["embedding_function"] = embed_fn

        self.collection = self.client.create_collection(**collection_kwargs)

        # Indexa apenas as respostas de texto
        ids = [qa.id for qa in qa_pairs]
        documents = [qa.answer for qa in qa_pairs]
        metadatas = [{"country": qa.country, "topic": qa.topic} for qa in qa_pairs]

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        self._qa_lookup = {qa.id: qa for qa in qa_pairs}

    def predict(self, question: str, n_results: int = 3) -> dict[str, Any]:
        """Recupera texto relevante e gera predicao."""
        results = self.collection.query(query_texts=[question], n_results=n_results)

        # A 'resposta' e o documento mais relevante encontrado
        if results["documents"] and results["documents"][0]:
            predicted_answer = results["documents"][0][0]
            retrieved_ids = results["ids"][0] if results["ids"] else []
        else:
            predicted_answer = ""
            retrieved_ids = []

        return {
            "predicted_answer": predicted_answer,
            "predicted_modalities": ["text"],
            "retrieved_ids": retrieved_ids,
        }


class MultimodalRAG:
    """Baseline 2: RAG multimodal (texto + descricoes de imagem)."""

    def __init__(self, qa_pairs: list[QAPair], embed_fn: Any = None) -> None:
        """Inicializa o indice multimodal."""
        self.name = "Multimodal RAG"
        self.client = chromadb.Client()

        collection_kwargs: dict[str, Any] = {"name": "multimodal"}
        if embed_fn:
            collection_kwargs["embedding_function"] = embed_fn

        self.collection = self.client.create_collection(**collection_kwargs)

        # Indexa texto + descricao de imagem concatenada
        ids = [qa.id for qa in qa_pairs]
        documents = []
        for qa in qa_pairs:
            doc = qa.answer
            if qa.image_description:
                doc += f" [IMAGEM: {qa.image_description}]"
            documents.append(doc)

        metadatas = [
            {
                "country": qa.country,
                "topic": qa.topic,
                "has_image": qa.image_description is not None,
            }
            for qa in qa_pairs
        ]

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        self._qa_lookup = {qa.id: qa for qa in qa_pairs}

    def predict(self, question: str, n_results: int = 3) -> dict[str, Any]:
        """Recupera conteudo multimodal e gera predicao."""
        results = self.collection.query(query_texts=[question], n_results=n_results)

        if results["documents"] and results["documents"][0]:
            predicted_answer = results["documents"][0][0]
            # Determina modalidades com base no conteudo recuperado
            modalities = ["text"]
            if "[IMAGEM:" in predicted_answer:
                modalities.append("image")
            retrieved_ids = results["ids"][0] if results["ids"] else []
        else:
            predicted_answer = ""
            modalities = ["text"]
            retrieved_ids = []

        return {
            "predicted_answer": predicted_answer,
            "predicted_modalities": modalities,
            "retrieved_ids": retrieved_ids,
        }


class RandomModalityRAG:
    """Baseline 3: RAG com selecao aleatoria de modalidade."""

    def __init__(self, qa_pairs: list[QAPair], embed_fn: Any = None) -> None:
        """Inicializa o indice com selecao aleatoria."""
        self.name = "Random Modality RAG"
        self.client = chromadb.Client()

        collection_kwargs: dict[str, Any] = {"name": "random_modality"}
        if embed_fn:
            collection_kwargs["embedding_function"] = embed_fn

        self.collection = self.client.create_collection(**collection_kwargs)

        ids = [qa.id for qa in qa_pairs]
        documents = [qa.answer for qa in qa_pairs]
        metadatas = [{"country": qa.country, "topic": qa.topic} for qa in qa_pairs]

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def predict(self, question: str, n_results: int = 3) -> dict[str, Any]:
        """Recupera texto e atribui modalidade aleatoriamente."""
        results = self.collection.query(query_texts=[question], n_results=n_results)

        if results["documents"] and results["documents"][0]:
            predicted_answer = results["documents"][0][0]
            retrieved_ids = results["ids"][0] if results["ids"] else []
        else:
            predicted_answer = ""
            retrieved_ids = []

        # Selecao aleatoria de modalidades
        n_modalities = random.randint(1, len(ALL_MODALITIES))
        modalities = random.sample(ALL_MODALITIES, n_modalities)

        return {
            "predicted_answer": predicted_answer,
            "predicted_modalities": modalities,
            "retrieved_ids": retrieved_ids,
        }


def run_baseline(
    baseline,
    qa_pairs: list[QAPair],
) -> list[dict[str, Any]]:
    """Executa um baseline sobre todos os QA pairs."""
    predictions: list[dict[str, Any]] = []

    for qa in qa_pairs:
        result = baseline.predict(qa.question)
        predictions.append({
            "id": qa.id,
            "predicted_answer": result["predicted_answer"],
            "predicted_modalities": result["predicted_modalities"],
        })

    return predictions


def save_predictions(
    predictions: list[dict[str, Any]],
    path: Path,
) -> None:
    """Salva predicoes em formato JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


def compute_quick_metrics(
    predictions: list[dict[str, Any]],
    qa_lookup: dict[str, QAPair],
) -> dict[str, float]:
    """
    Calcula metricas rapidas sem carregar sentence-transformers.

    Retorna: modality_f1 medio e cobertura.
    """
    from evaluation.metrics import modality_appropriateness

    f1_scores: list[float] = []
    for pred in predictions:
        qa_id = pred["id"]
        if qa_id in qa_lookup:
            gold = qa_lookup[qa_id]
            f1 = modality_appropriateness(
                pred["predicted_modalities"], gold.modalities
            )
            f1_scores.append(f1)

    return {
        "modality_f1_mean": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "modality_f1_std": float(np.std(f1_scores)) if f1_scores else 0.0,
        "n_evaluated": len(f1_scores),
    }


def print_comparison_table(
    baseline_results: dict[str, dict[str, float]],
) -> None:
    """Imprime tabela comparativa dos baselines."""
    table = Table(
        title="Comparacao de Baselines - FinEduBench-Luso",
        show_lines=True,
    )
    table.add_column("Baseline", style="bold cyan")
    table.add_column("Modality F1 (mean)", justify="right")
    table.add_column("Modality F1 (std)", justify="right")
    table.add_column("N avaliados", justify="right")

    for name, metrics in baseline_results.items():
        table.add_row(
            name,
            f"{metrics['modality_f1_mean']:.4f}",
            f"{metrics['modality_f1_std']:.4f}",
            str(metrics["n_evaluated"]),
        )

    console.print(table)


def main() -> None:
    """Executa todos os baselines e compara resultados."""
    console.print("\n[bold magenta]===== FinEduBench-Luso: Baselines RAG =====[/bold magenta]\n")

    results_dir = _PROJECT_ROOT / "baselines" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar dataset
    console.print("[bold green]1. Carregando dataset...[/bold green]")
    qa_pairs = load_all_countries(_PROJECT_ROOT / "dataset")
    if not qa_pairs:
        console.print("[red]Dataset nao encontrado. Execute: python -m dataset.generator[/red]")
        sys.exit(1)
    console.print(f"  {len(qa_pairs)} QA pairs carregados.")
    qa_lookup = {qa.id: qa for qa in qa_pairs}

    # 2. Carregar embeddings
    console.print("\n[bold green]2. Carregando modelo de embeddings...[/bold green]")
    embed_fn = get_embedding_function()

    # 3. Executar baselines
    random.seed(42)  # Reprodutibilidade para baseline random

    baselines = {
        "Text-Only RAG": TextOnlyRAG(qa_pairs, embed_fn),
        "Multimodal RAG": MultimodalRAG(qa_pairs, embed_fn),
        "Random Modality RAG": RandomModalityRAG(qa_pairs, embed_fn),
    }

    all_baseline_results: dict[str, dict[str, float]] = {}

    for name, baseline in baselines.items():
        console.print(f"\n[bold green]3. Executando: {name}...[/bold green]")
        predictions = run_baseline(baseline, qa_pairs)

        # Salvar predicoes
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        save_path = results_dir / f"{safe_name}.jsonl"
        save_predictions(predictions, save_path)
        console.print(f"  Predicoes salvas: {save_path.name}")

        # Metricas rapidas
        metrics = compute_quick_metrics(predictions, qa_lookup)
        all_baseline_results[name] = metrics
        console.print(f"  Modality F1: {metrics['modality_f1_mean']:.4f} (+/- {metrics['modality_f1_std']:.4f})")

    # 4. Comparacao
    console.print("\n[bold green]4. Comparacao dos baselines:[/bold green]\n")
    print_comparison_table(all_baseline_results)

    # 5. Salvar resumo
    summary_path = results_dir / "baseline_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_baseline_results, f, indent=2, ensure_ascii=False)
    console.print(f"\n  Resumo salvo: {summary_path}")

    console.print("\n[bold magenta]Baselines concluidos![/bold magenta]")
    console.print(f"  Para avaliacao completa, execute:")
    console.print(f"  python -m evaluation.run_evaluation --predictions {results_dir}/\n")


if __name__ == "__main__":
    main()

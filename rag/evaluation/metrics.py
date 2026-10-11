"""
Metricas de avaliacao para o FinEduBench-Luso.

Implementa metricas de acuracia factual, relevancia cultural,
adequacao de modalidade, e detecao de alucinacao para
avaliacao de sistemas RAG em educacao financeira lusofona.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

# Modelos sao carregados sob demanda para evitar custo de inicializacao
_SENTENCE_MODEL = None
_SENTENCE_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _get_sentence_model():
    """Carrega o modelo de sentence-transformers sob demanda."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SENTENCE_MODEL = SentenceTransformer(_SENTENCE_MODEL_NAME)
    return _SENTENCE_MODEL


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula a similaridade cosseno entre dois vetores."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Termos culturais especificos por pais
# ---------------------------------------------------------------------------

CULTURAL_TERMS: dict[str, list[str]] = {
    "BR": [
        "pix", "tesouro direto", "tesouro selic", "tesouro ipca",
        "cdb", "lci", "lca", "fgc", "serasa", "cpf", "cnpj",
        "bolsa familia", "mei", "clt", "inss", "fgts",
        "selic", "cdi", "nubank", "consorcio", "real", "reais",
        "desenrola", "procon", "cvm", "b3", "ibovespa",
        "poupanca", "cheque especial", "parcelamento", "boleto",
        "holerite", "13o salario", "decimo terceiro",
    ],
    "PT": [
        "irs", "e-fatura", "efatura", "subsidio de ferias", "subsidio de natal",
        "certificados de aforro", "ppr", "euribor", "spread",
        "taeg", "tan", "erse", "deco", "imi", "imt", "iuc",
        "seguranca social", "nif", "chave movel digital",
        "duodecimos", "portal das financas", "persi",
        "fundo de garantia de depositos", "banco de portugal",
        "ctt", "multibanco", "mb way", "euro", "euros",
        "credito habitacao", "rsi", "abono de familia",
    ],
    "AO": [
        "kwanza", "kz", "multicaixa", "multicaixa express",
        "kixikila", "zungueira", "alembamento", "candongueiro",
        "kupapata", "bodiva", "bna", "banco nacional de angola",
        "bfa", "bai", "bpc", "bda", "prodesi",
        "ende", "epal", "agt", "nif", "duat",
        "inapem", "das", "musseque", "inagbe",
        "economia informal", "agiota",
    ],
}


def factual_accuracy(predicted: str, gold: str) -> float:
    """
    Mede a acuracia factual comparando similaridade semantica
    entre a resposta predita e a resposta de referencia (gold).

    Utiliza sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
    para calcular similaridade cosseno no espaco de embeddings.

    Args:
        predicted: resposta gerada pelo sistema
        gold: resposta de referencia (gold standard)

    Returns:
        Similaridade cosseno entre 0.0 e 1.0
    """
    model = _get_sentence_model()
    embeddings = model.encode([predicted, gold], convert_to_numpy=True)
    return _cosine_similarity(embeddings[0], embeddings[1])


def cultural_relevance(
    predicted: str,
    country: str,
    cultural_notes: str | None = None,
) -> float:
    """
    Avalia se a resposta contem termos culturalmente especificos do pais.

    Combina duas estrategias:
    1) Presenca de termos culturais do pais na resposta (50% do score)
    2) Se cultural_notes fornecido, similaridade semantica com as notas (50%)

    Args:
        predicted: resposta gerada pelo sistema
        country: codigo do pais (BR, PT, AO)
        cultural_notes: notas culturais de referencia (opcional)

    Returns:
        Score de relevancia cultural entre 0.0 e 1.0
    """
    predicted_lower = predicted.lower()
    terms = CULTURAL_TERMS.get(country, [])

    # Componente 1: presenca de termos culturais
    if terms:
        found = sum(1 for term in terms if term in predicted_lower)
        term_score = min(found / max(len(terms) * 0.1, 1), 1.0)
    else:
        term_score = 0.0

    # Componente 2: similaridade com notas culturais
    if cultural_notes:
        semantic_score = factual_accuracy(predicted, cultural_notes)
    else:
        # Sem notas culturais, o score depende apenas dos termos
        return term_score

    return 0.5 * term_score + 0.5 * semantic_score


def modality_appropriateness(
    predicted_modalities: list[str],
    gold_modalities: list[str],
) -> float:
    """
    Calcula F1 score entre modalidades preditas e modalidades de referencia.

    Args:
        predicted_modalities: lista de modalidades preditas (ex: ["text", "image"])
        gold_modalities: lista de modalidades de referencia

    Returns:
        F1 score entre 0.0 e 1.0
    """
    pred_set = set(predicted_modalities)
    gold_set = set(gold_modalities)

    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0

    true_positives = len(pred_set & gold_set)
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(gold_set) if gold_set else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def hallucination_score(predicted: str, context: str) -> float:
    """
    Mede o grau de alucinacao na resposta predita.

    Calcula 1 - similaridade_semantica(predicted, context), ou seja,
    quanto MENOS a resposta esta fundamentada no contexto, maior o score.

    Um score alto indica MAIS alucinacao (menos fundamentacao).

    Args:
        predicted: resposta gerada pelo sistema
        context: contexto de referencia (resposta gold ou documentos fonte)

    Returns:
        Score de alucinacao entre 0.0 (sem alucinacao) e 1.0 (completa alucinacao)
    """
    if not context.strip() or not predicted.strip():
        return 1.0

    model = _get_sentence_model()

    # Divide em sentencas para analise mais granular
    pred_sentences = _split_sentences(predicted)
    if not pred_sentences:
        return 1.0

    context_embedding = model.encode([context], convert_to_numpy=True)[0]
    pred_embeddings = model.encode(pred_sentences, convert_to_numpy=True)

    # Para cada sentenca predita, calcula similaridade com o contexto
    similarities = [
        _cosine_similarity(emb, context_embedding)
        for emb in pred_embeddings
    ]

    # Media de fundamentacao; alucinacao = 1 - fundamentacao
    grounding = float(np.mean(similarities))
    return 1.0 - grounding


def _split_sentences(text: str) -> list[str]:
    """Divide texto em sentencas de forma simples."""
    sentences = re.split(r'[.!?;]\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


class BenchmarkEvaluator:
    """
    Avaliador completo do benchmark FinEduBench-Luso.

    Executa todas as metricas sobre uma lista de predicoes e
    retorna resultados agregados como DataFrame.
    """

    def __init__(self) -> None:
        """Inicializa o avaliador."""
        self.results: list[dict[str, Any]] = []

    def evaluate_single(
        self,
        qa_id: str,
        predicted_answer: str,
        gold_answer: str,
        predicted_modalities: list[str],
        gold_modalities: list[str],
        country: str,
        topic: str,
        difficulty: str,
        cultural_notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Avalia uma unica predicao contra a referencia.

        Returns:
            Dicionario com scores de todas as metricas.
        """
        result = {
            "id": qa_id,
            "country": country,
            "topic": topic,
            "difficulty": difficulty,
            "factual_accuracy": factual_accuracy(predicted_answer, gold_answer),
            "cultural_relevance": cultural_relevance(
                predicted_answer, country, cultural_notes
            ),
            "modality_f1": modality_appropriateness(
                predicted_modalities, gold_modalities
            ),
            "hallucination": hallucination_score(predicted_answer, gold_answer),
        }
        self.results.append(result)
        return result

    def evaluate_batch(
        self,
        predictions: list[dict[str, Any]],
        gold_data: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Avalia um lote de predicoes contra os dados de referencia.

        Args:
            predictions: lista de dicts com {id, predicted_answer, predicted_modalities}
            gold_data: lista de dicts com os campos do QAPair

        Returns:
            DataFrame com resultados por item.
        """
        gold_lookup = {g["id"]: g for g in gold_data}

        for pred in predictions:
            qa_id = pred["id"]
            if qa_id not in gold_lookup:
                print(f"  [AVISO] QA id '{qa_id}' nao encontrado nos dados de referencia. Ignorando.")
                continue

            gold = gold_lookup[qa_id]
            self.evaluate_single(
                qa_id=qa_id,
                predicted_answer=pred.get("predicted_answer", ""),
                gold_answer=gold.get("answer", ""),
                predicted_modalities=pred.get("predicted_modalities", []),
                gold_modalities=gold.get("modalities", []),
                country=gold.get("country", ""),
                topic=gold.get("topic", ""),
                difficulty=gold.get("difficulty", ""),
                cultural_notes=gold.get("cultural_notes"),
            )

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """Converte resultados para DataFrame."""
        return pd.DataFrame(self.results)

    def aggregate_by(self, column: str) -> pd.DataFrame:
        """
        Agrega metricas por uma coluna (country, topic, difficulty).

        Returns:
            DataFrame com media e desvio padrao por grupo.
        """
        df = self.to_dataframe()
        if df.empty:
            return df

        metric_cols = ["factual_accuracy", "cultural_relevance", "modality_f1", "hallucination"]
        agg_funcs = {col: ["mean", "std", "count"] for col in metric_cols if col in df.columns}
        grouped = df.groupby(column).agg(agg_funcs)
        grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
        return grouped

    def summary(self) -> dict[str, pd.DataFrame]:
        """
        Retorna resumo completo com agregacoes por pais, topico e dificuldade.

        Returns:
            Dicionario com DataFrames: 'by_country', 'by_topic', 'by_difficulty', 'overall'
        """
        df = self.to_dataframe()
        if df.empty:
            return {"by_country": df, "by_topic": df, "by_difficulty": df, "overall": df}

        metric_cols = ["factual_accuracy", "cultural_relevance", "modality_f1", "hallucination"]

        return {
            "by_country": self.aggregate_by("country"),
            "by_topic": self.aggregate_by("topic"),
            "by_difficulty": self.aggregate_by("difficulty"),
            "overall": df[metric_cols].describe(),
        }

    def to_latex(self, by: str = "country") -> str:
        """
        Gera tabela LaTeX com resultados agregados.

        Args:
            by: coluna para agrupar ('country', 'topic', 'difficulty')

        Returns:
            String com tabela LaTeX.
        """
        df = self.aggregate_by(by)
        if df.empty:
            return "% Sem resultados para gerar tabela."

        # Colunas de media
        mean_cols = [c for c in df.columns if c.endswith("_mean")]
        df_display = df[mean_cols].copy()
        df_display.columns = [c.replace("_mean", "") for c in mean_cols]

        latex = df_display.to_latex(
            float_format="%.3f",
            caption=f"Resultados do FinEduBench-Luso agrupados por {by}.",
            label=f"tab:finedu-{by}",
        )
        return latex

    def reset(self) -> None:
        """Limpa resultados anteriores."""
        self.results = []

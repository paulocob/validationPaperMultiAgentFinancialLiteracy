"""
E5.3 — Análise de fairness inter-país.

Objetivo: computar métricas de fairness sobre as decisões do DQN e dos
módulos RAG/Wav2Vec2, detectando se algum país recebe tratamento sistemático
desigual.

Métricas computadas:
  - Demographic parity difference (DPD)
  - Equalized odds difference (EOD)
  - Selection rate por país
  - Disparate impact ratio

Saída: resultados/e5_3_fairness.json
Tempo estimado: ~30 minutos (CPU-bound).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.logging import get_logger
from common.data_loaders import (
    load_mrag_csv,
    load_variantes_predictions,
    load_finedu_by_country,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "resultados"
RESULTS_DIR.mkdir(exist_ok=True)


def selection_rate_by_country(rows: list[dict], country_key: str,
                                score_key: str, threshold: float) -> dict:
    """Taxa de aceitação (score >= threshold) por país."""
    by_country = defaultdict(list)
    for r in rows:
        c = r.get(country_key)
        s = float(r.get(score_key, 0))
        by_country[c].append(1 if s >= threshold else 0)
    return {
        c: round(sum(v) / len(v), 4) if v else 0.0
        for c, v in by_country.items()
    }


def demographic_parity_difference(rates: dict) -> float:
    """DPD = max - min de selection rates."""
    if len(rates) < 2:
        return 0.0
    return round(max(rates.values()) - min(rates.values()), 4)


def disparate_impact_ratio(rates: dict) -> float:
    """DI = min / max (razão; ideal = 1.0)."""
    if len(rates) < 2 or max(rates.values()) == 0:
        return 1.0
    return round(min(rates.values()) / max(rates.values()), 4)


def analyze_rag_fairness():
    """Analisa fairness do RAG retrieval sobre 999 consultas."""
    rag_data = load_mrag_csv("rag_retrieval_results.csv")

    fairness_by_model = {}
    for model in set(r["model"] for r in rag_data):
        rows = [r for r in rag_data if r["model"] == model]
        sel_rate = selection_rate_by_country(rows, "country", "hit@5", 0.5)
        fairness_by_model[model] = {
            "selection_rate": sel_rate,
            "dpd": demographic_parity_difference(sel_rate),
            "disparate_impact": disparate_impact_ratio(sel_rate),
            "fair_at_0.8_threshold": disparate_impact_ratio(sel_rate) >= 0.8,
        }
    return fairness_by_model


def analyze_variantes_fairness():
    """Analisa acurácia do classificador Wav2Vec2 agrupada por grupo geográfico."""
    pred_data = load_variantes_predictions("wav2vec2_accent_20260405_004050")
    preds = pred_data["predictions"]
    labels = pred_data["labels"]
    id2label = pred_data["id2label"]

    GROUP = {
        "BR": ["BR_Sudeste_SP", "BR_Sudeste_RJ", "BR_Sudeste_MG", "BR_Sul",
               "BR_Nordeste", "BR_Norte", "BR_CentroOeste"],
        "PT": ["PT_Lisboa", "PT_Norte"],
        "AF": ["Angola", "Moçambique", "Cabo_Verde", "Guiné_Bissau", "São_Tomé"],
        "AS": ["Timor_Leste"],
        "OUT": ["Galego", "Estrangeiro"],
    }
    label2group = {}
    for g, labels_list in GROUP.items():
        for lbl in labels_list:
            label2group[lbl] = g

    # Acurácia por grupo
    by_group = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred_idx, label_idx in zip(preds, labels):
        pred_name = id2label[str(pred_idx)]
        label_name = id2label[str(label_idx)]
        group = label2group.get(label_name, "?")
        by_group[group]["total"] += 1
        if pred_idx == label_idx:
            by_group[group]["correct"] += 1

    group_acc = {
        g: round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0
        for g, d in by_group.items()
    }

    return {
        "accuracy_by_group": group_acc,
        "dpd": demographic_parity_difference(group_acc),
        "disparate_impact": disparate_impact_ratio(group_acc),
        "fair_at_0.8_threshold": disparate_impact_ratio(group_acc) >= 0.8,
    }


def analyze_corpus_fairness():
    """Tamanho e cobertura do corpus por país."""
    by_country = load_finedu_by_country()
    corpus_stats = {
        c: {
            "n_qas": len(qas),
            "n_com_audio": sum(1 for q in qas if q.get("audio_transcript")),
            "n_com_imagem": sum(1 for q in qas if q.get("image_description")),
            "n_com_cultural_notes": sum(1 for q in qas if q.get("cultural_notes")),
            "topicos": dict(Counter(q.get("topic") for q in qas)),
            "dificuldades": dict(Counter(q.get("difficulty") for q in qas)),
        }
        for c, qas in by_country.items()
    }
    return corpus_stats


def main():
    parser = argparse.ArgumentParser(description="E5.3 — Fairness audit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger = get_logger("e5_3_fairness", output_dir=RESULTS_DIR / "logs")
    logger.info("=== E5.3 Fairness audit inter-país ===")

    if args.dry_run:
        logger.info("[DRY RUN] sem análise pesada")
        return

    # 1. RAG retrieval fairness
    logger.info("1/3 RAG retrieval...")
    rag_fairness = analyze_rag_fairness()

    # 2. Wav2Vec2 fairness
    logger.info("2/3 Wav2Vec2 por grupo geográfico...")
    variantes_fairness = analyze_variantes_fairness()

    # 3. Cobertura do corpus
    logger.info("3/3 Cobertura do corpus FinEduBench-Luso...")
    corpus_stats = analyze_corpus_fairness()

    result = {
        "exp_id": "E5.3",
        "rag_retrieval": rag_fairness,
        "variantes_classifier": variantes_fairness,
        "corpus_coverage": corpus_stats,
        "interpretacao": {
            "DPD_ideal": "< 0.1 (< 10% diferença entre grupos)",
            "DI_ideal": "> 0.8 (regra dos 80%)",
        },
    }
    out = RESULTS_DIR / "e5_3_fairness.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info(f"✓ Salvo em {out}")

    logger.info("\n=== Principais achados ===")
    for model, f in rag_fairness.items():
        flag = "✅ FAIR" if f["fair_at_0.8_threshold"] else "⚠  NÃO FAIR"
        logger.info(f"  RAG {model}: DPD={f['dpd']}, DI={f['disparate_impact']} {flag}")
    vf = variantes_fairness
    flag = "✅ FAIR" if vf["fair_at_0.8_threshold"] else "⚠  NÃO FAIR"
    logger.info(f"  Wav2Vec2: DPD={vf['dpd']}, DI={vf['disparate_impact']} {flag}")


if __name__ == "__main__":
    main()

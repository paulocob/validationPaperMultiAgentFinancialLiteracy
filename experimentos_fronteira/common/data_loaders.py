"""
Carregadores unificados para os datasets da tese.

Todos resolvem paths relativos à raiz do repositório de validação,
independente de onde o script é executado. Na origem, os caminhos apontavam
para a estrutura de pastas do repositório de escrita da tese.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def _repo_root() -> Path:
    """Resolve a raiz do repositório de validação."""
    here = Path(__file__).resolve()
    # <raiz>/experimentos_fronteira/common/data_loaders.py
    return here.parent.parent.parent


# ======================= treino_DRL =======================

def _drl_data_dir() -> Path:
    return _repo_root() / "drl" / "data"


def load_interactions() -> list[dict]:
    """Carrega as 2.000 interações reais (state, action, reward, next_state)."""
    path = _drl_data_dir() / "interactions.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_users() -> list[dict]:
    """Carrega 25 perfis de aprendiz."""
    path = _drl_data_dir() / "users.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tasks() -> dict:
    """Carrega tasks.json (17 tarefas + grafo de precedências)."""
    path = _drl_data_dir() / "tasks.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_splits() -> dict:
    """Carrega splits.json (train/val/test com interaction_ids)."""
    path = _drl_data_dir() / "splits.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ======================= FinEduBench-Luso =======================

def _finedu_dir() -> Path:
    return _repo_root() / "rag" / "dataset"


def load_finedu_bench_luso(
    countries: tuple[str, ...] = ("BR", "PT", "AO"),
) -> list[dict]:
    """Carrega todos os pares QA do FinEduBench-Luso, concatenados."""
    entries: list[dict] = []
    for c in countries:
        path = _finedu_dir() / c / f"{c.lower()}_seed.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            entries.extend(json.loads(line) for line in f if line.strip())
    return entries


def load_finedu_by_country() -> dict[str, list[dict]]:
    """Mesmo que acima, mas organizado por país."""
    out = {}
    for c in ("BR", "PT", "AO"):
        path = _finedu_dir() / c / f"{c.lower()}_seed.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                out[c] = [json.loads(line) for line in f if line.strip()]
    return out


# ======================= Variantes Lusófonas =======================

def _variantes_dir() -> Path:
    # O repositório do classificador ainda não foi depositado (ver fala/README.md).
    # A variável VARIANTES_DIR indica onde está a cópia local dele.
    import os
    return Path(os.environ.get("VARIANTES_DIR", str(_repo_root() / "sbrt2026-lusophone-variants")))


def load_variantes_predictions(run_id: str) -> dict:
    """Carrega predictions.json e results.json de um run do classificador."""
    run_path = _variantes_dir() / "models" / run_id
    preds = json.loads((run_path / "test_predictions.json").read_text(encoding="utf-8"))
    results = json.loads((run_path / "results.json").read_text(encoding="utf-8"))
    return {
        "predictions": preds["predictions"],
        "labels": preds["labels"],
        "id2label": results.get("id2label", {}),
        "label2id": results.get("label2id", {}),
        "model_name": results.get("model_name"),
    }


# ======================= CSVs de MRAG =======================

def _mrag_results_dir() -> Path:
    return _repo_root() / "rag" / "embeddings" / "results"


def load_mrag_csv(name: str) -> list[dict]:
    """Carrega um CSV do diretório de resultados do MRAG."""
    import csv
    path = _mrag_results_dir() / name
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    print("=== Smoke test data_loaders ===")
    print(f"repo_root: {_repo_root()}")
    try:
        ints = load_interactions()
        print(f"✓ interactions: {len(ints)} (esperado 2000)")
    except Exception as e:
        print(f"✗ interactions: {e}")
    try:
        bench = load_finedu_bench_luso()
        print(f"✓ finedu_bench_luso: {len(bench)} (esperado 999)")
    except Exception as e:
        print(f"✗ finedu_bench_luso: {e}")

"""
E4.5_TSNE_UMAP — E4.5 Embedding t-SNE/UMAP das 17 classes

NOTA: Este é um esqueleto. Ver plano_de_trabalho.md Seção 2 (trilha correspondente)
para hipótese, metodologia completa e critérios de aceitação.

Saída esperada: resultados/e4_5_tsne_umap.json

Para completar a implementação:
  1. Ler documentação da trilha no plano_de_trabalho.md
  2. Importar common.stats, common.hardware, common.logging
  3. Usar common.data_loaders para carregar dados
  4. Seguir padrão de saída JSON com exp_id, métricas, IC95%, teste estatístico
  5. Prefixar execução com: bash scripts/run_template.sh trilha4_fala/e4_5_tsne_umap.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.hardware import print_hardware_summary
from common.logging import get_logger

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "resultados"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="E4.5 Embedding t-SNE/UMAP das 17 classes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print_hardware_summary()
    logger = get_logger("e4_5_tsne_umap", output_dir=RESULTS_DIR / "logs")
    logger.info("=== E4.5_TSNE_UMAP === ")
    logger.warning("⚠  Esqueleto. Complete a implementação seguindo o plano_de_trabalho.md.")

    result = {
        "exp_id": "E4.5_TSNE_UMAP",
        "status": "skeleton",
        "note": "Implementação pendente; ver plano_de_trabalho.md.",
    }
    out = RESULTS_DIR / "e4_5_tsne_umap.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info(f"✓ Esqueleto salvo em {out}")


if __name__ == "__main__":
    main()

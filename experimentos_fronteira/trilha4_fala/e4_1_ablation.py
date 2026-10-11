"""
E4.1 — Ablation completa do classificador Wav2Vec2 (Cap. 7 da tese).

Hipótese: cada componente do setup (class_weights, label_smoothing, warmup,
grad_clipping) contribui marginalmente; identificar o mais decisivo.

Metodologia:
  - 5 variantes: baseline + 4 ablações (uma removendo cada componente).
  - Mesmo split treino/val/teste do pipeline original.
  - Reportar impacto em acurácia, F1 macro, F1 por classe.

Saída: resultados/e4_1_ablation_wav2vec2.json
Tempo estimado em M4 Pro: 5 × 3h31min ≈ 18h (rodar overnight).

Uso:
  # Rodar tudo de madrugada
  bash scripts/run_template.sh trilha4_fala/e4_1_ablation.py

  # Só 1 variante (para teste)
  python trilha4_fala/e4_1_ablation.py --only baseline
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.logging import get_logger
from common.hardware import print_hardware_summary

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent  # Escrita Doutorado/
VARIANTES_DIR = REPO_ROOT / "sbrt2026-lusophone-variants"
RESULTS_DIR = ROOT / "resultados"
RESULTS_DIR.mkdir(exist_ok=True)


ABLATIONS = {
    "baseline": {
        "class_weights": True,
        "label_smoothing": 0.1,
        "warmup_ratio": 0.1,
        "grad_clip": 1.0,
    },
    "no_class_weights": {
        "class_weights": False,
        "label_smoothing": 0.1,
        "warmup_ratio": 0.1,
        "grad_clip": 1.0,
    },
    "no_label_smoothing": {
        "class_weights": True,
        "label_smoothing": 0.0,
        "warmup_ratio": 0.1,
        "grad_clip": 1.0,
    },
    "no_warmup": {
        "class_weights": True,
        "label_smoothing": 0.1,
        "warmup_ratio": 0.0,
        "grad_clip": 1.0,
    },
    "no_grad_clip": {
        "class_weights": True,
        "label_smoothing": 0.1,
        "warmup_ratio": 0.1,
        "grad_clip": None,
    },
}


def run_variant(variant_name: str, config: dict, dry_run: bool = False) -> dict:
    """
    Invoca o script de treino existente da tese com overrides de configuração.
    Assume que `sbrt2026-lusophone-variants/src/train.py` aceita argumentos
    para sobrescrever config.py. Se não aceitar, ajustar config.py via .env.
    """
    logger = get_logger(f"e4_1_{variant_name}", output_dir=RESULTS_DIR / "logs")
    logger.info(f"=== Variante: {variant_name} ===")
    logger.info(f"Config: {config}")

    train_script = VARIANTES_DIR / "src" / "train.py"
    if not train_script.exists():
        logger.error(f"Script não encontrado: {train_script}")
        return {"variant": variant_name, "error": "train.py não encontrado"}

    output_dir = VARIANTES_DIR / "models" / f"ablation_{variant_name}"
    cmd = [
        sys.executable, str(train_script),
        "--output_dir", str(output_dir),
        "--seed", "42",
    ]
    # Se train.py suporta flags por linha de comando:
    if config["class_weights"] is False:
        cmd.append("--no-class-weights")
    if config["label_smoothing"] == 0.0:
        cmd.extend(["--label-smoothing", "0.0"])
    if config["warmup_ratio"] == 0.0:
        cmd.extend(["--warmup-ratio", "0.0"])
    if config["grad_clip"] is None:
        cmd.append("--no-grad-clip")

    if dry_run:
        logger.info(f"[DRY RUN] {' '.join(cmd)}")
        return {"variant": variant_name, "dry_run": True}

    t0 = time.time()
    try:
        subprocess.check_call(cmd, cwd=VARIANTES_DIR)
        elapsed = time.time() - t0
    except subprocess.CalledProcessError as e:
        logger.error(f"Treino falhou: {e}")
        return {"variant": variant_name, "error": str(e)}

    # Ler métricas
    results_file = output_dir / "results.json"
    metrics = {}
    if results_file.exists():
        metrics = json.loads(results_file.read_text())
    return {
        "variant": variant_name,
        "config": config,
        "elapsed_hours": round(elapsed / 3600, 2),
        "test_accuracy": metrics.get("test_accuracy"),
        "best_val_accuracy": metrics.get("best_val_accuracy"),
    }


def main():
    parser = argparse.ArgumentParser(description="E4.1 — Ablation Wav2Vec2")
    parser.add_argument("--only", choices=list(ABLATIONS.keys()) + ["all"],
                        default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print_hardware_summary()
    logger = get_logger("e4_1_ablation_master", output_dir=RESULTS_DIR / "logs")

    variants = list(ABLATIONS.keys()) if args.only == "all" else [args.only]
    logger.info(f"Rodando {len(variants)} variante(s): {variants}")

    all_results = []
    for v in variants:
        result = run_variant(v, ABLATIONS[v], dry_run=args.dry_run)
        all_results.append(result)

    # Análise comparativa
    summary = {
        "exp_id": "E4.1",
        "variants": all_results,
    }
    if not args.dry_run:
        baseline_acc = next(
            (r["test_accuracy"] for r in all_results
             if r.get("variant") == "baseline" and r.get("test_accuracy") is not None),
            None,
        )
        if baseline_acc:
            summary["baseline_accuracy"] = baseline_acc
            summary["delta_from_baseline"] = {
                r["variant"]: round(baseline_acc - r["test_accuracy"], 4)
                for r in all_results
                if r.get("test_accuracy") is not None and r["variant"] != "baseline"
            }

    out = RESULTS_DIR / "e4_1_ablation_wav2vec2.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"✓ Resultados salvos em {out}")

    if not args.dry_run and "delta_from_baseline" in summary:
        logger.info("=== Impacto por componente (queda em acurácia) ===")
        for v, delta in sorted(summary["delta_from_baseline"].items(),
                                key=lambda x: -x[1]):
            logger.info(f"  {v}: Δ = {delta:+.4f}")


if __name__ == "__main__":
    main()

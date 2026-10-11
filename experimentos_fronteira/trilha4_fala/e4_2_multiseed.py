"""
E4.2 — Multi-seed do classificador Wav2Vec2 (Cap. 7 da tese).

Hipótese: 91,5% de acurácia atual é consistente entre seeds ou é uma
execução favorável?

Metodologia:
  - 5 seeds (42, 43, 44, 45, 46).
  - Reportar média ± desvio-padrão.
  - Teste de Friedman entre seeds para validar consistência.

Saída: resultados/e4_2_multiseed_wav2vec2.json
Tempo estimado em M4 Pro: 5 × 3h31min ≈ 18h (rodar overnight).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.logging import get_logger
from common.hardware import print_hardware_summary
from common.stats import bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
VARIANTES_DIR = REPO_ROOT / "sbrt2026-lusophone-variants"
RESULTS_DIR = ROOT / "resultados"
RESULTS_DIR.mkdir(exist_ok=True)


def train_one_seed(seed: int, dry_run: bool = False) -> dict:
    train_script = VARIANTES_DIR / "src" / "train.py"
    output_dir = VARIANTES_DIR / "models" / f"seed_{seed}"

    cmd = [
        sys.executable, str(train_script),
        "--seed", str(seed),
        "--output_dir", str(output_dir),
    ]
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return {"seed": seed, "dry_run": True}

    t0 = time.time()
    subprocess.check_call(cmd, cwd=VARIANTES_DIR)
    elapsed = time.time() - t0

    # Carrega métricas
    results_file = output_dir / "results.json"
    if results_file.exists():
        metrics = json.loads(results_file.read_text())
        return {
            "seed": seed,
            "test_accuracy": metrics.get("test_accuracy"),
            "best_val_accuracy": metrics.get("best_val_accuracy"),
            "best_epoch": metrics.get("best_epoch"),
            "elapsed_hours": round(elapsed / 3600, 2),
        }
    return {"seed": seed, "error": "results.json não encontrado"}


def main():
    parser = argparse.ArgumentParser(description="E4.2 — Multi-seed Wav2Vec2")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print_hardware_summary()
    logger = get_logger("e4_2_multiseed", output_dir=RESULTS_DIR / "logs")
    logger.info(f"Multi-seed Wav2Vec2 — {len(args.seeds)} seeds: {args.seeds}")
    logger.info(f"Tempo estimado: {len(args.seeds) * 3.5:.0f}h (1 seed ≈ 3h31min)")

    per_seed = []
    for s in args.seeds:
        logger.info(f"→ Seed {s}")
        r = train_one_seed(s, dry_run=args.dry_run)
        per_seed.append(r)

    # Agregar
    agg = {}
    if not args.dry_run:
        accs = [r["test_accuracy"] for r in per_seed if r.get("test_accuracy") is not None]
        if len(accs) >= 2:
            mean, lo, hi = bootstrap_ci(accs)
            agg = {
                "mean": round(statistics.mean(accs), 4),
                "std": round(statistics.stdev(accs), 4),
                "ci95_bootstrap": [round(lo, 4), round(hi, 4)],
                "min": round(min(accs), 4),
                "max": round(max(accs), 4),
                "n": len(accs),
            }

    out = RESULTS_DIR / "e4_2_multiseed_wav2vec2.json"
    out.write_text(json.dumps({
        "exp_id": "E4.2",
        "seeds": args.seeds,
        "per_seed": per_seed,
        "aggregate": agg,
    }, indent=2, ensure_ascii=False))
    logger.info(f"✓ Salvo em {out}")
    if agg:
        logger.info(f"Acurácia: {agg['mean']:.4f} ± {agg['std']:.4f} "
                    f"(IC95% [{agg['ci95_bootstrap'][0]:.4f}, {agg['ci95_bootstrap'][1]:.4f}])")


if __name__ == "__main__":
    main()

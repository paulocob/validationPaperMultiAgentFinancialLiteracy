"""
E4.3 — Data augmentation para classes com poucos dados (Cap. 7 da tese).

Hipótese: SpecAugment + speed perturbation (±10%) + copy-paste reduz o gap
das classes com n_test < 20 (PT-Lisboa, São Tomé, Guiné-Bissau).

Meta: F1 ≥ 0.85 para as 3 classes pequenas.

Saída: resultados/e4_3_augmentation.json
Tempo estimado em M4 Pro: ~4-5h para 1 treino com aug.
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
REPO_ROOT = ROOT.parent
VARIANTES_DIR = REPO_ROOT / "sbrt2026-lusophone-variants"
RESULTS_DIR = ROOT / "resultados"
RESULTS_DIR.mkdir(exist_ok=True)


AUGMENTATION_CONFIGS = {
    "spec_augment_only": {
        "time_masking": True,
        "frequency_masking": True,
        "speed_perturbation": False,
        "copy_paste": False,
    },
    "speed_pert_only": {
        "time_masking": False,
        "frequency_masking": False,
        "speed_perturbation": True,  # ±10%
        "copy_paste": False,
    },
    "copy_paste_only": {
        "time_masking": False,
        "frequency_masking": False,
        "speed_perturbation": False,
        "copy_paste": True,  # duplica amostras das classes pequenas
    },
    "all_combined": {
        "time_masking": True,
        "frequency_masking": True,
        "speed_perturbation": True,
        "copy_paste": True,
    },
}

SMALL_CLASSES = ["PT-Lisboa", "São Tomé", "Guiné-Bissau"]


def run_aug_variant(name: str, config: dict, dry_run: bool = False) -> dict:
    """
    Roda treino com augmentation específica.

    NOTA: Assume que o script src/train.py aceita flags de augmentation.
    Se não aceitar, ver `src/dataset_loader.py` para implementar
    o augmentation no DataLoader antes do batch.
    """
    logger = get_logger(f"e4_3_{name}", output_dir=RESULTS_DIR / "logs")
    logger.info(f"Augmentation: {name}")

    train_script = VARIANTES_DIR / "src" / "train.py"
    output_dir = VARIANTES_DIR / "models" / f"aug_{name}"
    cmd = [
        sys.executable, str(train_script),
        "--seed", "42",
        "--output_dir", str(output_dir),
    ]
    if config["time_masking"]:
        cmd.append("--time-masking")
    if config["frequency_masking"]:
        cmd.append("--freq-masking")
    if config["speed_perturbation"]:
        cmd.extend(["--speed-pert", "0.1"])
    if config["copy_paste"]:
        cmd.append("--copy-paste-small")

    if dry_run:
        logger.info(f"[DRY RUN] {' '.join(cmd)}")
        return {"name": name, "dry_run": True}

    t0 = time.time()
    try:
        subprocess.check_call(cmd, cwd=VARIANTES_DIR)
        elapsed = time.time() - t0
    except subprocess.CalledProcessError as e:
        logger.error(f"Treino falhou: {e}")
        return {"name": name, "error": str(e)}

    # Carrega resultados
    results_file = output_dir / "results.json"
    evaluation_file = output_dir / "evaluation_report.txt"

    result = {
        "name": name,
        "config": config,
        "elapsed_hours": round(elapsed / 3600, 2),
    }
    if results_file.exists():
        m = json.loads(results_file.read_text())
        result["test_accuracy"] = m.get("test_accuracy")
        result["macro_f1"] = m.get("macro_f1")

    # Tenta extrair F1 das 3 classes pequenas do evaluation_report.txt
    # (formato textual do sklearn classification_report)
    if evaluation_file.exists():
        txt = evaluation_file.read_text()
        result["small_classes_f1"] = {}
        for cls in SMALL_CLASSES:
            # Procurar linha começando com o nome da classe
            for line in txt.splitlines():
                if line.strip().startswith(cls):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            result["small_classes_f1"][cls] = float(parts[-2])
                        except ValueError:
                            pass
                    break
    return result


def main():
    parser = argparse.ArgumentParser(description="E4.3 — Data augmentation")
    parser.add_argument("--variants", nargs="+",
                        choices=list(AUGMENTATION_CONFIGS.keys()) + ["all"],
                        default=["all_combined"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print_hardware_summary()

    variants = args.variants
    if "all" in variants:
        variants = list(AUGMENTATION_CONFIGS.keys())

    all_results = []
    for v in variants:
        all_results.append(run_aug_variant(v, AUGMENTATION_CONFIGS[v],
                                           dry_run=args.dry_run))

    out = RESULTS_DIR / "e4_3_augmentation.json"
    out.write_text(json.dumps({
        "exp_id": "E4.3",
        "target_classes": SMALL_CLASSES,
        "target_f1_threshold": 0.85,
        "variants": all_results,
    }, indent=2, ensure_ascii=False))
    print(f"✓ Salvo em {out}")


if __name__ == "__main__":
    main()

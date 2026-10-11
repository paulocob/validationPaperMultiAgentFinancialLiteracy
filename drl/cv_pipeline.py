import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


def setup_cv_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "cv_pipeline.log"

    logger = logging.getLogger("cv_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_interaction_ids(data_dir: Path) -> list[str]:
    interactions_path = data_dir / "interactions.jsonl"
    ids = []
    with interactions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ids.append(item["interaction_id"])
    return ids


def build_folds(ids: list[str], k: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    fold_size = max(1, len(shuffled) // k)
    folds = []
    for i in range(k):
        start = i * fold_size
        if i == k - 1:
            fold = shuffled[start:]
        else:
            fold = shuffled[start : start + fold_size]
        folds.append(fold)
    return folds


def save_splits(data_dir: Path, train_ids: list[str], val_ids: list[str], test_ids: list[str]) -> None:
    payload = {"train": train_ids, "val": val_ids, "test": test_ids}
    splits_path = data_dir / "splits.json"
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, separators=(",", ": "))
        f.write("\n")


def run_step(logger: logging.Logger, name: str, cmd: list[str]) -> None:
    logger.info("Inicio: %s", name)
    logger.info("Comando: %s", " ".join(cmd))
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.perf_counter() - start

    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info("[%s][stdout] %s", name, line)
    if result.stderr:
        for line in result.stderr.splitlines():
            logger.warning("[%s][stderr] %s", name, line)

    logger.info("Fim: %s | exit_code=%s | duracao=%.2fs", name, result.returncode, duration)
    if result.returncode != 0:
        raise SystemExit(f"Falha no passo: {name}")


def aggregate_metrics(results: dict) -> dict:
    summary = {}
    for model_name, folds in results.items():
        metrics_keys = {}
        for fold_metrics in folds:
            for key, value in fold_metrics.items():
                if isinstance(value, (int, float)):
                    metrics_keys.setdefault(key, []).append(float(value))

        model_summary = {}
        for key, values in metrics_keys.items():
            if not values:
                continue
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            model_summary[key] = {"mean": mean, "std": var ** 0.5}
        summary[model_name] = model_summary
    return summary


def load_metrics(model_dir: Path) -> dict:
    metrics_path = model_dir / "eval_metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validacao cruzada para DRL")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.k < 3:
        raise SystemExit("k precisa ser >= 3 para ter train/val/test por fold.")

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    cv_dir = models_dir / "cv"
    logger = setup_cv_logger(cv_dir)
    logger.info("CV iniciado | k=%s seed=%s device=%s", args.k, args.seed, args.device)

    original_splits = data_dir / "splits.json"
    backup_splits = data_dir / "splits.json.bak"
    if original_splits.exists():
        shutil.copy2(original_splits, backup_splits)

    try:
        ids = load_interaction_ids(data_dir)
        folds = build_folds(ids, args.k, args.seed)

        python = sys.executable
        results = {"dqn": [], "ppo": [], "cql": []}

        for i in range(args.k):
            test_ids = folds[i]
            val_ids = folds[(i + 1) % args.k]
            train_ids = []
            for j, fold in enumerate(folds):
                if j not in (i, (i + 1) % args.k):
                    train_ids.extend(fold)

            save_splits(data_dir, train_ids, val_ids, test_ids)

            fold_dir = cv_dir / f"fold_{i + 1:02d}"
            dqn_dir = fold_dir / "dqn"
            ppo_dir = fold_dir / "ppo"
            cql_dir = fold_dir / "cql"

            logger.info(
                "Fold %s/%s | train=%s val=%s test=%s",
                i + 1,
                args.k,
                len(train_ids),
                len(val_ids),
                len(test_ids),
            )

            steps = [
                ("train_dqn", [python, str(base_dir / "train_dqn.py"), "--device", args.device, "--seed", str(args.seed), "--output-dir", str(dqn_dir)]),
                ("train_ppo", [python, str(base_dir / "train_ppo.py"), "--device", args.device, "--seed", str(args.seed), "--output-dir", str(ppo_dir)]),
                ("train_cql", [python, str(base_dir / "train_cql.py"), "--device", args.device, "--seed", str(args.seed), "--output-dir", str(cql_dir)]),
                ("eval_dqn", [python, str(base_dir / "eval_dqn.py"), "--device", args.device, "--seed", str(args.seed), "--model-dir", str(dqn_dir)]),
                ("eval_ppo", [python, str(base_dir / "eval_ppo.py"), "--device", args.device, "--seed", str(args.seed), "--model-dir", str(ppo_dir)]),
                ("eval_cql", [python, str(base_dir / "eval_cql.py"), "--device", args.device, "--seed", str(args.seed), "--model-dir", str(cql_dir)]),
            ]

            for name, cmd in steps:
                run_step(logger, f"fold_{i + 1:02d}_{name}", cmd)

            results["dqn"].append(load_metrics(dqn_dir))
            results["ppo"].append(load_metrics(ppo_dir))
            results["cql"].append(load_metrics(cql_dir))

        summary = aggregate_metrics(results)
        summary_path = cv_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("Resumo salvo em %s", summary_path)
        logger.info("CV concluido com sucesso")
    finally:
        if backup_splits.exists():
            shutil.move(backup_splits, original_splits)


if __name__ == "__main__":
    main()

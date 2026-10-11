import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "hparam_search.log"

    logger = logging.getLogger("hparam_search")
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


def load_metrics(model_dir: Path) -> dict:
    metrics_path = model_dir / "eval_metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_best(results: list[dict], key: str) -> dict:
    ranked = [r for r in results if key in r.get("metrics", {})]
    if not ranked:
        return {}
    ranked.sort(key=lambda item: item["metrics"][key], reverse=True)
    return ranked[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca simples de hiperparametros")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models" / "hparam"
    logger = setup_logger(models_dir)
    logger.info("Busca iniciada | device=%s seed=%s", args.device, args.seed)

    python = sys.executable
    results = {"ppo": [], "cql": []}

    ppo_configs = [
        {"name": "ppo_e60_clip0.2_ent0.01", "epochs": 60, "clip_eps": 0.2, "ent_coef": 0.01},
        {"name": "ppo_e60_clip0.1_ent0.005", "epochs": 60, "clip_eps": 0.1, "ent_coef": 0.005},
        {"name": "ppo_e40_clip0.1_ent0.001", "epochs": 40, "clip_eps": 0.1, "ent_coef": 0.001},
        {"name": "ppo_e40_clip0.2_ent0.001", "epochs": 40, "clip_eps": 0.2, "ent_coef": 0.001},
    ]

    for cfg in ppo_configs:
        model_dir = models_dir / "ppo" / cfg["name"]
        train_cmd = [
            python,
            str(base_dir / "train_ppo.py"),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--epochs",
            str(cfg["epochs"]),
            "--clip-eps",
            str(cfg["clip_eps"]),
            "--ent-coef",
            str(cfg["ent_coef"]),
            "--output-dir",
            str(model_dir),
        ]
        eval_cmd = [
            python,
            str(base_dir / "eval_ppo.py"),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--model-dir",
            str(model_dir),
        ]
        run_step(logger, f"ppo_train_{cfg['name']}", train_cmd)
        run_step(logger, f"ppo_eval_{cfg['name']}", eval_cmd)
        metrics = load_metrics(model_dir)
        results["ppo"].append({"config": cfg, "metrics": metrics})

    cql_configs = [
        {"name": "cql_e60_alpha0.5_tu10", "epochs": 60, "alpha": 0.5, "target_update": 10},
        {"name": "cql_e60_alpha0.2_tu10", "epochs": 60, "alpha": 0.2, "target_update": 10},
        {"name": "cql_e60_alpha0.2_tu20", "epochs": 60, "alpha": 0.2, "target_update": 20},
        {"name": "cql_e40_alpha0.1_tu20", "epochs": 40, "alpha": 0.1, "target_update": 20},
    ]

    for cfg in cql_configs:
        model_dir = models_dir / "cql" / cfg["name"]
        train_cmd = [
            python,
            str(base_dir / "train_cql.py"),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--epochs",
            str(cfg["epochs"]),
            "--alpha",
            str(cfg["alpha"]),
            "--target-update",
            str(cfg["target_update"]),
            "--output-dir",
            str(model_dir),
        ]
        eval_cmd = [
            python,
            str(base_dir / "eval_cql.py"),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--model-dir",
            str(model_dir),
        ]
        run_step(logger, f"cql_train_{cfg['name']}", train_cmd)
        run_step(logger, f"cql_eval_{cfg['name']}", eval_cmd)
        metrics = load_metrics(model_dir)
        results["cql"].append({"config": cfg, "metrics": metrics})

    summary = {
        "ppo": {
            "runs": results["ppo"],
            "best_primary": pick_best(results["ppo"], "primary_value"),
            "best_top1": pick_best(results["ppo"], "top1_accuracy"),
            "best_top5": pick_best(results["ppo"], "top5_accuracy"),
        },
        "cql": {
            "runs": results["cql"],
            "best_primary": pick_best(results["cql"], "primary_value"),
            "best_top1": pick_best(results["cql"], "top1_accuracy"),
            "best_top5": pick_best(results["cql"], "top5_accuracy"),
        },
    }

    summary_path = models_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Resumo salvo em %s", summary_path)
    logger.info("Busca concluida com sucesso")


if __name__ == "__main__":
    main()

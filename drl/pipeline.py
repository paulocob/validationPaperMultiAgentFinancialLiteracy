import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path


def setup_pipeline_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"

    logger = logging.getLogger("pipeline")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline treino + avaliacao DRL")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refined-reward", action="store_true")
    parser.add_argument("--refined-m2-boost", type=float, default=0.05)
    parser.add_argument("--balance-actions", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    logger = setup_pipeline_logger(models_dir)
    logger.info("Pipeline iniciado")
    logger.info("Device=%s Seed=%s", args.device, args.seed)

    python = sys.executable
    reward_flag = ["--refined-reward"] if args.refined_reward else []
    if args.refined_reward and args.refined_m2_boost:
        reward_flag += ["--refined-m2-boost", str(args.refined_m2_boost)]
    balance_flag = ["--balance-actions"] if args.balance_actions else []
    steps = [
        ("train_dqn", [python, str(base_dir / "train_dqn.py"), "--device", args.device, "--seed", str(args.seed), *reward_flag, *balance_flag]),
        ("train_ppo", [python, str(base_dir / "train_ppo.py"), "--device", args.device, "--seed", str(args.seed), *reward_flag, *balance_flag]),
        ("train_cql", [python, str(base_dir / "train_cql.py"), "--device", args.device, "--seed", str(args.seed), *reward_flag, *balance_flag]),
        ("eval_dqn", [python, str(base_dir / "eval_dqn.py"), "--device", args.device, "--seed", str(args.seed), *reward_flag]),
        ("eval_ppo", [python, str(base_dir / "eval_ppo.py"), "--device", args.device, "--seed", str(args.seed), *reward_flag]),
        ("eval_cql", [python, str(base_dir / "eval_cql.py"), "--device", args.device, "--seed", str(args.seed), *reward_flag]),
    ]

    for name, cmd in steps:
        run_step(logger, name, cmd)

    logger.info("Pipeline concluido com sucesso")


if __name__ == "__main__":
    main()

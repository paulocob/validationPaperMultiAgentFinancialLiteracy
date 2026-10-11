"""Logger padronizado para experimentos."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def get_logger(
    name: str,
    output_dir: Path | str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Cria logger com formatação padrão.
    Se `output_dir` for dado, também escreve em arquivo timestamped.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Arquivo (opcional)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(out / f"{name}_{ts}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


if __name__ == "__main__":
    log = get_logger("smoke_test", output_dir="/tmp/exp_logs")
    log.info("Teste do logger")
    log.warning("Pode haver pressão de memória")

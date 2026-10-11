"""Biblioteca compartilhada entre experimentos de fronteira."""
from .stats import (
    wilson_interval,
    bootstrap_ci,
    mcnemar_test,
    wilcoxon_signed_rank,
    cohens_d,
    cohens_h,
    cliffs_delta,
)
from .hardware import (
    check_mps,
    limit_mps_memory,
    cleanup_mps,
    device,
)
from .logging import get_logger
from .data_loaders import (
    load_interactions,
    load_users,
    load_tasks,
    load_splits,
    load_finedu_bench_luso,
)

__all__ = [
    "wilson_interval",
    "bootstrap_ci",
    "mcnemar_test",
    "wilcoxon_signed_rank",
    "cohens_d",
    "cohens_h",
    "cliffs_delta",
    "check_mps",
    "limit_mps_memory",
    "cleanup_mps",
    "device",
    "get_logger",
    "load_interactions",
    "load_users",
    "load_tasks",
    "load_splits",
    "load_finedu_bench_luso",
]

"""
Utilitários estatísticos compartilhados entre experimentos.

Funções:
  - wilson_interval: IC de Wilson para proporções
  - bootstrap_ci: IC percentil por bootstrap
  - mcnemar_test: McNemar (amostras pareadas, classificadores)
  - wilcoxon_signed_rank: Wilcoxon pareado
  - cohens_d: tamanho de efeito para médias
  - cohens_h: tamanho de efeito para proporções
  - cliffs_delta: tamanho de efeito não-paramétrico
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Iterable, Sequence


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC binomial de Wilson (95% com z=1.96)."""
    if n == 0:
        return 0.0, 0.0
    phat = correct / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))) / denom
    return center - margin, center + margin


def bootstrap_ci(
    values: Sequence[float],
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI para a média. Retorna (media, lo, hi)."""
    rng = random.Random(seed)
    if not values:
        return 0.0, 0.0, 0.0
    values_list = list(values)
    n = len(values_list)
    means = []
    for _ in range(n_iter):
        sample = [values_list[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int((1 - ci) / 2 * n_iter)]
    hi = means[int((1 + ci) / 2 * n_iter) - 1]
    return statistics.mean(values_list), lo, hi


def mcnemar_test(
    y_true: Sequence,
    pred_a: Sequence,
    pred_b: Sequence,
) -> dict:
    """
    Teste de McNemar com correção de continuidade.
    Retorna dict com b, c, chi2 e p_value.

    b = A acerta, B erra
    c = A erra, B acerta
    """
    b = sum(1 for t, a, ba in zip(y_true, pred_a, pred_b) if a == t and ba != t)
    c = sum(1 for t, a, ba in zip(y_true, pred_a, pred_b) if a != t and ba == t)
    if b + c == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p_value": 1.0}
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))
    return {"b": b, "c": c, "chi2": round(chi2, 3), "p_value": round(p, 6)}


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> dict:
    """Wilcoxon signed-rank pareado (implementação sem scipy)."""
    diffs = [(x - y) for x, y in zip(a, b) if x != y]
    if not diffs:
        return {"n_nonzero": 0, "W": 0, "z": 0.0, "p_value": 1.0}
    abs_diffs = [abs(d) for d in diffs]
    ranks: dict[float, float] = {}
    sorted_vals = sorted(abs_diffs)
    current_rank = 1
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (current_rank + current_rank + (j - i) - 1) / 2
        for k in range(i, j):
            ranks[sorted_vals[k]] = avg_rank
        current_rank += (j - i)
        i = j
    W_plus = sum(ranks[abs(d)] for d in diffs if d > 0)
    W_minus = sum(ranks[abs(d)] for d in diffs if d < 0)
    W = min(W_plus, W_minus)
    n = len(diffs)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (W - mu) / sigma if sigma > 0 else 0
    p = math.erfc(abs(z) / math.sqrt(2))
    return {
        "n_nonzero": n,
        "W": W,
        "W_plus": W_plus,
        "W_minus": W_minus,
        "z": round(z, 3),
        "p_value": round(p, 6),
    }


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """d de Cohen para duas médias, com desvio-padrão agrupado."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma = statistics.mean(a)
    mb = statistics.mean(b)
    sa = statistics.stdev(a)
    sb = statistics.stdev(b)
    sp = math.sqrt(((len(a) - 1) * sa ** 2 + (len(b) - 1) * sb ** 2) / (len(a) + len(b) - 2))
    return round((ma - mb) / sp, 3) if sp > 0 else 0.0


def cohens_h(p1: float, p2: float) -> float:
    """h de Cohen para duas proporções."""
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return round(phi1 - phi2, 3)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta: tamanho de efeito não-paramétrico em [-1, 1]."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return round((gt - lt) / (n1 * n2), 3)


def interpret_effect_size(value: float, kind: str = "d") -> str:
    """Rótulo textual para tamanho de efeito."""
    v = abs(value)
    if kind == "d" or kind == "h":
        if v < 0.2:
            return "trivial"
        elif v < 0.5:
            return "pequeno"
        elif v < 0.8:
            return "medio"
        else:
            return "grande"
    elif kind == "cliffs":
        if v < 0.147:
            return "trivial"
        elif v < 0.33:
            return "pequeno"
        elif v < 0.474:
            return "medio"
        else:
            return "grande"
    return "n/a"


if __name__ == "__main__":
    # Smoke test
    correct, n = 675, 737
    lo, hi = wilson_interval(correct, n)
    print(f"Wilson IC95% de {correct}/{n} = {correct/n:.4f} [{lo:.4f}, {hi:.4f}]")

    vals = [0.85, 0.87, 0.83, 0.89, 0.86]
    mean, lo, hi = bootstrap_ci(vals, n_iter=1000)
    print(f"Bootstrap CI: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    print(f"Cohen's h(0.85, 0.75) = {cohens_h(0.85, 0.75)}")
    print(f"Cohen's d entre [0.5,0.6,0.7] e [0.3,0.4,0.5] = "
          f"{cohens_d([0.5,0.6,0.7], [0.3,0.4,0.5])}")

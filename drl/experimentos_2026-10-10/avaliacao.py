"""Métricas por política (E4): Top-k, recompensa associada, teto, OPE (SNIS e
DR) com intervalos por bootstrap sobre transições.

Todas as políticas avaliadas são determinísticas (argmax), então os pesos de
importância são w_i = 1[π_e(s_i) = a_i] / π_b(a_i | s_i). π_b é estimada por
contagem empírica no próprio conjunto avaliado quando o estado (vetor
codificado) aparece ao menos duas vezes, e pela BC (softmax) nos demais casos.
π_b é truncada em PI_B_MIN para limitar o peso máximo (documentado).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from dados import Conjunto
from politicas import scores

PI_B_MIN = 0.01
N_BOOT = 1000
BOOT_SEED = 0


def ranking_from_scores(sc: np.ndarray, k: int = 3) -> np.ndarray:
    return np.argsort(-sc, axis=1)[:, :k]


def topk_hits(rank: np.ndarray, actions: np.ndarray) -> Dict[int, np.ndarray]:
    return {k: (rank[:, :k] == actions[:, None]).any(1) for k in (1, 3)}


def teto(rewards: np.ndarray, frac: float) -> float:
    """Média das k maiores recompensas do conjunto, k = round(frac·N)."""
    k = int(round(frac * len(rewards)))
    if k <= 0:
        return float("nan")
    return float(np.sort(rewards)[::-1][:k].mean())


def behavior_policy(conj: Conjunto, bc_model, actions: np.ndarray) -> Dict[str, np.ndarray]:
    """π_b(a_i | s_i): contagem por estado (≥ 2 observações) ou BC."""
    keys = conj.state_keys()
    counts: Dict[bytes, np.ndarray] = {}
    for key, a in zip(keys, actions):
        counts.setdefault(key, np.zeros(11))[a] += 1
    pb_bc = torch.softmax(torch.tensor(scores(bc_model, conj.states)), 1).numpy()
    pb = np.empty(len(actions))
    from_counts = np.zeros(len(actions), dtype=bool)
    for i, (key, a) in enumerate(zip(keys, actions)):
        c = counts[key]
        if c.sum() >= 2:
            pb[i] = c[a] / c.sum()
            from_counts[i] = True
        else:
            pb[i] = pb_bc[i, a]
    return {"pi_b": np.maximum(pb, PI_B_MIN), "from_counts": from_counts}


def per_transition(rank: np.ndarray, conj: Conjunto, pi_b: np.ndarray,
                   rhat: Optional[np.ndarray] = None, q_cql: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Quantidades por transição das quais todos os estimadores são razões de somas."""
    a = conj.actions
    r = conj.rewards
    pred = rank[:, 0]
    match = (pred == a).astype(np.float64)
    w = match / pi_b
    # A regra sequencial não tem predição em estados terminais (pred = −1); o
    # termo direto do DR é então a média condicional aos estados com predição,
    # coerente com o SNIS, que só vê transições com peso não nulo.
    defined = (pred >= 0).astype(np.float64)
    out = {"match": match, "top3": (rank[:, :3] == a[:, None]).any(1).astype(np.float64),
           "r": r, "w": w, "wr": w * r, "defined": defined}
    if rhat is not None:
        rh_pi = rhat[np.arange(len(a)), np.where(pred >= 0, pred, 0)] * (pred >= 0)
        rh_a = rhat[np.arange(len(a)), a]
        out["dr_term"] = rh_pi + w * (r - rh_a)
        out["dr_direct"] = rh_pi
        out["dr_corr"] = w * (r - rh_a)
    if q_cql is not None:
        q_pi = q_cql[np.arange(len(a)), np.where(pred >= 0, pred, 0)] * (pred >= 0)
        q_a = q_cql[np.arange(len(a)), a]
        out["drq_direct"] = q_pi
        out["drq_corr"] = w * (r - q_a)
    return out


def estimates(pt: Dict[str, np.ndarray], idx: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Estimadores pontuais; `idx` (B×N) vetoriza o bootstrap."""
    def S(key):
        v = pt[key]
        return v[idx].sum(1) if idx is not None else v.sum()
    n = (idx.shape[1] if idx is not None else len(pt["r"]))
    nm = S("match")
    sw = S("w")
    out = {
        "top1": nm / n,
        "top3": S("top3") / n,
        "reward_assoc": np.where(nm > 0, S("wr_unw") / np.maximum(nm, 1), np.nan) if "wr_unw" in pt else None,
        "snis": np.where(sw > 0, S("wr") / np.maximum(sw, 1e-12), np.nan),
        "reward_logged": S("r") / n,
    }
    if "dr_term" in pt:
        # DR auto-normalizado: termo direto médio + correção IS com pesos normalizados a N.
        out["dr"] = S("dr_direct") / np.maximum(S("defined"), 1) + np.where(sw > 0, S("dr_corr") / np.maximum(sw, 1e-12), 0.0)
    if "drq_direct" in pt:
        out["dr_q_cql"] = S("drq_direct") / np.maximum(S("defined"), 1) + np.where(sw > 0, S("drq_corr") / np.maximum(sw, 1e-12), 0.0)
    return out


def evaluate_policy(rank: np.ndarray, conj: Conjunto, pi_b: np.ndarray,
                    rhat: Optional[np.ndarray] = None, q_cql: Optional[np.ndarray] = None,
                    bootstrap: bool = True) -> Dict:
    pt = per_transition(rank, conj, pi_b, rhat, q_cql)
    pt["wr_unw"] = pt["match"] * pt["r"]
    point = {k: float(v) for k, v in estimates(pt).items() if v is not None}
    point["teto"] = teto(conj.rewards, point["top1"])
    point["pct_teto"] = point["reward_assoc"] / point["teto"] if point["teto"] > 0 else float("nan")
    w = pt["w"]
    point["ess"] = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))
    point["w_max"] = float(w.max())
    point["n"] = int(len(conj))
    res = {"point": point, "per_transition": pt}
    if bootstrap:
        rng = np.random.default_rng(BOOT_SEED)
        idx = rng.integers(0, len(conj), size=(N_BOOT, len(conj)))
        boot = estimates(pt, idx)
        ci = {}
        for k, v in boot.items():
            if v is None:
                continue
            v = np.asarray(v, dtype=np.float64)
            ci[k] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        res["ci95"] = ci
    return res


def pooled_ci(pts: List[Dict[str, np.ndarray]], n_boot: int = N_BOOT) -> Dict[str, List[float]]:
    """IC por bootstrap de transições quando o estimador é a média sobre
    várias execuções (sementes) avaliadas no MESMO conjunto: reamostram-se os
    índices uma vez e aplicam-se a todas as execuções."""
    n = len(pts[0]["r"])
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(n_boot, n))
    acc: Dict[str, List[np.ndarray]] = {}
    for pt in pts:
        for k, v in estimates(pt, idx).items():
            if v is not None:
                acc.setdefault(k, []).append(np.asarray(v, dtype=np.float64))
    return {k: [float(np.nanpercentile(np.nanmean(vs, 0), 2.5)), float(np.nanpercentile(np.nanmean(vs, 0), 97.5))]
            for k, vs in acc.items()}


def concat_pt(pts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in pts]) for k in pts[0]}


def by_group(rank: np.ndarray, conj: Conjunto, groups: List[str]) -> Dict[str, Dict[str, float]]:
    hit = rank[:, 0] == conj.actions
    out = {}
    for g in sorted(set(groups)):
        m = np.array([x == g for x in groups])
        out[g] = {"top1": float(hit[m].mean()), "n": int(m.sum())}
    return out

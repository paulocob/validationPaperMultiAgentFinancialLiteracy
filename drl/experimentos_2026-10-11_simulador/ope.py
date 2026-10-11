"""Validação da OPE (A2): os estimadores de 10/10 aplicados a logs do currículo + ε.

Mesmo desenho dos experimentos de 10/10: estimativa POR TRANSIÇÃO (um passo) do
valor de recompensa imediata de uma política determinística π_e,
    SNIS = Σ w_i r_i / Σ w_i,   w_i = 1[π_e(s_i) = a_i] / π_b(a_i | s_i)
    DR   = média_i r̂(s_i, π_e(s_i)) + Σ w_i (r_i − r̂(s_i, a_i)) / Σ w_i
com uma diferença: aqui π_b é CONHECIDA exatamente (0,8·1[a = currículo(s)] + 0,2/11),
o que favorece a OPE em relação a 10/10, onde π_b era estimada.

O "valor verdadeiro" correspondente é a recompensa média por passo quando π_e é
implantada no simulador (média sobre todas as decisões da população de teste). A
diferença entre os dois mede, entre outras coisas, o viés de distribuição de
estados: a estimativa por transição usa os estados visitados pelo currículo, não
os que π_e visitaria. Também se compara a ordenação da OPE com a do Hake verdadeiro.

IC 95 % por bootstrap de APRENDIZES (reamostragem por conglomerado), porque as
transições de um mesmo aprendiz não são independentes.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

N_BOOT_OPE = 1000


def estimar(log: Dict[str, np.ndarray], acao_pi: np.ndarray, pi_b: np.ndarray, rhat: np.ndarray,
            n_boot: int = N_BOOT_OPE, seed: int = 0) -> Dict:
    a, r, apr = log["acao"], log["rec"].astype(np.float64), log["aprendiz"]
    idx = np.arange(len(a))
    w = (acao_pi == a) / pi_b
    termo_dir = rhat[idx, acao_pi]
    corr = w * (r - rhat[idx, a])
    # somas por aprendiz → bootstrap por conglomerado vetorizado
    ids, inv = np.unique(apr, return_inverse=True)
    n_apr = len(ids)
    soma = lambda v: np.bincount(inv, v, minlength=n_apr)  # noqa: E731
    Swr, Sw, Sdir, Scorr, Sn = soma(w * r), soma(w), soma(termo_dir), soma(corr), soma(np.ones(len(a)))

    def est(pesos):
        sw = pesos @ Sw
        snis = (pesos @ Swr) / np.maximum(sw, 1e-12)
        dr = (pesos @ Sdir) / (pesos @ Sn) + (pesos @ Scorr) / np.maximum(sw, 1e-12)
        return snis, dr

    snis, dr = est(np.ones(n_apr))
    rng = np.random.default_rng(seed)
    B = rng.multinomial(n_apr, np.full(n_apr, 1.0 / n_apr), size=n_boot).astype(np.float64)
    bs, bd = est(B)  # B: (n_boot, n_aprendizes) contagens de reamostragem → (n_boot,)
    return {
        "snis": float(snis), "dr": float(dr),
        "snis_ic": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "dr_ic": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
        "ess": float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12)),
        "frac_concord": float((acao_pi == a).mean()),
        "n_trans": int(len(a)),
    }

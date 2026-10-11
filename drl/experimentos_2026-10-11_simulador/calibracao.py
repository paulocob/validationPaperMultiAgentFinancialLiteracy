"""Calibração (S6, S4): dois parâmetros, duas âncoras.

1. c_T (escala de P(T)) para que o currículo fixo produza Hake médio 0,50 em
   h = 0,5. Âncora: piloto brasileiro (g = 0,51, tutor). É a ÚNICA âncora
   empírica do simulador; os demais parâmetros são plausíveis, não estimados.
2. alfa_abandono para que o currículo fixo abandone 15 % em h = 1 (centro da
   faixa de 10 % a 20 % da especificação). Não é âncora empírica: é a escolha de
   desenho que impede o mecanismo de abandono de dominar a comparação.

Como os dois alvos dependem dos dois parâmetros (P(T) maior reduz a frustração e
portanto o abandono), a calibração alterna bisseções até estabilizar. Usa uma
população própria (semente 999, disjunta das de treino e teste).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from politicas_sim import Curriculo, simular
from simulador import Ambiente, ParamsSim, amostrar_populacao, carregar_catalogo

ALVO_HAKE, H_HAKE = 0.50, 0.5
ALVO_ABANDONO, H_ABANDONO = 0.15, 1.0
SEMENTE_CALIB = 999


def _curriculo(cat, p, h, n):
    return simular(Ambiente(cat, p, amostrar_populacao(n, h, SEMENTE_CALIB, p)), Curriculo())


def _bissecao(f, lo, hi, alvo, crescente=True, it=22):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if (f(mid) < alvo) == crescente:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrar(n: int = 20000, rodadas: int = 3, log=print) -> ParamsSim:
    cat = carregar_catalogo()
    p = ParamsSim()
    for r in range(rodadas):
        c_T = _bissecao(lambda c: _curriculo(cat, replace(p, c_T=c), H_HAKE, n)["hake"].mean(), 0.2, 1.2, ALVO_HAKE)
        p = replace(p, c_T=c_T)
        alfa = _bissecao(lambda a: _curriculo(cat, replace(p, alfa_abandono=a), H_ABANDONO, n)["abandono"].mean(),
                         0.0, 0.2, ALVO_ABANDONO)
        p = replace(p, alfa_abandono=alfa)
        d05, d1 = _curriculo(cat, p, H_HAKE, n), _curriculo(cat, p, H_ABANDONO, n)
        log(f"[calibração] rodada {r + 1}: c_T = {c_T:.4f}, alfa = {alfa:.4f}; currículo h=0,5: Hake "
            f"{d05['hake'].mean():.3f}; h=1: abandono {d1['abandono'].mean():.3f}")
    return p


if __name__ == "__main__":
    calibrar()

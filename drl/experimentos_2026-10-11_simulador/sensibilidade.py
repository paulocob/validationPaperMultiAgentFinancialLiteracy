"""Análise de sensibilidade aos três parâmetros NÃO calibrados do simulador.

Grade: multiplicador de P(T) com pré-requisito não dominado {0,2; 0,3; 0,5} ×
limiar de pré-requisito {0,5; 0,6; 0,7} × escala de dispersão {0,75; 1; 1,25}
(multiplica todos os dp de heterogeneidade, sd_*; em h = 1 é a dispersão máxima).

Em cada combinação, c_T é recalibrado por bisseção para manter o Hake do currículo
em 0,50 em h = 0,5, porque sem isso uma mudança de parâmetro alteraria ao mesmo
tempo o nível de aprendizagem e a vantagem da adaptação. α (abandono) fica no
valor calibrado da execução principal; o abandono do currículo em h = 1 passa a
variar e é reportado.

Para cada combinação, em h = 1, 3 sementes × 1.000 aprendizes: currículo,
LinUCB, DQN online, guloso pelo rastreador e oráculo míope. Sementes e
populações iguais às da execução principal (experimento._sementes com h_idx = 2),
de modo que a combinação central reproduz as sementes 0–2 da tabela principal.

Uso: python sensibilidade.py [--quick] [--workers 15]
Saída: chave "sensibilidade" em resultados.json, seção 9 do RESULTADOS.md e
fig_sim_sensibilidade.pdf.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # uma thread de BLAS por processo (ver run_all.py)

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import calibracao  # noqa: E402
import experimento  # noqa: E402
import politicas_sim as PS  # noqa: E402
from simulador import Ambiente, ParamsSim, amostrar_populacao, carregar_catalogo  # noqa: E402

MULTS = [0.2, 0.3, 0.5]
LIMIARES = [0.5, 0.6, 0.7]
ESCALAS = [0.75, 1.0, 1.25]
SEEDS = [0, 1, 2]
H = 1.0
H_IDX = 2  # índice de h = 1 na grade principal → mesmas sementes da execução principal
CAMPOS_SD = ["sd_L0_aprendiz", "sd_L0_habilidade", "sd_T_aprendiz", "sd_T_habilidade", "sd_S", "sd_G", "sd_F"]
POLS = ["curriculo", "linucb", "dqn_on", "guloso_kt", "oraculo"]


def _params_base() -> ParamsSim:
    R = json.loads((AQUI / "resultados.json").read_text(encoding="utf-8"))
    return replace(ParamsSim(), **{k: (tuple(v) if isinstance(v, list) else v) for k, v in R["params"].items()})


def _combo_params(base: ParamsSim, mult, limiar, escala) -> ParamsSim:
    p = replace(base, mult_prereq=mult, limiar_prereq=limiar)
    return replace(p, **{c: getattr(base, c) * escala for c in CAMPOS_SD})


def rodar_combo(mult, limiar, escala, base_dict, cfg):
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    base = replace(ParamsSim(), **{k: (tuple(v) if isinstance(v, list) else v) for k, v in base_dict.items()})
    cat = carregar_catalogo()
    p = _combo_params(base, mult, limiar, escala)
    # Recalibração de c_T (Hake do currículo = 0,50 em h = 0,5), α fixo.
    c_T = calibracao._bissecao(
        lambda c: calibracao._curriculo(cat, replace(p, c_T=c), calibracao.H_HAKE, cfg["n_calib"])["hake"].mean(),
        0.2, 1.5, calibracao.ALVO_HAKE)
    p = replace(p, c_T=c_T)
    hake_cal = float(calibracao._curriculo(cat, p, calibracao.H_HAKE, cfg["n_calib"])["hake"].mean())
    am = lambda n, hh, ss: amostrar_populacao(n, hh, ss, p)  # noqa: E731
    por_seed = []
    for s in cfg["seeds"]:
        sem = experimento._sementes(H_IDX, s)
        pop = am(cfg["n_teste"], H, sem["teste"])
        res = {}

        def av(nome, pol):
            d = PS.simular(Ambiente(cat, p, pop), pol)
            res[nome] = {k: d[k].astype(np.float32) for k in ("hake", "hake_ret", "abandono", "n_diverg", "n_decisoes")}

        av("curriculo", PS.Curriculo())
        av("oraculo", PS.OraculoMiope())
        av("guloso_kt", PS.GulosoRastreador())
        b = PS.BanditLinear("ucb", sem["bandit"])
        for j in range(cfg["n_bandit"] // 100):
            PS.simular(Ambiente(cat, p, am(100, H, sem["bandit"] * 10 + j)), b, aprendiz_online=b.aprender)
        b.congelado = True
        av("linucb", b)
        q, _ = PS.treinar_dqn_online(cat, p, H, H_IDX * 100 + s, am(cfg["n_val_on"], H, sem["val_on"]), am,
                                     dict(PS.CFG_DQN_ON, aprendizes=cfg["n_online"]))
        av("dqn_on", PS.PoliticaQ(q, "dqn_on"))
        por_seed.append(res)
    return {"mult": mult, "limiar": limiar, "escala": escala, "c_T": c_T, "hake_calib_h05": hake_cal,
            "por_seed": por_seed, "tempo_s": time.time() - t0}


def resumir(c, rng):
    """Médias agrupadas nas sementes e diferenças pareadas vs currículo (IC bootstrap)."""
    cat_ = lambda pol, k: np.concatenate([r[pol][k] for r in c["por_seed"]]).astype(np.float64)  # noqa: E731
    base = cat_("curriculo", "hake")
    out = {"mult": c["mult"], "limiar": c["limiar"], "escala": c["escala"], "c_T": c["c_T"],
           "hake_calib_h05": c["hake_calib_h05"], "tempo_s": c["tempo_s"], "n": int(len(base)), "politicas": {}}
    for pol in POLS:
        h = cat_(pol, "hake")
        e = {"hake": float(h.mean()), "abandono": float(cat_(pol, "abandono").mean()),
             "diverg": float(cat_(pol, "n_diverg").sum() / cat_(pol, "n_decisoes").sum())}
        if pol != "curriculo":
            dif = h - base
            m = rng.choice(dif, size=(1000, len(dif)), replace=True).mean(1)
            e["dg"] = float(dif.mean())
            e["dg_ic"] = [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
            e["d"] = float(dif.mean() / np.sqrt((h.var(ddof=1) + base.var(ddof=1)) / 2))
        out["politicas"][pol] = e
    o = out["politicas"]["oraculo"]["dg"]
    melhor = max(out["politicas"][x]["dg"] for x in ("linucb", "dqn_on", "guloso_kt"))
    out["frac_teto_perdida_dqn_on"] = 1 - out["politicas"]["dqn_on"]["dg"] / o if o > 0 else float("nan")
    out["frac_teto_perdida_melhor"] = 1 - melhor / o if o > 0 else float("nan")
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=15)
    a = ap.parse_args()
    cfg = dict(n_teste=1000, seeds=SEEDS, n_bandit=2000, n_val_on=500, n_online=24000, n_calib=20000)
    combos = list(itertools.product(MULTS, LIMIARES, ESCALAS))
    if a.quick:
        cfg.update(n_teste=200, seeds=[0], n_bandit=300, n_val_on=100, n_online=512, n_calib=3000)
        combos = [(0.3, 0.6, 1.0), (0.2, 0.7, 1.25)]
    base = _params_base()
    t0 = time.time()
    brutos = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(rodar_combo, m, l, e, base.as_dict(), cfg) for m, l, e in combos]
        for f in as_completed(futs):
            c = f.result()
            brutos.append(c)
            print(time.strftime("%H:%M:%S"), f"mult={c['mult']} limiar={c['limiar']} escala={c['escala']}: "
                  f"c_T={c['c_T']:.4f} ({c['tempo_s']:.0f} s)", flush=True)
    rng = np.random.default_rng(54321)
    linhas = [resumir(c, rng) for c in sorted(brutos, key=lambda c: (c["mult"], c["limiar"], c["escala"]))]
    dgs = {pol: [l["politicas"][pol]["dg"] for l in linhas] for pol in POLS if pol != "curriculo"}
    sens = {
        "grade": {"mult_prereq": MULTS, "limiar_prereq": LIMIARES, "escala_dispersao": ESCALAS},
        "cfg": cfg, "h": H, "alfa_abandono_fixo": base.alfa_abandono, "quick": a.quick,
        "combinacoes": linhas,
        "faixas_dg": {pol: [float(min(v)), float(max(v))] for pol, v in dgs.items()},
        "faixa_frac_teto_perdida_dqn_on": [float(np.nanmin([l["frac_teto_perdida_dqn_on"] for l in linhas])),
                                           float(np.nanmax([l["frac_teto_perdida_dqn_on"] for l in linhas]))],
        "faixa_frac_teto_perdida_melhor": [float(np.nanmin([l["frac_teto_perdida_melhor"] for l in linhas])),
                                           float(np.nanmax([l["frac_teto_perdida_melhor"] for l in linhas]))],
        "tempo_total_s": time.time() - t0,
    }
    if a.quick:
        (AQUI / "sensibilidade_quick.json").write_text(json.dumps(sens, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: sens[k] for k in ("faixas_dg", "faixa_frac_teto_perdida_melhor")}, ensure_ascii=False))
        return
    R = json.loads((AQUI / "resultados.json").read_text(encoding="utf-8"))
    R["sensibilidade"] = sens
    (AQUI / "resultados.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    import figuras_sim
    import relatorio_sim
    figuras_sim.fig_sensibilidade(R, figuras_sim.FIG_DIR / "fig_sim_sensibilidade.pdf")
    notas = AQUI / "NOTAS_INTERPRETACAO.md"
    if notas.exists():
        R["NOTAS"] = notas.read_text(encoding="utf-8")
    notas_s = AQUI / "NOTAS_SENSIBILIDADE.md"  # interpretação escrita após a execução
    if notas_s.exists():
        R["NOTAS_SENS"] = notas_s.read_text(encoding="utf-8")
    relatorio_sim.escrever(R, AQUI / "RESULTADOS.md")
    print(json.dumps({k: sens[k] for k in ("faixas_dg", "faixa_frac_teto_perdida_dqn_on", "faixa_frac_teto_perdida_melhor")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

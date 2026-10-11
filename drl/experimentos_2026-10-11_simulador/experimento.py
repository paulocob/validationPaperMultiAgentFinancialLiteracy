"""Uma unidade de trabalho = (h, semente de população). Roda em processo separado.

Para cada (h, s): gera as populações disjuntas (treino dos bandits, logs de
treino/validação do DQN offline, logs da OPE, validação do DQN online, teste),
treina as políticas que aprendem, avalia todas na MESMA população de teste
(números aleatórios comuns) e calcula a OPE.

Espaço de sementes: base = 100_000·índice(h) + 1_000·s; cada população soma um
deslocamento fixo. Treino, validação e teste nunca compartilham semente.
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict

import numpy as np
import torch

import ope
import politicas_sim as PS
from simulador import Ambiente, ParamsSim, Registro, amostrar_populacao, carregar_catalogo

LAMBDA_GRID = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0]
EPS_LOG = 0.2
CAMPOS = ["hake", "hake_ret", "hake10", "abandono", "passos", "passo_maestria", "diverg",
          "n_decisoes", "n_diverg", "retorno", "pre", "pos", "frustr", "tedio", "seg"]
OFFSETS = dict(bandit=1, log_tr=50, log_va=51, log_ope=52, val_on=53, teste=99)


def _sementes(h_idx: int, s: int) -> Dict[str, int]:
    base = 100_000 * h_idx + 1_000 * s
    return {k: base + v for k, v in OFFSETS.items()}


def _resumo(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(d[k], dtype=np.float32) for k in CAMPOS}


def rodar_job(h: float, h_idx: int, s: int, p_dict: Dict, cfg: Dict) -> Dict:
    torch.set_num_threads(1)
    t0 = time.time()
    p = replace(ParamsSim(), **{k: (tuple(v) if isinstance(v, list) else v) for k, v in p_dict.items()})
    cat = carregar_catalogo()
    sem = _sementes(h_idx, s)
    am = lambda n, hh, ss: amostrar_populacao(n, hh, ss, p)  # noqa: E731
    pop_teste = am(cfg["n_teste"], h, sem["teste"])
    res = {"h": h, "seed": s, "politicas": {}, "tempos": {}, "hparams": {}, "ope": {}}
    modelos = {}

    def avaliar(nome, pol, modo_kt="beta"):
        res["politicas"][nome] = _resumo(PS.simular(Ambiente(cat, p, pop_teste, modo_kt), pol))

    # P1, P2, P6
    avaliar("curriculo", PS.Curriculo())
    avaliar("aleatoria", PS.Aleatoria(sem["teste"] + 7))
    avaliar("oraculo", PS.OraculoMiope())
    avaliar("guloso_kt", PS.GulosoRastreador())

    # P3: bandits — aprendem online na população de treino (lotes de 100 em paralelo), congelados no teste
    t = time.time()
    variantes = [("linucb", "ucb", False, "beta"), ("ts", "ts", False, "beta")]
    if cfg.get("sensibilidade", True):
        variantes += [("linucb_crenca", "ucb", True, "beta"), ("linucb_kt_binario", "ucb", False, "binario")]
    for nome, tipo, alt, modo in variantes:
        b = PS.BanditLinear(tipo, sem["bandit"])
        n_lotes = cfg["n_bandit"] // 100
        for j in range(n_lotes):
            PS.simular(Ambiente(cat, p, am(100, h, sem["bandit"] * 10 + j), modo), b, aprendiz_online=b.aprender, rec_alt=alt)
        b.congelado = True
        b.nome = nome
        modelos[nome] = b
        avaliar(nome, b, modo)
    res["tempos"]["bandits"] = time.time() - t

    # P4a: DQN offline com logs do currículo + ε
    t = time.time()
    reg = Registro()
    PS.simular(Ambiente(cat, p, am(cfg["n_log_tr"], h, sem["log_tr"])), PS.CurriculoEps(EPS_LOG, sem["log_tr"]), reg)
    log_tr = reg.empilhar()
    reg = Registro()
    PS.simular(Ambiente(cat, p, am(cfg["n_log_va"], h, sem["log_va"])), PS.CurriculoEps(EPS_LOG, sem["log_va"]), reg)
    log_va = reg.empilhar()
    pib = PS.CurriculoEps(EPS_LOG, 0)
    pib_va = pib.prob_acao(log_va["acao"], log_va["curr"])
    rmodel, hp_r = PS.P1010.train_reward_model(PS.ConjuntoLog(log_tr), PS.ConjuntoLog(log_va), s)
    with torch.no_grad():
        rhat_va = rmodel(torch.from_numpy(log_va["obs"])).numpy().astype(np.float64)
    grade = {}
    for lam in cfg["lambdas"]:
        m, hp = PS.treinar_dqn_offline(log_tr, log_va, s, lam)
        pol = PS.PoliticaQ(m, f"dqn_off_{lam}")
        a_va = pol.acao_obs(log_va["obs"])
        o = ope.estimar(log_va, a_va, pib_va, rhat_va, n_boot=200, seed=s)
        grade[lam] = {"top1_val": float((a_va == log_va["acao"]).mean()), "snis_val": o["snis"],
                      "snis_val_ic": o["snis_ic"], "dr_val": o["dr"], "epocas": hp["epochs_run"], "modelo": pol}
    n_va = len(log_va["acao"])
    top = {l: g["top1_val"] for l, g in grade.items()}
    l_max = max(top, key=lambda l: (top[l], -l))
    ep = np.sqrt(top[l_max] * (1 - top[l_max]) / n_va)
    lam_1ep = min(l for l in top if top[l] >= top[l_max] - ep)  # regra de 1 EP (binomial), como no 5-fold de 10/10
    lam_snis = max(grade, key=lambda l: (grade[l]["snis_val"], -l))
    for nome, lam in (("dqn_off_1ep", lam_1ep), ("dqn_off_snis", lam_snis)):
        modelos[nome] = grade[lam]["modelo"]
        modelos[nome].nome = nome
        avaliar(nome, modelos[nome])
    res["hparams"]["dqn_off"] = {"lambda_1ep": lam_1ep, "lambda_top1max": l_max, "lambda_snis": lam_snis,
                                 "ep_binomial": float(ep), "n_val_trans": n_va, "cfg": {k: v for k, v in PS.CFG_DQN_OFF.items()},
                                 "grade": {str(l): {k: v for k, v in g.items() if k != "modelo"} for l, g in grade.items()}}
    res["n_trans_log"] = {"treino": int(len(log_tr["acao"])), "val": n_va}
    res["tempos"]["dqn_off"] = time.time() - t

    # P4b: DQN online
    t = time.time()
    q, hp_on = PS.treinar_dqn_online(cat, p, h, h_idx * 100 + s, am(cfg["n_val_on"], h, sem["val_on"]), am,
                                     dict(PS.CFG_DQN_ON, aprendizes=cfg["n_online"]))
    modelos["dqn_on"] = PS.PoliticaQ(q, "dqn_on")
    avaliar("dqn_on", modelos["dqn_on"])
    res["hparams"]["dqn_on"] = {k: v for k, v in hp_on.items()}
    res["tempos"]["dqn_on"] = time.time() - t

    # A2: OPE nos logs do currículo + ε (população própria)
    t = time.time()
    reg = Registro()
    PS.simular(Ambiente(cat, p, am(cfg["n_log_ope"], h, sem["log_ope"])), PS.CurriculoEps(EPS_LOG, sem["log_ope"]), reg)
    log_ope = reg.empilhar()
    pib_ope = pib.prob_acao(log_ope["acao"], log_ope["curr"])
    with torch.no_grad():
        rhat = rmodel(torch.from_numpy(log_ope["obs"])).numpy().astype(np.float64)
    acoes_ope = {"curriculo": log_ope["curr"]}
    for nome in ("linucb", "ts", "dqn_off_1ep", "dqn_off_snis", "dqn_on"):
        acoes_ope[nome] = modelos[nome].acao_obs(log_ope["obs"], log_ope["curr"])
    for nome, a_pi in acoes_ope.items():
        res["ope"][nome] = ope.estimar(log_ope, a_pi, pib_ope, rhat, seed=s)
    res["ope_logs"] = {"n_trans": int(len(log_ope["acao"])), "rec_media_log": float(log_ope["rec"].mean())}
    res["tempos"]["ope"] = time.time() - t
    res["tempos"]["total"] = time.time() - t0
    return res

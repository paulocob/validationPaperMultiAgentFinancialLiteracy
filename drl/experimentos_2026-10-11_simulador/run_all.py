"""Comparação de políticas de sequenciamento no simulador cognitivo (7ª rodada).

Uso (um comando reproduz tudo, CPU):
    python run_all.py            # completo
    python run_all.py --quick    # fumaça: populações e grades reduzidas
    python run_all.py --sem-llm  # pula o planejador LLM

Etapas: (1) calibração; (2) teste de sanidade (oráculo, currículo, aleatória);
(3) trabalhos (h, semente) em paralelo; (4) planejador LLM; (5) agregação, IC,
tamanhos de efeito, poder (A3) e validação da OPE (A2); (6) resultados.json,
figuras e RESULTADOS.md.
"""
from __future__ import annotations

import os

# Uma thread de BLAS/OpenMP por processo: com 15 processos paralelos, o padrão
# (uma thread por núcleo em cada processo) esgotou a memória do OpenBLAS. Precisa
# vir antes do import do numpy; os processos filhos herdam o ambiente.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import calibracao  # noqa: E402
import experimento  # noqa: E402
import politicas_sim as PS  # noqa: E402
from simulador import Ambiente, ParamsSim, amostrar_populacao, carregar_catalogo  # noqa: E402

H_GRID = [0.0, 0.5, 1.0]
N_BOOT = 2000
Z_A, Z_B = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
MDE_PLANO, N_PLANO = 0.15, 90
POL_PRINCIPAIS = ["curriculo", "aleatoria", "linucb", "ts", "dqn_off_1ep", "dqn_off_snis", "dqn_on",
                  "llm_zero", "llm_few", "guloso_kt", "oraculo"]
POL_SENSIB = ["linucb_crenca", "linucb_kt_binario"]
ROTULO = {
    "curriculo": "Currículo fixo (P1)", "aleatoria": "Aleatória c/ pré-req. (P2)", "linucb": "LinUCB (P3)",
    "ts": "Thompson linear (P3)", "dqn_off_1ep": "DQN offline, λ por Top-1/1EP (P4a)",
    "dqn_off_snis": "DQN offline, λ por SNIS (P4a)", "dqn_on": "DQN online (P4b)",
    "llm_zero": "Planejador LLM zero-shot (P5)", "llm_few": "Planejador LLM few-shot (P5)",
    "oraculo": "Oráculo míope (P6, teto)",
    "guloso_kt": "Guloso pelo rastreador (referência implementável)",
    "linucb_crenca": "LinUCB, recompensa c/ revisão de crença (sens.)",
    "linucb_kt_binario": "LinUCB, rastreador BKT binário (sens.)",
}

CFG_FULL = dict(n_teste=1000, seeds=[0, 1, 2, 3, 4], n_bandit=2000, n_log_tr=2000, n_log_va=500, n_log_ope=1000,
                n_val_on=500, n_online=24000, lambdas=experimento.LAMBDA_GRID, n_calib=20000, rodadas_calib=3,
                n_sanidade=5000, llm_aprendizes=300, llm_seeds=[0, 1, 2], llm_orcamento_s=5400)
CFG_QUICK = dict(n_teste=300, seeds=[0, 1], n_bandit=500, n_log_tr=500, n_log_va=200, n_log_ope=300,
                 n_val_on=200, n_online=1024, lambdas=[0.0, 0.1, 1.0], n_calib=4000, rodadas_calib=2,
                 n_sanidade=2000, llm_aprendizes=20, llm_seeds=[0], llm_orcamento_s=300)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------------
# (2) Teste de sanidade
# --------------------------------------------------------------------------
def sanidade(cat, p, n) -> Dict:
    """Antes de treinar qualquer coisa: o oráculo míope deve igualar ou superar o
    currículo em todo h, coincidir com ele em h = 0 (currículo quase ótimo em
    população homogênea), e a política aleatória deve ficar abaixo de ambos."""
    out, ok = {}, True
    for h in H_GRID:
        r = {}
        for pol in (PS.Curriculo(), PS.Aleatoria(5), PS.OraculoMiope()):
            d = PS.simular(Ambiente(cat, p, amostrar_populacao(n, h, 777, p)), pol)
            r[pol.nome] = {"hake": float(d["hake"].mean()), "abandono": float(d["abandono"].mean()),
                           "diverg": float(d["n_diverg"].sum() / d["n_decisoes"].sum())}
        out[str(h)] = r
        c, a, o = r["curriculo"]["hake"], r["aleatoria"]["hake"], r["oraculo"]["hake"]
        ok &= (o >= c - 0.005) and (a < c) and (a < o)
        if h == 0.0:
            ok &= abs(o - c) < 0.01
        log(f"[sanidade] h={h}: currículo {c:.3f}, aleatória {a:.3f}, oráculo {o:.3f}")
    out["aprovado"] = bool(ok)
    return out


# --------------------------------------------------------------------------
# (4) Planejador LLM
# --------------------------------------------------------------------------
def rodar_llm(cat, p, cfg) -> Dict:
    import planejador_llm as PL
    out = {"executado": False}
    t0 = time.time()
    pont, nome, erro = PL.carregar_pontuador(log=log)
    out["tempo_carga_s"] = time.time() - t0
    am = lambda n, hh, ss: amostrar_populacao(n, hh, ss, p)  # noqa: E731
    ex = PL.exemplos_few_shot(cat, p, am)
    exemplo_estado = PL.descrever_estado(Ambiente(cat, p, am(1, 0.5, 4243)), 0)
    out["prompt"] = {"sistema": PL.SISTEMA, "exemplo_estado": exemplo_estado,
                     "few_shot": [{"estado": t, "resposta": l} for t, l in ex]}
    if pont is None:
        out["motivo"] = "modelo indisponível localmente: " + erro
        return out
    out["modelo"] = nome
    # Estimativa de custo com 16 prompts e redução da população se necessário (registrada).
    amb = Ambiente(cat, p, am(16, 0.5, 4244))
    t = time.perf_counter()
    pont([PL.montar_mensagens(PL.descrever_estado(amb, i), ex) for i in range(16)])
    s_por_prompt = (time.perf_counter() - t) / 16
    n_apr = cfg["llm_aprendizes"]
    decisoes_previstas = 2 * len(H_GRID) * len(cfg["llm_seeds"]) * n_apr * 14
    if decisoes_previstas * s_por_prompt > cfg["llm_orcamento_s"]:
        n_apr = max(20, int(cfg["llm_orcamento_s"] / (s_por_prompt * 2 * len(H_GRID) * len(cfg["llm_seeds"]) * 14)))
        out["reducao"] = f"população reduzida de {cfg['llm_aprendizes']} para {n_apr} aprendizes por semente " \
                         f"(custo estimado {s_por_prompt:.2f} s/prompt; orçamento {cfg['llm_orcamento_s']} s)"
        log("[llm] " + out["reducao"])
    out.update(s_por_prompt_estimado=s_por_prompt, n_aprendizes=n_apr, seeds=cfg["llm_seeds"], resultados={})
    for variante, exs in (("llm_zero", []), ("llm_few", ex)):
        for hi, h in enumerate(H_GRID):
            for s in cfg["llm_seeds"]:
                sem = experimento._sementes(hi, s)["teste"]
                # Mesmos aprendizes (primeiros n_apr da população de teste) para pareamento com as demais políticas.
                pop = amostrar_populacao(cfg["n_teste"], h, sem, p)
                idx = np.arange(n_apr)
                sub = replace(pop, seg=pop.seg[idx], L0=pop.L0[idx], T=pop.T[idx], S=pop.S[idx], G=pop.G[idx],
                              F=pop.F[idx], U_resp=pop.U_resp[idx], U_aband=pop.U_aband[idx], U_sent=pop.U_sent[idx],
                              U_pre_obs=pop.U_pre_obs[idx], U_teste=pop.U_teste[idx])
                pol = PL.PlanejadorLLM(pont, exs, variante)
                t = time.time()
                d = PS.simular(Ambiente(cat, p, sub), pol)
                out["resultados"][f"{variante}|{h}|{s}"] = {
                    "res": experimento._resumo(d), "tempo_s": time.time() - t, "decisoes": pol.n_decisoes,
                    "chamadas_modelo": pol.n_chamadas, "seg_modelo": pol.seg_modelo}
                log(f"[llm] {variante} h={h} s={s}: Hake {d['hake'].mean():.3f}, {pol.n_chamadas} chamadas, "
                    f"{time.time() - t:.0f} s")
    out["executado"] = True
    out["tempo_total_s"] = time.time() - t0
    return out


# --------------------------------------------------------------------------
# (5) Agregação
# --------------------------------------------------------------------------
def _boot_media(x: np.ndarray, rng, n_boot=N_BOOT) -> List[float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float("nan")] * 2
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _boot_razao(num, den, rng, n_boot=N_BOOT):
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def n_por_braco(delta: float, sd: float) -> float:
    if not np.isfinite(delta) or abs(delta) < 1e-9:
        return float("inf")
    return float(math.ceil(2 * (Z_A + Z_B) ** 2 * sd ** 2 / delta ** 2))


def agregar(jobs: List[Dict], llm: Dict) -> Dict:
    rng = np.random.default_rng(12345)
    por = {}  # (pol, h) -> lista de dicionários por semente
    for j in jobs:
        for nome, d in j["politicas"].items():
            por.setdefault((nome, j["h"]), []).append((j["seed"], d))
    if llm.get("executado"):
        for chave, v in llm["resultados"].items():
            nome, h, s = chave.split("|")
            por.setdefault((nome, float(h)), []).append((int(s), v["res"]))
    tab = {}
    for (nome, h), lst in sorted(por.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lst = sorted(lst, key=lambda t: t[0])
        cat_ = lambda k: np.concatenate([d[k] for _, d in lst]).astype(np.float64)  # noqa: E731
        hake, ret, h10 = cat_("hake"), cat_("hake_ret"), cat_("hake10")
        ab, nd, ndv = cat_("abandono"), cat_("n_decisoes"), cat_("n_diverg")
        pm = cat_("passo_maestria")
        e = {
            "n_aprendizes": int(len(hake)), "seeds": [s for s, _ in lst],
            "hake": float(hake.mean()), "hake_ic": _boot_media(hake, rng), "hake_dp": float(hake.std(ddof=1)),
            "hake_dp_entre_sementes": float(np.std([d["hake"].mean() for _, d in lst], ddof=1)) if len(lst) > 1 else 0.0,
            "hake_ret": float(ret.mean()), "hake_ret_ic": _boot_media(ret, rng),
            "hake10": float(np.nanmean(h10)), "hake10_dp": float(np.nanstd(h10, ddof=1)),
            "abandono": float(ab.mean()), "abandono_ic": _boot_media(ab, rng),
            "diverg": float(ndv.sum() / nd.sum()), "diverg_ic": _boot_razao(ndv, nd, rng),
            "frac_maestria": float((pm > 0).mean()),
            "passos_ate_maestria": float(pm[pm > 0].mean()) if (pm > 0).any() else float("nan"),
            "passos_medios": float(cat_("passos").mean()),
            "retorno_treino": float(cat_("retorno").mean()),
            "rec_por_passo": float(cat_("retorno").sum() / nd.sum()),
            "frustr_por_aprendiz": float(cat_("frustr").mean()), "tedio_por_aprendiz": float(cat_("tedio").mean()),
        }
        # Efeito vs currículo, pareado nos MESMOS aprendizes (mesmas sementes e índices).
        base = dict(por[("curriculo", h)])
        if nome != "curriculo":
            ch = np.concatenate([base[s]["hake"][: len(d["hake"])] for s, d in lst]).astype(np.float64)
            ch10 = np.concatenate([base[s]["hake10"][: len(d["hake"])] for s, d in lst]).astype(np.float64)
            dif = hake - ch
            sd_pool = math.sqrt((hake.var(ddof=1) + ch.var(ddof=1)) / 2)
            dif10 = np.nanmean(h10) - np.nanmean(ch10)
            sd10 = math.sqrt((np.nanvar(h10, ddof=1) + np.nanvar(ch10, ddof=1)) / 2)
            e["vs_curriculo"] = {
                "dif_hake": float(dif.mean()), "dif_hake_ic": _boot_media(dif, rng),
                "d_cohen": float(dif.mean() / sd_pool) if sd_pool > 0 else float("nan"),
                "sd_agrupado": sd_pool,
                "n_por_braco": n_por_braco(float(dif.mean()), sd_pool),
                "dif_hake10": float(dif10), "sd_agrupado10": sd10, "d_cohen10": float(dif10 / sd10) if sd10 > 0 else float("nan"),
                "n_por_braco10": n_por_braco(float(dif10), sd10),
                "dif_ret": float((ret - np.concatenate([base[s]["hake_ret"][: len(d["hake"])] for s, d in lst])).mean()),
                "dif_abandono": float((ab - np.concatenate([base[s]["abandono"][: len(d["hake"])] for s, d in lst])).mean()),
            }
        tab[f"{nome}|{h}"] = e
    # Validação da OPE (A2)
    ope_out = {}
    for h in H_GRID:
        js = [j for j in jobs if j["h"] == h]
        linhas = {}
        for nome in js[0]["ope"]:
            snis = np.mean([j["ope"][nome]["snis"] for j in js])
            dr = np.mean([j["ope"][nome]["dr"] for j in js])
            verd = tab[f"{nome}|{h}"]["rec_por_passo"]
            linhas[nome] = {
                "snis": float(snis), "dr": float(dr), "verdadeiro_rec_passo": verd, "hake": tab[f"{nome}|{h}"]["hake"],
                "snis_ic_medio": [float(np.mean([j["ope"][nome]["snis_ic"][k] for j in js])) for k in (0, 1)],
                "dr_ic_medio": [float(np.mean([j["ope"][nome]["dr_ic"][k] for j in js])) for k in (0, 1)],
                "ess_medio": float(np.mean([j["ope"][nome]["ess"] for j in js])),
                "concord_log": float(np.mean([j["ope"][nome]["frac_concord"] for j in js])),
                "erro_abs_snis": float(abs(snis - verd)), "erro_abs_dr": float(abs(dr - verd)),
            }
        nomes = list(linhas)
        v = np.array([linhas[n]["verdadeiro_rec_passo"] for n in nomes])
        hk = np.array([linhas[n]["hake"] for n in nomes])
        conc = {}
        for est in ("snis", "dr"):
            x = np.array([linhas[n][est] for n in nomes])
            conc[est] = {"kendall_vs_rec_passo": float(stats.kendalltau(x, v)[0]),
                         "kendall_vs_hake": float(stats.kendalltau(x, hk)[0]),
                         "top1_coincide_hake": bool(nomes[int(np.argmax(x))] == nomes[int(np.argmax(hk))]),
                         "erro_abs_medio": float(np.mean(np.abs(x - v)))}
        conc["rec_passo_vs_hake"] = float(stats.kendalltau(v, hk)[0])
        ope_out[str(h)] = {"politicas": linhas, "concordancia": conc}
    return {"tabela": tab, "ope": ope_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sem-llm", action="store_true")
    ap.add_argument("--workers", type=int, default=15)
    args = ap.parse_args()
    cfg = dict(CFG_QUICK if args.quick else CFG_FULL)
    t0 = time.time()
    tempos = {}
    cat = carregar_catalogo()

    log("(1) calibração")
    t = time.time()
    p = calibracao.calibrar(n=cfg["n_calib"], rodadas=cfg["rodadas_calib"], log=log)
    cal_cur = {str(h): {} for h in H_GRID}
    for h in H_GRID:
        d = calibracao._curriculo(cat, p, h, cfg["n_calib"])
        cal_cur[str(h)] = {"hake": float(d["hake"].mean()), "abandono": float(d["abandono"].mean())}
    tempos["calibracao"] = time.time() - t

    log("(2) teste de sanidade")
    san = sanidade(cat, p, cfg["n_sanidade"])
    if not san["aprovado"]:
        log("[sanidade] REPROVADO — a execução para aqui; ver resultados.json")
    R = {"quick": args.quick, "cfg": {k: v for k, v in cfg.items()}, "params": p.as_dict(),
         "calibracao": {"alvo_hake_h05": calibracao.ALVO_HAKE, "alvo_abandono_h1": calibracao.ALVO_ABANDONO,
                        "c_T": p.c_T, "alfa_abandono": p.alfa_abandono, "curriculo_por_h": cal_cur,
                        "semente": calibracao.SEMENTE_CALIB, "n": cfg["n_calib"]},
         "sanidade": san,
         "ambiente": {"python": platform.python_version(), "numpy": np.__version__,
                      "torch": __import__("torch").__version__, "cpu": platform.processor()}}
    if not san["aprovado"]:
        (AQUI / ("resultados_quick.json" if args.quick else "resultados.json")).write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
        sys.exit(1)

    log("(3) trabalhos (h, semente) em paralelo")
    t = time.time()
    jobs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(experimento.rodar_job, h, hi, s, p.as_dict(), cfg): (h, s)
                for hi, h in enumerate(H_GRID) for s in cfg["seeds"]}
        for f in as_completed(futs):
            j = f.result()
            jobs.append(j)
            log(f"  h={j['h']} s={j['seed']} concluído em {j['tempos']['total']:.0f} s "
                f"(λ_1EP={j['hparams']['dqn_off']['lambda_1ep']}, λ_SNIS={j['hparams']['dqn_off']['lambda_snis']})")
    tempos["trabalhos"] = time.time() - t

    llm = {"executado": False, "motivo": "--sem-llm"}
    if not args.sem_llm:
        log("(4) planejador LLM")
        t = time.time()
        llm = rodar_llm(cat, p, cfg)
        tempos["llm"] = time.time() - t

    log("(5) agregação")
    A = agregar(jobs, llm)
    R.update(A)
    R["jobs"] = [{k: v for k, v in j.items() if k != "politicas"} for j in sorted(jobs, key=lambda j: (j["h"], j["seed"]))]
    R["llm"] = {k: v for k, v in llm.items() if k != "resultados"}
    if llm.get("executado"):
        R["llm"]["por_execucao"] = {k: {kk: vv for kk, vv in v.items() if kk != "res"} for k, v in llm["resultados"].items()}
    tempos["total"] = time.time() - t0
    R["tempos_s"] = tempos
    R["rotulos"] = ROTULO
    R["mde_plano"], R["n_plano"] = MDE_PLANO, N_PLANO
    saida = AQUI / ("resultados_quick.json" if args.quick else "resultados.json")
    saida.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log("(6) figuras e relatório")
    notas = AQUI / "NOTAS_INTERPRETACAO.md"  # texto interpretativo escrito após a execução completa
    if notas.exists() and not args.quick:
        R["NOTAS"] = notas.read_text(encoding="utf-8")
    import figuras_sim
    import relatorio_sim as relatorio
    figuras_sim.gerar(R, sufixo="_quick" if args.quick else "")
    relatorio.escrever(R, AQUI / ("RESULTADOS_quick.md" if args.quick else "RESULTADOS.md"))
    log(f"fim: {tempos['total'] / 60:.1f} min")


if __name__ == "__main__":
    main()

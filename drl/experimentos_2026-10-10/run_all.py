"""Reexecução dos experimentos de DRL (Parte E da 5ª rodada).

Uso (reproduz tudo com um comando, CPU):
    python run_all.py            # completo (~30–60 min em CPU)
    python run_all.py --quick    # fumaça: 1 semente, 2 folds, grade reduzida

Saídas em experimentos_2026-10-10/: resultados.json, RESULTADOS.md e as
figuras em ./figuras/ ou em $FIG_DIR (com backup
das antigas em figures/_antigas_2026-10-10/).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy import stats

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import dados  # noqa: E402
import politicas as P  # noqa: E402
from avaliacao import (  # noqa: E402
    behavior_policy, by_group, concat_pt, evaluate_policy, pooled_ci, ranking_from_scores,
)
from dados import Base, Conjunto, build_conjunto, fold_partition, items_by_ids, items_by_users, profile_folds  # noqa: E402
from politicas import scores  # noqa: E402

LAMBDA_GRID = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]  # {2, 5, 10}: extensão pedida
SEEDS = [42, 43, 44, 45, 46]
CV_SEED = 42
CULTURAL_WEIGHT = 0.15
POLICIES = ["seq", "seq_rep", "bc", "dqn0", "dqn_lstar", "dqn_l1se", "cql", "ppo"]
POLICY_LABEL = {
    "seq": "Regra sequencial (próxima do catálogo)",
    "seq_rep": "Regra sequencial + repetir terminal",
    "bc": "BC", "dqn0": "DQN λ = 0", "dqn_lstar": "DQN λ*", "dqn_l1se": "DQN λ_1EP", "cql": "CQL", "ppo": "PPO offline",
}
METRICS = ["top1", "top3", "reward_assoc", "teto", "pct_teto", "snis", "dr", "dr_q_cql", "ess"]

torch.set_num_threads(4)
# Console do Windows em cp1252: força UTF-8 para os rótulos com λ.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def conjuntos(base: Base, part: Dict[str, List[str]], with_country=False, cw=0.0) -> Dict[str, Conjunto]:
    return {k: build_conjunto(items_by_users(base.interactions, v), base.vocab, base.nxt, with_country, cw)
            for k, v in part.items()}


def msd(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return float(np.nanmean(xs)), float(np.nanstd(xs, ddof=1)) if len(xs) > 1 else 0.0


def seq_ranks(conj: Conjunto, variant: str) -> np.ndarray:
    if variant == "seq":
        return conj.seq_rank
    # Variante com repetição da tarefa terminal: recalcula o ranking.
    return np.array([dados.sequential_rank(t, BASE.nxt, BASE.vocab, 3, True) for t in conj.current_tasks])


def train_all_on(c: Dict[str, Conjunto], seed: int, lambdas: List[float], log) -> Dict:
    """Treina todas as políticas numa partição e devolve modelos + hparams."""
    t0 = time.time()
    out = {"models": {}, "hparams": {}}
    out["models"]["bc"], out["hparams"]["bc"] = P.train_bc(c["train"], c["val"], seed)
    out["models"]["rmodel"], out["hparams"]["rmodel"] = P.train_reward_model(c["train"], c["val"], seed)
    out["models"]["cql"], out["hparams"]["cql"] = P.train_cql(c["train"], c["val"], seed)
    out["models"]["ppo"], out["hparams"]["ppo"] = P.train_ppo(c["train"], c["val"], seed)
    for lam in lambdas:
        out["models"][f"dqn_{lam}"], out["hparams"][f"dqn_{lam}"] = P.train_dqn(c["train"], c["val"], seed, lam)
    log(f"    treinos concluídos em {time.time() - t0:.0f} s")
    return out


def eval_all(c: Dict[str, Conjunto], tr: Dict, lambdas: List[float], lam_star: float, lam_1se: float) -> Dict:
    """Avalia no teste (com OPE) e na validação (sem OPE) todas as políticas."""
    test, val = c["test"], c["val"]
    pb = behavior_policy(test, tr["models"]["bc"], test.actions)
    rhat = scores(tr["models"]["rmodel"], test.states)
    qcql = scores(tr["models"]["cql"], test.states)
    ranks = {
        "seq": seq_ranks(test, "seq"), "seq_rep": seq_ranks(test, "seq_rep"),
        "bc": ranking_from_scores(scores(tr["models"]["bc"], test.states)),
        "cql": ranking_from_scores(qcql),
        "ppo": ranking_from_scores(scores(tr["models"]["ppo"], test.states)),
    }
    for lam in lambdas:
        ranks[f"dqn_{lam}"] = ranking_from_scores(scores(tr["models"][f"dqn_{lam}"], test.states))
    ranks["dqn0"] = ranks["dqn_0.0"]
    ranks["dqn_lstar"] = ranks[f"dqn_{lam_star}"]
    ranks["dqn_l1se"] = ranks[f"dqn_{lam_1se}"]
    res = {"test": {}, "val": {}, "pi_b_from_counts": float(pb["from_counts"].mean()), "ranks": ranks}
    for name, rk in ranks.items():
        res["test"][name] = evaluate_policy(rk, test, pb["pi_b"], rhat, qcql, bootstrap=True)
    # Validação: Top-1/Top-3/recompensa associada (para a grade de λ e a figura).
    ones = np.ones(len(val))
    for lam in lambdas:
        rk = ranking_from_scores(scores(tr["models"][f"dqn_{lam}"], val.states))
        res["val"][f"dqn_{lam}"] = evaluate_policy(rk, val, ones, bootstrap=False)["point"]
    return res


def leakage(train: Conjunto, test: Conjunto) -> Dict[str, float]:
    tr_keys = set(train.state_keys())
    tr_pairs = set(zip(train.state_keys(), train.actions.tolist()))
    te_keys = test.state_keys()
    te_pairs = list(zip(te_keys, test.actions.tolist()))
    return {
        "estados_teste_no_treino": float(np.mean([k in tr_keys for k in te_keys])),
        "pares_sa_teste_no_treino": float(np.mean([p in tr_pairs for p in te_pairs])),
        "n_teste": len(test), "n_treino": len(train),
    }


def encoder_check(base: Base) -> Dict:
    """E1: distribuição por grupo de atributos de s e s' (codificador corrigido)
    e de s' pelo codificador ORIGINAL (que perdia profile e missions)."""
    from train_utils import encode_state as encode_old
    items = list(base.interactions.values())
    T = len(base.vocab.task_to_idx)
    s = torch.stack([dados.encode_state_corrigido(it["state"], it["state"]["profile"], base.vocab, T) for it in items])
    s2 = torch.stack([dados.encode_state_corrigido(it["next_state"], it["state"]["profile"], base.vocab, T) for it in items])
    s2_old = torch.stack([encode_old(it["next_state"], base.vocab, T) for it in items])
    out = {}
    i = 0
    for name, size in dados.FEATURE_GROUPS:
        sl = slice(i, i + size)
        out[name] = {"s": s[:, sl].mean().item(), "s_linha_corrigido": s2[:, sl].mean().item(),
                     "s_linha_original": s2_old[:, sl].mean().item()}
        i += size
    out["_nota"] = ("médias por grupo de atributos sobre as 2.000 transições; no codificador original "
                    "s' tinha literacia 0, segmento sempre 'iniciante' e flags de missão zeradas")
    return out


# --------------------------------------------------------------------------
# Experimento cultural (E6)
# --------------------------------------------------------------------------
@torch.no_grad()
def divergencia_pais(model, conj: Conjunto) -> float:
    """Fração de estados de teste em que o argmax muda ao trocar só o país."""
    model.eval()
    base_idx = conj.states.shape[1] - len(dados.PAISES)
    muda = 0
    for x in conj.states:
        acts = set()
        for p, idx in dados.PAIS_TO_IDX.items():
            y = x.clone()
            y[base_idx:] = 0.0
            y[base_idx + idx] = 1.0
            acts.add(int(model(y.unsqueeze(0)).argmax(1)))
        muda += len(acts) > 1
    return muda / len(conj)


def experimento_cultural(part: Dict[str, List[str]], lam_star: float, seeds: List[int], log) -> Dict:
    base_c = dados.load_base(dados.DATA_DIR_CULTURAL)
    out = {"configs": {}, "paises_teste": None}
    for cw in (0.0, CULTURAL_WEIGHT):
        c = conjuntos(base_c, part, with_country=True, cw=cw)
        out["paises_teste"] = dict(Counter(c["test"].countries))
        rows = []
        for seed in seeds:
            m, hp = P.train_dqn(c["train"], c["val"], seed, lam_star)
            rk = ranking_from_scores(scores(m, c["test"].states))
            top1 = float((rk[:, 0] == c["test"].actions).mean())
            rows.append({"seed": seed, "top1": top1, "divergencia": divergencia_pais(m, c["test"]),
                         "epochs_run": hp["epochs_run"]})
            log(f"    cultural cw={cw} seed={seed}: top1={top1:.3f} div={rows[-1]['divergencia']:.3f}")
        out["configs"][str(cw)] = {"rows": rows, "top1": msd([r["top1"] for r in rows]),
                                   "divergencia": msd([r["divergencia"] for r in rows]),
                                   "hparams": {**hp, "cultural_weight": cw, "pais_no_estado": True}}
    d0 = [r["divergencia"] for r in out["configs"]["0.0"]["rows"]]
    d1 = [r["divergencia"] for r in out["configs"][str(CULTURAL_WEIGHT)]["rows"]]
    welch = stats.ttest_ind(d1, d0, equal_var=False)
    pareado = stats.ttest_rel(d1, d0) if len(d0) > 1 else None
    out["teste_t"] = {"welch_t": float(welch.statistic), "welch_p": float(welch.pvalue),
                      "pareado_t": float(pareado.statistic) if pareado else None,
                      "pareado_p": float(pareado.pvalue) if pareado else None, "n_por_grupo": len(d0)}
    out["nota_f_cult"] = ("f_cult só discrimina ações no par (AO, M6): BR vale 1,0 em todas as missões "
                          "e PT vale 0,5 em todas; portanto a mistura 0,85·R + 0,15·f_cult altera a "
                          "ordenação de ações apenas para aprendizes de Angola, e só entre M6 e as demais.")
    return out


# --------------------------------------------------------------------------
# Principal
# --------------------------------------------------------------------------
BASE: Base = None


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(AQUI))
    args = ap.parse_args()
    out_dir = Path(args.out)
    seeds = SEEDS[:1] if args.quick else SEEDS
    lambdas = [0.0, 0.05, 1.0, 10.0] if args.quick else LAMBDA_GRID
    n_folds_run = 2 if args.quick else 5

    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    T0 = time.time()
    tempos = {}
    BASE = dados.load_base()
    folds = profile_folds(BASE.users, 5, 42)
    fixed = fold_partition(folds, 0)
    R = {"meta": {
        "data": "2026-10-10", "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "device": "cpu", "sementes_split_fixo": seeds, "semente_cv": CV_SEED,
        "grade_lambda": lambdas, "quick": args.quick, "pi_b_min": 0.01, "n_bootstrap": 1000,
        "folds_perfis": folds, "split_fixo": fixed,
        "segmento_por_perfil": {u: d["profile"]["segment"] for u, d in BASE.users.items()},
    }}

    # ---- E1 / E5 / E7: codificador, r_miss, vazamento --------------------
    t = time.time()
    R["codificador"] = encoder_check(BASE)
    todos = build_conjunto(list(BASE.interactions.values()), BASE.vocab, BASE.nxt)
    R["recompensa"] = {
        "fracao_r_miss_ativo": float(todos.r_miss.mean()),
        "media_refinada": float(todos.rewards.mean()),
        "fracao_done": float(todos.dataset.dones.mean().item()),
        "n": len(todos),
    }
    orig = {k: build_conjunto(items_by_ids(BASE.interactions, v), BASE.vocab, BASE.nxt) for k, v in BASE.original_splits.items()}
    cfix = conjuntos(BASE, fixed)
    R["vazamento"] = {"antes_particao_original": leakage(orig["train"], orig["test"]),
                      "depois_por_perfil_split_fixo": leakage(cfix["train"], cfix["test"])}
    R["tamanhos"] = {"original": {k: len(v) for k, v in orig.items()},
                     "split_fixo": {k: len(v) for k, v in cfix.items()}}
    # Regra sequencial no teste ORIGINAL (reprodução do fato verificado: 0,592).
    R["regra_seq_teste_original"] = {
        "top1": float((orig["test"].seq_pred == orig["test"].actions).mean()),
        "top1_com_repeticao": float((seq_ranks(orig["test"], "seq_rep")[:, 0] == orig["test"].actions).mean()),
    }
    tempos["preparacao_s"] = time.time() - t
    log(f"[prep] r_miss ativo = {R['recompensa']['fracao_r_miss_ativo']:.3f}; vazamento antes/depois = "
        f"{R['vazamento']['antes_particao_original']['estados_teste_no_treino']:.3f} / "
        f"{R['vazamento']['depois_por_perfil_split_fixo']['estados_teste_no_treino']:.3f}")

    # ---- Split fixo, 5 sementes, grade de λ ------------------------------
    t = time.time()
    fixo_runs = {}
    for seed in seeds:
        log(f"[split fixo] semente {seed}")
        tr = train_all_on(cfix, seed, lambdas, log)
        fixo_runs[seed] = {"train": tr, "eval": None}
    # λ*: maior Top-1 médio de validação; desempate pelo menor λ.
    val_top1 = {lam: msd([fixo_runs[s]["train"]["hparams"][f"dqn_{lam}"]["best_val_top1"] for s in seeds]) for lam in lambdas}
    best = max(val_top1[l][0] for l in lambdas)
    lam_star = min(l for l in lambdas if abs(val_top1[l][0] - best) < 1e-12)
    # Regra de um erro-padrão: menor λ cujo Top-1 médio de validação fica a até
    # 1 EP (dp entre sementes / sqrt(n) do λ de máximo) do máximo.
    se_1se = val_top1[lam_star][1] / np.sqrt(len(seeds))
    lam_1se = min(l for l in lambdas if val_top1[l][0] >= best - se_1se - 1e-12)
    log(f"[split fixo] λ* = {lam_star} (Top-1 val {best:.3f}); λ_1EP = {lam_1se} (EP = {se_1se:.4f})")
    for seed in seeds:
        fixo_runs[seed]["eval"] = eval_all(cfix, fixo_runs[seed]["train"], lambdas, lam_star, lam_1se)
    tempos["split_fixo_s"] = time.time() - t

    # Alternativa com EP binomial sqrt(p(1-p)/n_val), o mesmo critério usado no 5-fold.
    se_bin = float(np.sqrt(best * (1 - best) / len(cfix["val"])))
    lam_1se_bin = min(l for l in lambdas if val_top1[l][0] >= best - se_bin - 1e-12)
    fixo = {"lambda_star": lam_star, "lambda_1se": lam_1se, "ep_1se": float(se_1se),
            "lambda_1se_binomial": lam_1se_bin, "ep_binomial": se_bin, "politicas": {}, "grade_lambda": {}, "hparams": {}}
    for name in POLICIES:
        pts = [fixo_runs[s]["eval"]["test"][name]["point"] for s in seeds]
        fixo["politicas"][name] = {
            "n_execucoes": len(seeds), "n_teste": len(cfix["test"]),
            **{m: msd([p[m] for p in pts]) for m in METRICS},
            "ci95_bootstrap": pooled_ci([fixo_runs[s]["eval"]["test"][name]["per_transition"] for s in seeds]),
            "por_semente": {str(s): fixo_runs[s]["eval"]["test"][name]["point"] for s in seeds},
        }
    for lam in lambdas:
        key = f"dqn_{lam}"
        tp = [fixo_runs[s]["eval"]["test"][key]["point"] for s in seeds]
        vp = [fixo_runs[s]["eval"]["val"][key] for s in seeds]
        fixo["grade_lambda"][str(lam)] = {
            "val_top1": val_top1[lam], "val_top3": msd([p["top3"] for p in vp]),
            "val_reward_assoc": msd([p["reward_assoc"] for p in vp]),
            "test_top1": msd([p["top1"] for p in tp]), "test_top3": msd([p["top3"] for p in tp]),
            "test_reward_assoc": msd([p["reward_assoc"] for p in tp]), "test_snis": msd([p["snis"] for p in tp]),
            "test_dr": msd([p["dr"] for p in tp]),
            "epochs_run": msd([fixo_runs[s]["train"]["hparams"][key]["epochs_run"] for s in seeds]),
        }
    fixo["hparams"] = {k: v for k, v in fixo_runs[seeds[0]]["train"]["hparams"].items()}
    fixo["pi_b_from_counts"] = msd([fixo_runs[s]["eval"]["pi_b_from_counts"] for s in seeds])
    # Concordância DQN λ* × regra sequencial (teste do split fixo).
    for pol in ("dqn_lstar", "dqn_l1se"):
        conc = []
        for s in seeds:
            rk = fixo_runs[s]["eval"]["ranks"][pol][:, 0]
            sp = cfix["test"].seq_pred
            conc.append({"todas": float((rk == sp).mean()), "regra_definida": float((rk[sp >= 0] == sp[sp >= 0]).mean())})
        key = "concordancia_dqn_seq" if pol == "dqn_lstar" else "concordancia_dqn_l1se_seq"
        fixo[key] = {k: msd([c[k] for c in conc]) for k in conc[0]}
        fixo[key]["fracao_regra_definida"] = float((cfix["test"].seq_pred >= 0).mean())
    # Diagnóstico por segmento/missão no split fixo (5 sementes).
    for gname, groups in (("segmento", cfix["test"].segments), ("missao", cfix["test"].action_missions)):
        per = [by_group(fixo_runs[s]["eval"]["ranks"]["dqn_lstar"], cfix["test"], groups) for s in seeds]
        fixo[f"top1_por_{gname}"] = {g: {"top1": msd([p[g]["top1"] for p in per]), "n": per[0][g]["n"]} for g in per[0]}
    R["split_fixo"] = fixo

    # ---- Validação cruzada 5-fold por perfil (semente 42) ----------------
    t = time.time()
    cv = {"folds": {}, "politicas": {}, "lambda_star_por_fold": {}, "lambda_1se_por_fold": {}, "lambda_star_fixo": lam_star, "lambda_1se_fixo": lam_1se}
    oof = {name: [] for name in POLICIES + ["dqn_lstar_fixo"]}
    oof_ranks = {name: [] for name in ("dqn_lstar", "dqn_l1se", "seq")}
    oof_conj = []
    for k in range(n_folds_run):
        part = fold_partition(folds, k)
        c = conjuntos(BASE, part)
        log(f"[cv] fold {k}: teste={part['test']} val={part['val']}")
        tr = train_all_on(c, CV_SEED, lambdas, log)
        vt = {lam: tr["hparams"][f"dqn_{lam}"]["best_val_top1"] for lam in lambdas}
        bestk = max(vt.values())
        lam_k = min(l for l in lambdas if abs(vt[l] - bestk) < 1e-12)
        cv["lambda_star_por_fold"][str(k)] = lam_k
        # 1 EP com uma única execução por fold: erro-padrão binomial do Top-1 de validação.
        se_k = float(np.sqrt(bestk * (1 - bestk) / len(c["val"])))
        lam_1se_k = min(l for l in lambdas if vt[l] >= bestk - se_k - 1e-12)
        cv["lambda_1se_por_fold"][str(k)] = lam_1se_k
        ev = eval_all(c, tr, lambdas, lam_k, lam_1se_k)
        ev["test"]["dqn_lstar_fixo"] = ev["test"][f"dqn_{lam_star}"]
        cv["folds"][str(k)] = {
            "particao": part, "lambda_star": lam_k, "lambda_1se": lam_1se_k, "ep_1se": se_k, "n_teste": len(c["test"]),
            "val_top1_grade": vt,
            "politicas": {name: ev["test"][name]["point"] for name in POLICIES + ["dqn_lstar_fixo"]},
            "ci95": {name: ev["test"][name]["ci95"] for name in POLICIES},
            "epochs_run": {n: h["epochs_run"] for n, h in tr["hparams"].items()},
            "pi_b_from_counts": ev["pi_b_from_counts"],
        }
        for name in oof:
            oof[name].append(ev["test"][name]["per_transition"])
        oof_ranks["dqn_lstar"].append(ev["ranks"]["dqn_lstar"])
        oof_ranks["dqn_l1se"].append(ev["ranks"]["dqn_l1se"])
        oof_ranks["seq"].append(ev["ranks"]["seq"])
        oof_conj.append(c["test"])
    for name in oof:
        per_fold = [cv["folds"][str(k)]["politicas"][name] for k in range(n_folds_run)]
        pooled = concat_pt(oof[name])
        from avaliacao import estimates, teto as teto_fn
        pooled_pt = {m: float(v) for m, v in estimates(pooled).items() if v is not None}
        pooled_r = np.concatenate([c.rewards for c in oof_conj])
        pooled_pt["teto"] = teto_fn(pooled_r, pooled_pt["top1"])
        pooled_pt["pct_teto"] = pooled_pt["reward_assoc"] / pooled_pt["teto"]
        cv["politicas"][name] = {
            "n_folds": n_folds_run, "n_transicoes_oof": int(len(pooled["r"])),
            **{m: msd([p[m] for p in per_fold]) for m in METRICS},
            "pooled_oof": pooled_pt, "ci95_bootstrap_oof": pooled_ci([pooled]),
        }
    # Diagnósticos por segmento e missão (política d, λ* por fold), OOF.
    segs = sum([c.segments for c in oof_conj], [])
    miss = sum([c.action_missions for c in oof_conj], [])
    allrank = np.concatenate(oof_ranks["dqn_lstar"])
    allact = np.concatenate([c.actions for c in oof_conj])
    seqp = np.concatenate([c.seq_pred for c in oof_conj])

    def grp(groups):
        hit = allrank[:, 0] == allact
        out = {}
        for g in sorted(set(groups)):
            m = np.array([x == g for x in groups])
            per = []
            off = 0
            for c in oof_conj:  # dp entre folds
                mk = np.array([x == g for x in groups[off:off + len(c)]])
                per.append(float(hit[off:off + len(c)][mk].mean()) if mk.any() else np.nan)
                off += len(c)
            out[g] = {"top1": float(hit[m].mean()), "n": int(m.sum()), "dp_entre_folds": msd(per)[1]}
        return out
    cv["top1_por_segmento"] = grp(segs)
    cv["top1_por_missao"] = grp(miss)
    cv["concordancia_dqn_seq"] = {"todas": float((allrank[:, 0] == seqp).mean()),
                                  "regra_definida": float((allrank[:, 0][seqp >= 0] == seqp[seqp >= 0]).mean()),
                                  "fracao_regra_definida": float((seqp >= 0).mean())}
    r1 = np.concatenate(oof_ranks["dqn_l1se"])[:, 0]
    cv["concordancia_dqn_l1se_seq"] = {"todas": float((r1 == seqp).mean()),
                                       "regra_definida": float((r1[seqp >= 0] == seqp[seqp >= 0]).mean()),
                                       "fracao_regra_definida": float((seqp >= 0).mean())}
    tempos["cv_s"] = time.time() - t
    R["cv"] = cv

    # ---- Cultural ----------------------------------------------------------
    t = time.time()
    log("[cultural]")
    R["cultural"] = experimento_cultural(fixed, lam_star, seeds, log)
    tempos["cultural_s"] = time.time() - t

    tempos["total_s"] = time.time() - T0
    R["tempos"] = tempos
    R["log"] = log_lines
    with (out_dir / "resultados.json").open("w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1, default=float)
    log(f"[fim] total {tempos['total_s'] / 60:.1f} min; resultados.json salvo")

    import relatorio
    relatorio.escrever(R, out_dir / "RESULTADOS.md")
    if not args.quick:
        import figuras
        figuras.gerar(R)


if __name__ == "__main__":
    main()

"""Escreve RESULTADOS.md a partir de resultados.json (tabelas em Markdown)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

POL_ORDER = ["seq", "seq_rep", "bc", "dqn0", "dqn_lstar", "dqn_l1se", "cql", "ppo"]
POL_LABEL = {
    "seq": "Regra sequencial (próxima do catálogo)",
    "seq_rep": "Regra sequencial + repetir terminal (variante)",
    "bc": "BC", "dqn0": "DQN λ = 0", "dqn_lstar": "DQN λ* (Top-1 máx.)", "dqn_l1se": "DQN λ_1EP", "cql": "CQL", "ppo": "PPO offline",
    "dqn_lstar_fixo": "DQN λ* (λ do split fixo)",
}


def f(x, nd=3):
    if x is None:
        return "—"
    try:
        if x != x:
            return "—"
    except TypeError:
        pass
    return f"{x:.{nd}f}".replace(".", ",")


def pm(ms, nd=3):
    return f"{f(ms[0], nd)} ± {f(ms[1], nd)}"


def ci(c, nd=3):
    return f"[{f(c[0], nd)}; {f(c[1], nd)}]"


def tabela_politicas(pols: Dict, chave_ci: str, n_label: str) -> str:
    rows = ["| Política | n | Top-1 | Top-3 | Rec. associada | Teto | % teto | SNIS [IC 95 %] | DR [IC 95 %] | DR-Q(CQL) | ESS |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in POL_ORDER + (["dqn_lstar_fixo"] if "dqn_lstar_fixo" in pols else []):
        if p not in pols:
            continue
        d = pols[p]
        c = d[chave_ci]
        rows.append(
            f"| {POL_LABEL[p]} | {d.get(n_label)} | {pm(d['top1'])} | {pm(d['top3'])} | {pm(d['reward_assoc'])} | "
            f"{pm(d['teto'])} | {pm([d['pct_teto'][0] * 100, d['pct_teto'][1] * 100], 1)} % | "
            f"{pm(d['snis'])} {ci(c['snis'])} | {pm(d['dr'])} {ci(c['dr'])} | {pm(d['dr_q_cql'], 2)} | {f(d['ess'][0], 0)} |")
    return "\n".join(rows)


def tabela_hparams(hp: Dict) -> str:
    rows = ["| Política | hiperparâmetros efetivamente usados |", "|---|---|"]
    nomes = {"bc": "BC", "cql": "CQL", "ppo": "PPO offline", "rmodel": "r̂(s,a) (modelo direto do DR)"}
    for k, v in hp.items():
        if k.startswith("dqn_") and k != "dqn_0.0":
            continue
        nome = nomes.get(k, "DQN (todos os λ da grade)")
        items = {kk: vv for kk, vv in v.items() if kk not in ("seed", "epochs_run", "best_val_top1", "best_val_loss", "best_val_mse", "lambda_reg")}
        rows.append(f"| {nome} | " + ", ".join(f"{kk}={vv}" for kk, vv in items.items()) + " |")
    return "\n".join(rows)


def escrever(R: Dict, path: Path) -> None:
    m = R["meta"]
    fx, cv, cu = R["split_fixo"], R["cv"], R["cultural"]
    L = []
    L.append("# Resultados da reexecução dos experimentos de DRL — 2026-10-10\n")
    L.append(f"Ambiente: Python {m['python']}, torch {m['torch']} (CPU), numpy {m['numpy']}. "
             f"Sementes (split fixo): {m['sementes_split_fixo']}; semente da validação cruzada: {m['semente_cv']}. "
             f"Bootstrap: {m['n_bootstrap']} reamostras por transição. Tempo total: {R['tempos']['total_s'] / 60:.1f} min "
             f"(preparação {R['tempos']['preparacao_s']:.0f} s; split fixo {R['tempos']['split_fixo_s'] / 60:.1f} min; "
             f"5-fold {R['tempos']['cv_s'] / 60:.1f} min; cultural {R['tempos']['cultural_s'] / 60:.1f} min).\n")
    L.append("Reprodução: `python run_all.py` nesta pasta (regenera resultados.json, este arquivo e as três figuras).\n")

    L.append("## 1. Partições por perfil (E2)\n")
    L.append("Os 25 perfis foram divididos em cinco grupos disjuntos, estratificados por segmento de `users.jsonl` "
             "(rodízio após embaralhar com semente 42). Split fixo: teste = grupo 0, validação = grupo 1, "
             "treino = grupos 2–4 (5/5/15 perfis). Validação cruzada: fold k usa teste = grupo k, validação = grupo k+1 "
             "(parada antecipada e seleção de λ), treino = os três grupos restantes.\n")
    seg = m["segmento_por_perfil"]
    for i, g in enumerate(m["folds_perfis"]):
        L.append(f"- grupo {i}: " + ", ".join(f"{u} ({seg[u][:3]})" for u in g))
    L.append("")
    L.append(f"Split fixo — treino: {', '.join(m['split_fixo']['train'])}; validação: {', '.join(m['split_fixo']['val'])}; "
             f"teste: {', '.join(m['split_fixo']['test'])}. Transições: {R['tamanhos']['split_fixo']} "
             f"(partição original por interação: {R['tamanhos']['original']}).\n")

    L.append("## 2. Tabela principal — validação cruzada 5-fold por perfil (semente 42)\n")
    L.append("Média ± dp entre folds (n = 5 folds); IC 95 % por bootstrap das transições fora-da-amostra agrupadas "
             f"(N = {cv['politicas']['bc']['n_transicoes_oof']}). λ* reselecionado na validação de cada fold: "
             + ", ".join(f"fold {k}: {v}" for k, v in cv["lambda_star_por_fold"].items()) + "; λ_1EP por fold: "
             + ", ".join(f"fold {k}: {v}" for k, v in cv["lambda_1se_por_fold"].items()) + ".\n")
    L.append(tabela_politicas(cv["politicas"], "ci95_bootstrap_oof", "n_folds"))
    L.append("\nValores agrupados (OOF) de Top-1 / recompensa associada / SNIS / DR: " + "; ".join(
        f"{POL_LABEL[p]}: {f(cv['politicas'][p]['pooled_oof']['top1'])} / {f(cv['politicas'][p]['pooled_oof']['reward_assoc'])} / "
        f"{f(cv['politicas'][p]['pooled_oof']['snis'])} / {f(cv['politicas'][p]['pooled_oof']['dr'])}" for p in POL_ORDER) + ".\n")

    L.append("## 3. Split fixo por perfil (5 sementes)\n")
    L.append(f"Média ± dp entre sementes (n = 5 execuções; a regra sequencial é determinística, dp = 0); IC 95 % por bootstrap "
             f"das {fx['politicas']['bc']['n_teste']} transições de teste, com a estimativa média entre sementes. "
             f"λ* = {fx['lambda_star']} (maior Top-1 médio de validação; desempate pelo menor λ). "
             f"π_b por contagem em {f(fx['pi_b_from_counts'][0] * 100, 1)} % das transições de teste (BC nas demais).\n")
    L.append(tabela_politicas(fx["politicas"], "ci95_bootstrap", "n_execucoes"))
    L.append("")

    L.append("## 4. Seleção de λ — grade de λ_reg (split fixo, 5 sementes)\n")
    L.append(f"Grade {[float(k) for k in fx['grade_lambda']]} (estendida com 2, 5 e 10). λ* (Top-1 máximo de validação, "
             f"desempate pelo menor λ) = {fx['lambda_star']}. Regra de um erro-padrão: EP = dp entre sementes / √5 do λ de máximo "
             f"= {f(fx['ep_1se'], 4)}; menor λ com Top-1 de validação ≥ máximo − EP: **λ_1EP = {fx['lambda_1se']}**. "
             "No 5-fold (uma execução por fold) o EP é o binomial √(p(1−p)/n_val) do máximo. Marcas: * = λ*, † = λ_1EP.\n")
    gb = fx["grade_lambda"][str(fx["lambda_1se_binomial"])]
    L.append(f"Com o EP binomial também no split fixo (EP = {f(fx['ep_binomial'], 4)}), λ_1EP = {fx['lambda_1se_binomial']}: "
             f"Top-1 teste {pm(gb['test_top1'])}, Top-3 {pm(gb['test_top3'])}, rec. assoc. {pm(gb['test_reward_assoc'])}, "
             f"SNIS {pm(gb['test_snis'])}, DR {pm(gb['test_dr'])}. O EP entre sementes ({f(fx['ep_1se'], 4)}) é muito menor que o binomial "
             "porque as sementes quase não alteram o Top-1 de validação; com ele a regra seleciona o próprio λ*.\n")
    L.append("| λ_reg | Top-1 val | Top-3 val | Rec. assoc. val | Top-1 teste | Top-3 teste | Rec. assoc. teste | SNIS teste | DR teste | épocas |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for lam, d in fx["grade_lambda"].items():
        star = (" *" if float(lam) == fx["lambda_star"] else "") + (" †" if float(lam) == fx["lambda_1se"] else "")
        L.append(f"| {f(float(lam), 3)}{star} | {pm(d['val_top1'])} | {pm(d['val_top3'])} | {pm(d['val_reward_assoc'])} | "
                 f"{pm(d['test_top1'])} | {pm(d['test_top3'])} | {pm(d['test_reward_assoc'])} | {pm(d['test_snis'])} | "
                 f"{pm(d['test_dr'])} | {f(d['epochs_run'][0], 0)} |")
    L.append("\nTop-1 de validação da grade em cada fold da validação cruzada: " + "; ".join(
        f"fold {k}: " + ", ".join(f"{lam}→{f(v)}" for lam, v in cv['folds'][k]['val_top1_grade'].items()) for k in cv["folds"]) + ".\n")

    L.append("## 5. Experimento cultural (E6)\n")
    L.append("País no estado (one-hot BR/AO/PT, 3 dimensões) em ambas as configurações; recompensa R (sem) ou "
             f"0,85·R + 0,15·f_cult (com). DQN λ* = {fx['lambda_star']}, split fixo por perfil, 5 sementes. "
             f"Países no teste: {cu['paises_teste']}.\n")
    L.append("| Configuração | Top-1 teste | Divergência por país |")
    L.append("|---|---|---|")
    for cw, d in cu["configs"].items():
        L.append(f"| f_cult peso {f(float(cw), 2)} | {pm(d['top1'])} | {pm([d['divergencia'][0] * 100, d['divergencia'][1] * 100], 1)} % |")
    tt = cu["teste_t"]
    L.append(f"\nTeste t de Welch (com vs sem): t = {f(tt['welch_t'], 2)}, p = {f(tt['welch_p'], 3)}; "
             f"pareado por semente: t = {f(tt['pareado_t'], 2)}, p = {f(tt['pareado_p'], 3)} (n = {tt['n_por_grupo']} por grupo).\n")
    L.append(cu["nota_f_cult"] + "\n")

    L.append("## 6. Diagnósticos (E7)\n")
    L.append("### Top-1 por segmento (DQN λ*, 5-fold OOF; dp entre folds)\n")
    L.append("| Segmento | Top-1 | n | dp entre folds |\n|---|---|---|---|")
    for g, d in cv["top1_por_segmento"].items():
        L.append(f"| {g} | {f(d['top1'])} | {d['n']} | {f(d['dp_entre_folds'])} |")
    L.append("\nSplit fixo (5 sementes): " + "; ".join(f"{g}: {pm(d['top1'])} (n = {d['n']})" for g, d in fx["top1_por_segmento"].items()) + ".\n")
    L.append("### Top-1 por missão da ação logada (DQN λ*, 5-fold OOF)\n")
    L.append("| Missão | Top-1 | n | dp entre folds |\n|---|---|---|---|")
    for g, d in cv["top1_por_missao"].items():
        L.append(f"| {g} | {f(d['top1'])} | {d['n']} | {f(d['dp_entre_folds'])} |")
    L.append("\nSplit fixo (5 sementes): " + "; ".join(f"{g}: {pm(d['top1'])} (n = {d['n']})" for g, d in fx["top1_por_missao"].items()) + ".\n")
    v = R["vazamento"]
    L.append("### Vazamento entre treino e teste\n")
    L.append("| Partição | estados de teste presentes no treino | pares (s,a) de teste presentes no treino | n teste | n treino |\n|---|---|---|---|---|")
    for k, d in v.items():
        L.append(f"| {k} | {f(d['estados_teste_no_treino'] * 100, 1)} % | {f(d['pares_sa_teste_no_treino'] * 100, 1)} % | {d['n_teste']} | {d['n_treino']} |")
    rc = R["recompensa"]
    L.append(f"\nRecompensa refinada: r_miss = 1 em {f(rc['fracao_r_miss_ativo'] * 100, 1)} % das {rc['n']} transições "
             f"(era 0 % com o codificador antigo); média {f(rc['media_refinada'])}; done = 1 em {f(rc['fracao_done'] * 100, 1)} %.\n")
    c1, c2 = fx["concordancia_dqn_seq"], cv["concordancia_dqn_seq"]
    L.append(f"Concordância transição a transição entre o argmax do DQN λ* e a regra sequencial: split fixo "
             f"{pm([c1['todas'][0] * 100, c1['todas'][1] * 100], 1)} % de todas as transições "
             f"({pm([c1['regra_definida'][0] * 100, c1['regra_definida'][1] * 100], 1)} % das {f(c1['fracao_regra_definida'] * 100, 1)} % em que a regra "
             f"tem predição); 5-fold OOF {f(c2['todas'] * 100, 1)} % / {f(c2['regra_definida'] * 100, 1)} %.\n")
    c3, c4 = fx["concordancia_dqn_l1se_seq"], cv["concordancia_dqn_l1se_seq"]
    L.append(f"Concordância DQN λ_1EP × regra sequencial: split fixo {pm([c3['todas'][0] * 100, c3['todas'][1] * 100], 1)} % "
             f"(todas) / {pm([c3['regra_definida'][0] * 100, c3['regra_definida'][1] * 100], 1)} % (regra definida); "
             f"5-fold OOF {f(c4['todas'] * 100, 1)} % / {f(c4['regra_definida'] * 100, 1)} %.\n")
    L.append(f"Regra sequencial no teste da partição ORIGINAL: Top-1 {f(R['regra_seq_teste_original']['top1'])} "
             f"(com repetição no terminal: {f(R['regra_seq_teste_original']['top1_com_repeticao'])}).\n")
    L.append("### Codificador (E1): médias por grupo de atributos\n")
    L.append("| grupo | s | s' corrigido | s' original |\n|---|---|---|---|")
    for g, d in R["codificador"].items():
        if g.startswith("_"):
            continue
        L.append(f"| {g} | {f(d['s'])} | {f(d['s_linha_corrigido'])} | {f(d['s_linha_original'])} |")
    L.append("\n" + R["codificador"]["_nota"] + ".\n")

    L.append("## 7. Hiperparâmetros executados\n")
    L.append(tabela_hparams(fx["hparams"]))
    L.append("\nÉpocas efetivamente treinadas (parada antecipada), split fixo, média entre sementes: " + ", ".join(
        f"DQN λ={lam}: {f(d['epochs_run'][0], 0)}" for lam, d in fx["grade_lambda"].items()) + "; " + ", ".join(
        f"{k}: {v['epochs_run']}" for k, v in fx["hparams"].items() if not k.startswith("dqn_")) + " (semente 42).\n")
    L.append("PPO offline: implementação de imitação ponderada por vantagem com recorte — a razão π/π_old é avaliada "
             "apenas nas ações logadas, a vantagem é r + γV(s') − V(s) com V ajustada nos mesmos dados, e não há "
             "correção off-policy em relação à política de comportamento.\n")
    L.append("OPE: SNIS com π_b por contagem empírica quando o estado codificado aparece ≥ 2 vezes no conjunto avaliado "
             "e pela BC (softmax) nos demais; π_b truncada em 0,01. DR = média do termo direto r̂(s, π(s)) + correção por "
             "importância auto-normalizada com r̂(s,a) (regressor de recompensa imediata, MLP 128, treinado no treino). "
             "DR-Q(CQL) usa a Q do CQL como modelo direto, como pede a especificação; está na escala de retorno "
             "descontado (γ = 0,95) e não na de recompensa imediata, por isso não é comparável às demais colunas.\n")

    L.append("## 8. O que mudou em relação a 2026-08-29 e por quê\n")
    L.append("(1) O codificador do estado seguinte recebia `next_state` sem `profile` e sem `missions_completed`: "
             "literacia 0, segmento 'iniciante' e flags de missão zeradas em TODOS os s'. Agora s' usa o profile da "
             "transição e missões derivadas de `tasks_completed` (também em s, para a mesma distribuição); r_miss, "
             "antes sempre 0, passa a ser ativo. (2) A partição original era por interação, com vazamento de estados entre "
             "treino e teste; a nova é por perfil (user_id), em split fixo e em 5-fold. (3) Entrou a regra sequencial "
             "determinística como baseline e OPE (SNIS, DR) com IC por bootstrap. (4) λ_reg foi reselecionado na "
             "validação, na nova grade, e a tabela principal passa a ser o 5-fold por perfil. Os números antigos "
             "(DQN 0,574/0,709; BC 0,901; CQL 0,809; λ = 0,05 → 0,832/1,000/0,563; divergência 8,7 % vs 7,2 %) "
             "foram obtidos com o codificador defeituoso e com vazamento, e não são comparáveis aos daqui.\n")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "resultados.json"
    escrever(json.loads(p.read_text(encoding="utf-8")), p.parent / "RESULTADOS.md")

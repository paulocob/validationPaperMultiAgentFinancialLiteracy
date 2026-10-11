"""Gera RESULTADOS.md a partir de resultados.json (todas as tabelas são calculadas
aqui; o texto interpretativo fica em NOTAS_INTERPRETACAO, escrito após a execução
completa e marcado como tal)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
HS = [0.0, 0.5, 1.0]


def f(x, nd=3):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    if isinstance(x, float) and x == float("inf"):
        return "∞"
    return (f"%.{nd}f" % x).replace(".", ",")


def fn(n, delta):
    """n por braço legível: sem sentido se o efeito não favorece a política."""
    if delta is None or delta <= 0:
        return "n/a (Δ ≤ 0)"
    if n > 10000:
        return "> 10.000"
    return f(n, 0)


def ic(v, nd=3):
    return f"[{f(v[0], nd)}; {f(v[1], nd)}]"


def hs(h):
    return ("%g" % h).replace(".", ",")


def escrever(R, destino: Path):
    T, rot = R["tabela"], R["rotulos"]
    p, cal, cfg = R["params"], R["calibracao"], R["cfg"]
    pols = [x for x in ["curriculo", "aleatoria", "linucb", "ts", "dqn_off_1ep", "dqn_off_snis", "dqn_on",
                        "llm_zero", "llm_few", "guloso_kt", "oraculo", "linucb_crenca", "linucb_kt_binario"]
            if any(f"{x}|{h}" in T for h in HS)]
    L = []
    w = L.append
    w("# Resultados — simulador cognitivo e comparação de políticas (2026-10-11)\n")
    if R.get("quick"):
        w("> **Execução de fumaça (`--quick`)**: populações e grades reduzidas; números sem valor substantivo.\n")
    amb = R["ambiente"]
    tp = R["tempos_s"]
    w(f"Ambiente: Python {amb['python']}, numpy {amb['numpy']}, torch {amb['torch']} (CPU). "
      f"Tempo total {f(tp['total'] / 60, 1)} min (calibração {f(tp['calibracao'] / 60, 1)}; trabalhos em paralelo "
      f"{f(tp['trabalhos'] / 60, 1)}; LLM {f(tp.get('llm', 0) / 60, 1)}).\n")
    w("Reprodução: `python run_all.py` nesta pasta (regenera resultados.json, este arquivo e as três figuras em "
      "`./figuras/fig_sim_*.pdf` ou em `$FIG_DIR`); `python run_all.py --quick` para fumaça.\n")
    w(f"Desenho: h ∈ {{0; 0,5; 1}} × {len(cfg['seeds'])} sementes de população; população de teste de "
      f"{cfg['n_teste']} aprendizes por semente (total {cfg['n_teste'] * len(cfg['seeds'])} por célula), disjunta das "
      f"de treino/validação; todas as políticas enfrentam os mesmos aprendizes e o mesmo ruído (números aleatórios "
      f"comuns), o que permite diferenças pareadas. IC 95 % por bootstrap de aprendizes ({2000} reamostras).\n")

    w("## 1. Simulador\n")
    w("Catálogo: 11 tarefas de `data/tasks.json` (M1 5, M2 2, M3 2, M6 2), uma habilidade por tarefa; pré-requisito "
      "de cada tarefa = a anterior na cadeia do arquivo. O catálogo NÃO bloqueia ações: as 11 tarefas são sempre "
      "escolhíveis (pular e repetir são justamente as liberdades de uma política adaptativa); os pré-requisitos agem "
      "pelo modelo cognitivo (P(T)×0,3 e frustração). Currículo fixo = primeira tarefa não concluída na ordem do "
      "catálogo; com tudo concluído, revisão da tarefa praticada há mais tempo (1…11 e depois 1–4).\n")
    w("| Parâmetro | Distribuição (h = heterogeneidade) |")
    w("|---|---|")
    w(f"| Segmentos | iniciante/intermediário/avançado com proporções 9/11/5 (perfis de users.jsonl) |")
    w(f"| P(L0) | média do segmento {p['L0_seg']} → μ̄ + h(μ_seg − μ̄); aprendiz ~ Beta(média, dp = {p['sd_L0_aprendiz']}·h); "
      f"habilidade ~ Beta(média do aprendiz, dp = {p['sd_L0_habilidade']}·h) |")
    w(f"| P(T) | c_T × {p['T_seg_rel']} por segmento, c_T = {f(p['c_T'], 4)} (calibrado); aprendiz dp = {p['sd_T_aprendiz']}·h; "
      f"habilidade dp = {p['sd_T_habilidade']}·h |")
    w(f"| P(S), P(G), P(F) | Beta com médias {p['S_media']}, {p['G_media']}, {p['F_media']}/passo; dp = "
      f"{p['sd_S']}·h, {p['sd_G']}·h, {p['sd_F']}·h (por aprendiz) |")
    w(f"| Pré-requisitos | P(T) efetivo × {p['mult_prereq']} se maestria média dos pré-req. < {p['limiar_prereq']} |")
    w(f"| Esquecimento | L ← L − F(L − L0) por passo sem exposição (conhecimento prévio tratado como consolidado) |")
    w(f"| Abandono por passo | {p['abandono_base']} + α(Fr + Bo), α = {f(p['alfa_abandono'], 4)} (calibrado); Fr/Bo = "
      f"acumuladores (decaimento {p['decaimento_afeto']}) de frustração (pré-req. < {p['limiar_prereq']}) e tédio "
      f"(maestria > {p['limiar_tedio']}) |")
    w(f"| Sentimento observável | ansioso se Fr ≥ 0,75; cansado se Bo ≥ 0,75; motivado se acertou a última; senão neutro; "
      f"{int(p['ruido_sentimento'] * 100)} % de ruído uniforme |")
    w("| Episódio | até H = 15 decisões, abandono, ou maestria ≥ 0,95 em todas as habilidades |")
    w("| Desfecho | pré/pós = média de P(correto) = L(1−S)+(1−L)G nas 11 habilidades; pós no relógio H para todos "
      "(quem sai antes esquece até H: intenção de tratar); Hake g = (pós−pré)/(1−pré); retenção após +30 passos |")
    w("")
    w(f"**Calibração (S6).** Única âncora empírica: Hake do tutor no piloto brasileiro (0,51). c_T ajustado para "
      f"Hake do currículo = {cal['alvo_hake_h05']} em h = 0,5; α ajustado para abandono do currículo = "
      f"{cal['alvo_abandono_h1']} em h = 1 (escolha de desenho, centro da faixa 10–20 % da especificação, não âncora "
      f"empírica). Bisseções alternadas, população de calibração própria (semente {cal['semente']}, n = {cal['n']}). "
      "Os demais parâmetros são plausíveis, NÃO estimados. Currículo na população de calibração:\n")
    w("| h | Hake do currículo | Abandono do currículo |")
    w("|---|---|---|")
    for h in HS:
        c = cal["curriculo_por_h"][str(h)]
        w(f"| {hs(h)} | {f(c['hake'])} | {f(c['abandono'])} |")
    w("")
    w("**Observação do agente (S7)** — 46 dimensões: as 34 da família de 10/10 (literacia ≈ maestria média "
      "estimada; pontos/1000; razão de tarefas; razão de missões; progresso na missão atual; acerto na última "
      "tarefa (ocupa o lugar de cur_completed); tarefa atual terminal; 4 flags de missão; segmento (3); sentimento "
      "(5); missão atual (4); tarefa atual (11)) + maestria estimada por habilidade (11) + passo/H. A maestria "
      "verdadeira não é observada. O rastreador é um BKT com maestria latente contínua (crença Beta por habilidade, "
      "atualização por casamento de momentos), com os parâmetros MÉDIOS do segmento no nível h e a mesma estrutura do "
      "simulador (pré-requisitos, esquecimento) — hipótese favorável às políticas adaptativas. Ele recebe um pré-teste "
      "observado de 2 itens por habilidade.\n")
    w("**Recompensa de treino (S8)** = aprendizado estimado pelo rastreador no passo (termo de transição do BKT, "
      "sem a revisão de crença causada pela resposta) − 0,01 por passo − 1,0 se o aprendiz abandona. O desfecho de "
      "avaliação é o Hake no pós-teste simulado, calculado com a maestria verdadeira: nenhuma política é treinada nem "
      "selecionada com ele.\n")

    w("## 2. Teste de sanidade (antes de treinar)\n")
    w("| h | Currículo | Aleatória | Oráculo míope | divergência do oráculo |")
    w("|---|---|---|---|---|")
    for h in HS:
        s = R["sanidade"][str(h)]
        w(f"| {hs(h)} | {f(s['curriculo']['hake'])} | {f(s['aleatoria']['hake'])} | {f(s['oraculo']['hake'])} | "
          f"{f(100 * s['oraculo']['diverg'], 1)} % |")
    w(f"\nCritérios: oráculo ≥ currículo em todo h; |oráculo − currículo| < 0,01 em h = 0; aleatória abaixo de ambos. "
      f"Resultado: **{'aprovado' if R['sanidade']['aprovado'] else 'REPROVADO'}**.\n")

    w("## 3. Tabela principal (população de teste, todas as sementes agrupadas)\n")
    w("Hake imediato e de retenção: média [IC 95 %]; abandono: proporção [IC 95 %]; divergência: fração das decisões "
      "em que a política escolhe outra tarefa que não a do currículo no mesmo estado. Maestria: fração que atinge "
      "≥ 0,95 em todas as habilidades e passos médios até isso.\n")
    for h in HS:
        w(f"### h = {hs(h)}\n")
        w("| Política | n | Hake imediato | Hake retenção | Abandono | Divergência | Maestria (fração; passos) | dp Hake entre sementes |")
        w("|---|---|---|---|---|---|---|---|")
        for x in pols:
            k = f"{x}|{h}"
            if k not in T:
                continue
            e = T[k]
            w(f"| {rot[x]} | {e['n_aprendizes']} | {f(e['hake'])} {ic(e['hake_ic'])} | {f(e['hake_ret'])} "
              f"{ic(e['hake_ret_ic'])} | {f(e['abandono'])} {ic(e['abandono_ic'])} | {f(e['diverg'])} "
              f"{ic(e['diverg_ic'])} | {f(e['frac_maestria'])}; {f(e['passos_ate_maestria'], 1)} | "
              f"{f(e['hake_dp_entre_sementes'], 4)} |")
        w("")

    w("## 4. Tamanhos de efeito vs currículo e implicação para o ensaio (A1, A3)\n")
    w("Diferença de Hake pareada (mesmos aprendizes) com IC 95 %; d de Cohen = diferença / dp agrupado entre aprendizes "
      "(como num ensaio com braços independentes). n por braço para poder 0,80 e α = 0,05 bilateral: "
      "n = 2(z₀,₉₇₅ + z₀,₈₀)²σ²/Δ². Duas versões: (a) Hake calculado com a probabilidade de acerto (sem erro de "
      "medida); (b) Hake num teste de 10 itens dicotômicos amostrado (instrumento do piloto), que acrescenta erro de "
      f"medida. Plano A13: MDE Δg = {f(R['mde_plano'], 2)} com n ≈ {R['n_plano']} por braço, o que equivale a supor "
      f"σ ≈ {f(R['mde_plano'] * (R['n_plano'] / (2 * 7.849)) ** 0.5, 2)}.\n")
    w("| h | Política | Δ Hake [IC 95 %] | d | n/braço (a) | Δ Hake 10 itens | dp 10 itens | d 10 itens | n/braço (b) | Δ retenção | Δ abandono |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for h in HS:
        for x in pols:
            k = f"{x}|{h}"
            if x == "curriculo" or k not in T:
                continue
            v = T[k]["vs_curriculo"]
            w(f"| {hs(h)} | {rot[x]} | {f(v['dif_hake'])} {ic(v['dif_hake_ic'])} | {f(v['d_cohen'], 2)} | "
              f"{fn(v['n_por_braco'], v['dif_hake'])} | {f(v['dif_hake10'])} | {f(v['sd_agrupado10'])} | {f(v['d_cohen10'], 2)} | "
              f"{fn(v['n_por_braco10'], v['dif_hake10'])} | {f(v['dif_ret'])} | {f(v['dif_abandono'])} |")
    w("")

    w("## 5. Validação da OPE (A2)\n")
    w("Logs: currículo com ε = 0,2 uniforme sobre as 11 tarefas, numa população própria por semente "
      f"({cfg['n_log_ope']} aprendizes). Estimadores de 10/10 por transição (SNIS e DR com r̂(s,a) = MLP de 10/10), "
      "com π_b CONHECIDA. Verdadeiro = recompensa média por passo com a política implantada no simulador. Médias "
      "entre sementes; IC = média dos IC por semente (bootstrap por aprendiz).\n")
    for h in HS:
        o = R["ope"][str(h)]
        w(f"### h = {hs(h)}\n")
        w("| Política | Verdadeiro (rec./passo) | SNIS [IC] | DR [IC] | Hake verdadeiro | ESS | concordância com a ação logada |")
        w("|---|---|---|---|---|---|---|")
        for x, e in o["politicas"].items():
            w(f"| {rot[x]} | {f(e['verdadeiro_rec_passo'])} | {f(e['snis'])} {ic(e['snis_ic_medio'])} | {f(e['dr'])} "
              f"{ic(e['dr_ic_medio'])} | {f(e['hake'])} | {f(e['ess_medio'], 0)} | {f(e['concord_log'])} |")
        c = o["concordancia"]
        w(f"\nτ de Kendall (ordenação das {len(o['politicas'])} políticas): SNIS × verdadeiro {f(c['snis']['kendall_vs_rec_passo'], 2)}, "
          f"DR × verdadeiro {f(c['dr']['kendall_vs_rec_passo'], 2)}; SNIS × Hake {f(c['snis']['kendall_vs_hake'], 2)}, "
          f"DR × Hake {f(c['dr']['kendall_vs_hake'], 2)}; recompensa verdadeira × Hake {f(c['rec_passo_vs_hake'], 2)}. "
          f"Erro absoluto médio: SNIS {f(c['snis']['erro_abs_medio'])}, DR {f(c['dr']['erro_abs_medio'])}. "
          f"A melhor política pela SNIS {'é' if c['snis']['top1_coincide_hake'] else 'NÃO é'} a melhor pelo Hake; "
          f"pela DR {'é' if c['dr']['top1_coincide_hake'] else 'NÃO é'}.\n")

    w("## 6. Seleção de λ do DQN offline\n")
    w("Grade λ_reg ∈ " + str(cfg["lambdas"]) + ". Dois critérios de seleção na validação (logs do currículo + ε): "
      "(i) como em 10/10, Top-1 contra a ação logada e regra de um erro-padrão binomial (menor λ com Top-1 ≥ máx − EP); "
      "(ii) maior SNIS de validação. Configuração do DQN: a de 10/10 (MLP 256×2, dropout 0,1, lr 3e-3, γ 0,9, Huber, "
      "alvo a cada 5 épocas, parada antecipada pelo Top-1 de validação) com lote 256 e 40 épocas/paciência 8 (há ~15× "
      "mais transições que em 10/10).\n")
    w("| h | semente | transições treino/val | λ_1EP | λ (Top-1 máx.) | λ_SNIS | Top-1 val por λ | SNIS val por λ |")
    w("|---|---|---|---|---|---|---|---|")
    for j in R["jobs"]:
        g = j["hparams"]["dqn_off"]
        w(f"| {hs(j['h'])} | {j['seed']} | {j['n_trans_log']['treino']}/{j['n_trans_log']['val']} | {g['lambda_1ep']} | "
          f"{g['lambda_top1max']} | {g['lambda_snis']} | "
          + ", ".join(f"{l}: {f(v['top1_val'])}" for l, v in g["grade"].items()) + " | "
          + ", ".join(f"{l}: {f(v['snis_val'])}" for l, v in g["grade"].items()) + " |")
    w("")
    j0 = R["jobs"][0]["hparams"]["dqn_on"]
    w(f"DQN online: Double DQN, MLP 256×2, lr {j0['lr']}, γ {j0['gamma']}, lote {j0['batch']}, replay {j0['buffer']}, "
      f"{j0['aprendizes']} aprendizes de treino por (h, semente) em lotes de {j0['envs']}, "
      f"{j0['updates_por_passo']} atualizações por passo, ε de {j0['eps_ini']} a {j0['eps_fim']} na primeira metade, "
      "checkpoint escolhido pelo retorno médio de treino numa população de validação (não pelo Hake). "
      "Bandits: 2.000 aprendizes de treino (lotes de 100 em paralelo, atualização a cada passo), LinUCB α = 1, TS "
      "v = 0,5 (um sorteio de θ por braço e por passo do lote), ridge 1; congelados = argmax da média posterior.\n")

    w("## 7. Planejador LLM (P5)\n")
    llm = R["llm"]
    if llm.get("executado"):
        w(f"Modelo: `{llm['modelo']}` via transformers em CPU (bf16). Tempo de carga {f(llm['tempo_carga_s'], 0)} s; "
          f"custo estimado {f(llm['s_por_prompt_estimado'], 2)} s/prompt; {llm['n_aprendizes']} aprendizes × "
          f"{len(llm['seeds'])} sementes × 3 h × 2 variantes. {llm.get('reducao', '')}\n")
        w("| execução | decisões | chamadas ao modelo | tempo (s) |")
        w("|---|---|---|---|")
        for k, v in llm["por_execucao"].items():
            w(f"| {k} | {v['decisoes']} | {v['chamadas_modelo']} | {f(v['tempo_s'], 0)} |")
    else:
        w(f"**NÃO executado.** Motivo: {llm.get('motivo', '—')}\n")
        w("O código (`planejador_llm.py`) está completo e a mecânica (texto do estado, memorização por prompt, escolha "
          "restrita A–K por log-probabilidade) foi verificada com um pontuador determinístico de teste "
          "(`python planejador_llm.py`); a chamada ao modelo real não pôde ser exercitada.\n")
    if "prompt" in llm:
        w("Prompt do sistema (exato):\n")
        w("```\n" + llm["prompt"]["sistema"] + "\n```\n")
        w("Exemplo de mensagem de estado (aprendiz fora das populações de avaliação):\n")
        w("```\n" + llm["prompt"]["exemplo_estado"] + "\n```\n")
        w("Few-shot: os três pares (estado, letra do currículo) abaixo entram como turnos user/assistant antes do estado:\n")
        for e in llm["prompt"]["few_shot"]:
            w("```\n" + e["estado"] + "\n→ " + e["resposta"] + "\n```\n")

    if "NOTAS" in R:
        w("## 8. Interpretação, decisões e pendências\n")
        w(R["NOTAS"])
    if "sensibilidade" in R and not R["sensibilidade"].get("quick"):
        S = R["sensibilidade"]
        w("## 9. Análise de sensibilidade (parâmetros não calibrados, h = 1)\n")
        w(f"Grade: multiplicador de P(T) com pré-requisito não dominado {S['grade']['mult_prereq']}, limiar de "
          f"pré-requisito {S['grade']['limiar_prereq']}, escala de todos os dp de heterogeneidade "
          f"{S['grade']['escala_dispersao']} (27 combinações). c_T recalibrado por bisseção em CADA combinação para Hake "
          f"do currículo = 0,50 em h = 0,5; α fixo em {f(S['alfa_abandono_fixo'], 4)}, por isso o abandono do currículo "
          f"em h = 1 varia entre combinações. Em h = 1, {len(S['cfg']['seeds'])} sementes × {S['cfg']['n_teste']} "
          f"aprendizes, com as sementes e populações da execução principal; a combinação 0,3/0,6/1 corresponde às "
          f"sementes 0–2 da tabela principal. Δg = diferença pareada vs currículo [IC 95 %, bootstrap de aprendizes]. "
          f"Tempo: {f(S['tempo_total_s'] / 60, 1)} min. Figura: fig_sim_sensibilidade.pdf. Reprodução: "
          f"`python sensibilidade.py` depois de `python run_all.py`.\n")
        w("| mult. | limiar | escala | c_T | Hake currículo | aband. currículo | Δg LinUCB | Δg DQN online | Δg guloso | Δg oráculo | teto perdido (DQN on.) |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in S["combinacoes"]:
            P = c["politicas"]
            w(f"| {hs(c['mult'])} | {hs(c['limiar'])} | {hs(c['escala'])} | {f(c['c_T'], 3)} | {f(P['curriculo']['hake'])} | "
              f"{f(P['curriculo']['abandono'])} | {f(P['linucb']['dg'])} {ic(P['linucb']['dg_ic'])} | "
              f"{f(P['dqn_on']['dg'])} {ic(P['dqn_on']['dg_ic'])} | {f(P['guloso_kt']['dg'])} | "
              f"{f(P['oraculo']['dg'])} | {f(100 * c['frac_teto_perdida_dqn_on'], 0)} % |")
        fx = S["faixas_dg"]
        w(f"\nFaixas de Δg nas 27 combinações: LinUCB {f(fx['linucb'][0])} a {f(fx['linucb'][1])}; DQN online "
          f"{f(fx['dqn_on'][0])} a {f(fx['dqn_on'][1])}; guloso {f(fx['guloso_kt'][0])} a {f(fx['guloso_kt'][1])}; "
          f"oráculo {f(fx['oraculo'][0])} a {f(fx['oraculo'][1])}. Fração do teto do oráculo perdida pelo DQN online: "
          f"{f(100 * S['faixa_frac_teto_perdida_dqn_on'][0], 0)} % a {f(100 * S['faixa_frac_teto_perdida_dqn_on'][1], 0)} %; "
          f"pela melhor política implementável da combinação: {f(100 * S['faixa_frac_teto_perdida_melhor'][0], 0)} % a "
          f"{f(100 * S['faixa_frac_teto_perdida_melhor'][1], 0)} %.\n")
        if "NOTAS_SENS" in R:
            w(R["NOTAS_SENS"])
    destino.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[relatório] {destino}")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI / "resultados.json"
    R = json.loads(p.read_text(encoding="utf-8"))
    notas = AQUI / "NOTAS_INTERPRETACAO.md"
    if notas.exists():
        R["NOTAS"] = notas.read_text(encoding="utf-8")
    notas_s = AQUI / "NOTAS_SENSIBILIDADE.md"
    if notas_s.exists():
        R["NOTAS_SENS"] = notas_s.read_text(encoding="utf-8")
    escrever(R, AQUI / ("RESULTADOS_quick.md" if R.get("quick") else "RESULTADOS.md"))

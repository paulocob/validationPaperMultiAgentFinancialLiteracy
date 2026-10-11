# Resultados — simulador cognitivo e comparação de políticas (2026-10-11)

Ambiente: Python 3.12.10, numpy 2.5.3, torch 2.14.1+cpu (CPU). Tempo total 17,5 min (calibração 1,7; trabalhos em paralelo 13,8; LLM 1,7).

Reprodução: `python run_all.py` nesta pasta (regenera resultados.json, este arquivo e as três figuras em `Qualificação de Doutorado - Versão Enxuta/figures/fig_sim_*.pdf`); `python run_all.py --quick` para fumaça.

Desenho: h ∈ {0; 0,5; 1} × 5 sementes de população; população de teste de 1000 aprendizes por semente (total 5000 por célula), disjunta das de treino/validação; todas as políticas enfrentam os mesmos aprendizes e o mesmo ruído (números aleatórios comuns), o que permite diferenças pareadas. IC 95 % por bootstrap de aprendizes (2000 reamostras).

## 1. Simulador

Catálogo: 11 tarefas de `data/tasks.json` (M1 5, M2 2, M3 2, M6 2), uma habilidade por tarefa; pré-requisito de cada tarefa = a anterior na cadeia do arquivo. O catálogo NÃO bloqueia ações: as 11 tarefas são sempre escolhíveis (pular e repetir são justamente as liberdades de uma política adaptativa); os pré-requisitos agem pelo modelo cognitivo (P(T)×0,3 e frustração). Currículo fixo = primeira tarefa não concluída na ordem do catálogo; com tudo concluído, revisão da tarefa praticada há mais tempo (1…11 e depois 1–4).

| Parâmetro | Distribuição (h = heterogeneidade) |
|---|---|
| Segmentos | iniciante/intermediário/avançado com proporções 9/11/5 (perfis de users.jsonl) |
| P(L0) | média do segmento [0.15, 0.35, 0.6] → μ̄ + h(μ_seg − μ̄); aprendiz ~ Beta(média, dp = 0.1·h); habilidade ~ Beta(média do aprendiz, dp = 0.2·h) |
| P(T) | c_T × [0.75, 1.0, 1.25] por segmento, c_T = 0,6586 (calibrado); aprendiz dp = 0.1·h; habilidade dp = 0.05·h |
| P(S), P(G), P(F) | Beta com médias 0.1, 0.2, 0.01/passo; dp = 0.04·h, 0.06·h, 0.005·h (por aprendiz) |
| Pré-requisitos | P(T) efetivo × 0.3 se maestria média dos pré-req. < 0.6 |
| Esquecimento | L ← L − F(L − L0) por passo sem exposição (conhecimento prévio tratado como consolidado) |
| Abandono por passo | 0.005 + α(Fr + Bo), α = 0,0108 (calibrado); Fr/Bo = acumuladores (decaimento 0.5) de frustração (pré-req. < 0.6) e tédio (maestria > 0.9) |
| Sentimento observável | ansioso se Fr ≥ 0,75; cansado se Bo ≥ 0,75; motivado se acertou a última; senão neutro; 25 % de ruído uniforme |
| Episódio | até H = 15 decisões, abandono, ou maestria ≥ 0,95 em todas as habilidades |
| Desfecho | pré/pós = média de P(correto) = L(1−S)+(1−L)G nas 11 habilidades; pós no relógio H para todos (quem sai antes esquece até H: intenção de tratar); Hake g = (pós−pré)/(1−pré); retenção após +30 passos |

**Calibração (S6).** Única âncora empírica: Hake do tutor no piloto brasileiro (0,51). c_T ajustado para Hake do currículo = 0.5 em h = 0,5; α ajustado para abandono do currículo = 0.15 em h = 1 (escolha de desenho, centro da faixa 10–20 % da especificação, não âncora empírica). Bisseções alternadas, população de calibração própria (semente 999, n = 20000). Os demais parâmetros são plausíveis, NÃO estimados. Currículo na população de calibração:

| h | Hake do currículo | Abandono do currículo |
|---|---|---|
| 0 | 0,542 | 0,071 |
| 0,5 | 0,500 | 0,098 |
| 1 | 0,436 | 0,150 |

**Observação do agente (S7)** — 46 dimensões: as 34 da família de 10/10 (literacia ≈ maestria média estimada; pontos/1000; razão de tarefas; razão de missões; progresso na missão atual; acerto na última tarefa (ocupa o lugar de cur_completed); tarefa atual terminal; 4 flags de missão; segmento (3); sentimento (5); missão atual (4); tarefa atual (11)) + maestria estimada por habilidade (11) + passo/H. A maestria verdadeira não é observada. O rastreador é um BKT com maestria latente contínua (crença Beta por habilidade, atualização por casamento de momentos), com os parâmetros MÉDIOS do segmento no nível h e a mesma estrutura do simulador (pré-requisitos, esquecimento) — hipótese favorável às políticas adaptativas. Ele recebe um pré-teste observado de 2 itens por habilidade.

**Recompensa de treino (S8)** = aprendizado estimado pelo rastreador no passo (termo de transição do BKT, sem a revisão de crença causada pela resposta) − 0,01 por passo − 1,0 se o aprendiz abandona. O desfecho de avaliação é o Hake no pós-teste simulado, calculado com a maestria verdadeira: nenhuma política é treinada nem selecionada com ele.

## 2. Teste de sanidade (antes de treinar)

| h | Currículo | Aleatória | Oráculo míope | divergência do oráculo |
|---|---|---|---|---|
| 0 | 0,541 | 0,289 | 0,541 | 0,0 % |
| 0,5 | 0,502 | 0,283 | 0,542 | 29,1 % |
| 1 | 0,436 | 0,259 | 0,526 | 62,3 % |

Critérios: oráculo ≥ currículo em todo h; |oráculo − currículo| < 0,01 em h = 0; aleatória abaixo de ambos. Resultado: **aprovado**.

## 3. Tabela principal (população de teste, todas as sementes agrupadas)

Hake imediato e de retenção: média [IC 95 %]; abandono: proporção [IC 95 %]; divergência: fração das decisões em que a política escolhe outra tarefa que não a do currículo no mesmo estado. Maestria: fração que atinge ≥ 0,95 em todas as habilidades e passos médios até isso.

### h = 0

| Política | n | Hake imediato | Hake retenção | Abandono | Divergência | Maestria (fração; passos) | dp Hake entre sementes |
|---|---|---|---|---|---|---|---|
| Currículo fixo (P1) | 5000 | 0,541 [0,538; 0,543] | 0,400 [0,398; 0,401] | 0,075 [0,068; 0,082] | 0,000 [0,000; 0,000] | 0,000; — | 0,0015 |
| Aleatória c/ pré-req. (P2) | 5000 | 0,287 [0,285; 0,289] | 0,212 [0,211; 0,214] | 0,166 [0,156; 0,176] | 0,669 [0,667; 0,672] | 0,000; — | 0,0019 |
| LinUCB (P3) | 5000 | 0,540 [0,538; 0,542] | 0,399 [0,398; 0,401] | 0,075 [0,068; 0,082] | 0,111 [0,107; 0,115] | 0,000; — | 0,0012 |
| Thompson linear (P3) | 5000 | 0,540 [0,538; 0,542] | 0,399 [0,398; 0,401] | 0,075 [0,067; 0,083] | 0,057 [0,054; 0,060] | 0,000; — | 0,0015 |
| DQN offline, λ por Top-1/1EP (P4a) | 5000 | 0,541 [0,538; 0,543] | 0,400 [0,398; 0,401] | 0,075 [0,068; 0,082] | 0,000 [0,000; 0,000] | 0,000; — | 0,0015 |
| DQN offline, λ por SNIS (P4a) | 5000 | 0,528 [0,526; 0,530] | 0,391 [0,389; 0,392] | 0,080 [0,072; 0,087] | 0,225 [0,223; 0,226] | 0,000; — | 0,0018 |
| DQN online (P4b) | 5000 | 0,538 [0,536; 0,540] | 0,398 [0,396; 0,399] | 0,075 [0,067; 0,082] | 0,198 [0,196; 0,200] | 0,000; — | 0,0014 |
| Guloso pelo rastreador (referência implementável) | 5000 | 0,541 [0,538; 0,543] | 0,400 [0,398; 0,401] | 0,075 [0,068; 0,082] | 0,000 [0,000; 0,000] | 0,000; — | 0,0015 |
| Oráculo míope (P6, teto) | 5000 | 0,541 [0,538; 0,543] | 0,400 [0,398; 0,401] | 0,075 [0,068; 0,082] | 0,000 [0,000; 0,000] | 0,000; — | 0,0015 |
| LinUCB, recompensa c/ revisão de crença (sens.) | 5000 | 0,540 [0,538; 0,542] | 0,399 [0,398; 0,401] | 0,075 [0,068; 0,082] | 0,111 [0,107; 0,114] | 0,000; — | 0,0012 |
| LinUCB, rastreador BKT binário (sens.) | 5000 | 0,296 [0,292; 0,300] | 0,219 [0,216; 0,222] | 0,237 [0,225; 0,249] | 0,706 [0,699; 0,714] | 0,000; — | 0,0047 |

### h = 0,5

| Política | n | Hake imediato | Hake retenção | Abandono | Divergência | Maestria (fração; passos) | dp Hake entre sementes |
|---|---|---|---|---|---|---|---|
| Currículo fixo (P1) | 5000 | 0,500 [0,497; 0,504] | 0,372 [0,369; 0,375] | 0,102 [0,094; 0,111] | 0,000 [0,000; 0,000] | 0,000; — | 0,0058 |
| Aleatória c/ pré-req. (P2) | 5000 | 0,279 [0,276; 0,281] | 0,207 [0,205; 0,209] | 0,173 [0,163; 0,184] | 0,670 [0,668; 0,673] | 0,000; — | 0,0032 |
| LinUCB (P3) | 5000 | 0,499 [0,495; 0,503] | 0,371 [0,368; 0,374] | 0,104 [0,095; 0,112] | 0,096 [0,092; 0,099] | 0,000; — | 0,0056 |
| Thompson linear (P3) | 5000 | 0,498 [0,494; 0,502] | 0,370 [0,367; 0,373] | 0,104 [0,095; 0,112] | 0,108 [0,105; 0,112] | 0,000; — | 0,0050 |
| DQN offline, λ por Top-1/1EP (P4a) | 5000 | 0,500 [0,497; 0,504] | 0,372 [0,369; 0,375] | 0,102 [0,094; 0,110] | 0,000 [0,000; 0,000] | 0,000; — | 0,0058 |
| DQN offline, λ por SNIS (P4a) | 5000 | 0,484 [0,480; 0,487] | 0,360 [0,357; 0,363] | 0,112 [0,103; 0,121] | 0,215 [0,213; 0,217] | 0,000; — | 0,0090 |
| DQN online (P4b) | 5000 | 0,495 [0,491; 0,499] | 0,368 [0,365; 0,371] | 0,106 [0,097; 0,115] | 0,174 [0,172; 0,176] | 0,000; — | 0,0070 |
| Guloso pelo rastreador (referência implementável) | 5000 | 0,496 [0,492; 0,500] | 0,369 [0,366; 0,372] | 0,106 [0,098; 0,114] | 0,162 [0,160; 0,163] | 0,000; — | 0,0059 |
| Oráculo míope (P6, teto) | 5000 | 0,541 [0,538; 0,543] | 0,402 [0,400; 0,404] | 0,077 [0,070; 0,084] | 0,290 [0,285; 0,295] | 0,000; — | 0,0039 |
| LinUCB, recompensa c/ revisão de crença (sens.) | 5000 | 0,499 [0,495; 0,503] | 0,371 [0,368; 0,374] | 0,104 [0,097; 0,113] | 0,096 [0,092; 0,099] | 0,000; — | 0,0058 |
| LinUCB, rastreador BKT binário (sens.) | 5000 | 0,315 [0,311; 0,318] | 0,234 [0,231; 0,237] | 0,212 [0,200; 0,224] | 0,745 [0,739; 0,751] | 0,000; — | 0,0064 |

### h = 1

| Política | n | Hake imediato | Hake retenção | Abandono | Divergência | Maestria (fração; passos) | dp Hake entre sementes |
|---|---|---|---|---|---|---|---|
| Currículo fixo (P1) | 5000 | 0,436 [0,430; 0,441] | 0,327 [0,323; 0,331] | 0,150 [0,140; 0,161] | 0,000 [0,000; 0,000] | 0,007; 12,0 | 0,0074 |
| Aleatória c/ pré-req. (P2) | 5000 | 0,258 [0,256; 0,261] | 0,194 [0,192; 0,196] | 0,185 [0,175; 0,196] | 0,667 [0,665; 0,670] | 0,000; — | 0,0025 |
| LinUCB (P3) | 5000 | 0,448 [0,443; 0,453] | 0,336 [0,332; 0,340] | 0,126 [0,116; 0,135] | 0,280 [0,273; 0,286] | 0,006; 12,1 | 0,0067 |
| Thompson linear (P3) | 5000 | 0,441 [0,436; 0,446] | 0,331 [0,327; 0,335] | 0,138 [0,128; 0,148] | 0,322 [0,316; 0,328] | 0,005; 12,1 | 0,0070 |
| DQN offline, λ por Top-1/1EP (P4a) | 5000 | 0,436 [0,430; 0,442] | 0,327 [0,322; 0,331] | 0,150 [0,141; 0,161] | 0,004 [0,003; 0,005] | 0,007; 12,0 | 0,0074 |
| DQN offline, λ por SNIS (P4a) | 5000 | 0,423 [0,418; 0,428] | 0,318 [0,313; 0,322] | 0,143 [0,134; 0,153] | 0,338 [0,333; 0,344] | 0,004; 11,2 | 0,0115 |
| DQN online (P4b) | 5000 | 0,459 [0,454; 0,464] | 0,345 [0,341; 0,349] | 0,121 [0,112; 0,130] | 0,367 [0,362; 0,372] | 0,004; 12,8 | 0,0066 |
| Guloso pelo rastreador (referência implementável) | 5000 | 0,459 [0,454; 0,464] | 0,345 [0,340; 0,349] | 0,125 [0,116; 0,134] | 0,329 [0,324; 0,332] | 0,006; 12,5 | 0,0064 |
| Oráculo míope (P6, teto) | 5000 | 0,527 [0,523; 0,531] | 0,395 [0,391; 0,399] | 0,073 [0,065; 0,080] | 0,622 [0,616; 0,629] | 0,025; 12,6 | 0,0066 |
| LinUCB, recompensa c/ revisão de crença (sens.) | 5000 | 0,449 [0,444; 0,454] | 0,336 [0,332; 0,340] | 0,126 [0,117; 0,136] | 0,277 [0,271; 0,284] | 0,006; 12,0 | 0,0070 |
| LinUCB, rastreador BKT binário (sens.) | 5000 | 0,348 [0,344; 0,352] | 0,261 [0,258; 0,265] | 0,192 [0,181; 0,203] | 0,768 [0,762; 0,774] | 0,001; 14,3 | 0,0059 |

## 4. Tamanhos de efeito vs currículo e implicação para o ensaio (A1, A3)

Diferença de Hake pareada (mesmos aprendizes) com IC 95 %; d de Cohen = diferença / dp agrupado entre aprendizes (como num ensaio com braços independentes). n por braço para poder 0,80 e α = 0,05 bilateral: n = 2(z₀,₉₇₅ + z₀,₈₀)²σ²/Δ². Duas versões: (a) Hake calculado com a probabilidade de acerto (sem erro de medida); (b) Hake num teste de 10 itens dicotômicos amostrado (instrumento do piloto), que acrescenta erro de medida. Plano A13: MDE Δg = 0,15 com n ≈ 90 por braço, o que equivale a supor σ ≈ 0,36.

| h | Política | Δ Hake [IC 95 %] | d | n/braço (a) | Δ Hake 10 itens | dp 10 itens | d 10 itens | n/braço (b) | Δ retenção | Δ abandono |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Aleatória c/ pré-req. (P2) | -0,253 [-0,256; -0,251] | -3,27 | n/a (Δ ≤ 0) | -0,284 | 0,431 | -0,66 | n/a (Δ ≤ 0) | -0,187 | 0,091 |
| 0 | LinUCB (P3) | -0,001 [-0,001; -0,001] | -0,01 | n/a (Δ ≤ 0) | -0,001 | 0,370 | -0,00 | n/a (Δ ≤ 0) | -0,001 | 0,000 |
| 0 | Thompson linear (P3) | -0,000 [-0,000; -0,000] | -0,01 | n/a (Δ ≤ 0) | -0,001 | 0,370 | -0,00 | n/a (Δ ≤ 0) | -0,000 | 0,000 |
| 0 | DQN offline, λ por Top-1/1EP (P4a) | 0,000 [0,000; 0,000] | 0,00 | n/a (Δ ≤ 0) | 0,000 | 0,370 | 0,00 | n/a (Δ ≤ 0) | 0,000 | 0,000 |
| 0 | DQN offline, λ por SNIS (P4a) | -0,012 [-0,013; -0,012] | -0,16 | n/a (Δ ≤ 0) | -0,013 | 0,371 | -0,04 | n/a (Δ ≤ 0) | -0,009 | 0,005 |
| 0 | DQN online (P4b) | -0,003 [-0,003; -0,003] | -0,04 | n/a (Δ ≤ 0) | -0,003 | 0,370 | -0,01 | n/a (Δ ≤ 0) | -0,002 | 0,000 |
| 0 | Guloso pelo rastreador (referência implementável) | 0,000 [0,000; 0,000] | 0,00 | n/a (Δ ≤ 0) | 0,000 | 0,370 | 0,00 | n/a (Δ ≤ 0) | 0,000 | 0,000 |
| 0 | Oráculo míope (P6, teto) | 0,000 [0,000; 0,000] | 0,00 | n/a (Δ ≤ 0) | 0,000 | 0,370 | 0,00 | n/a (Δ ≤ 0) | 0,000 | 0,000 |
| 0 | LinUCB, recompensa c/ revisão de crença (sens.) | -0,001 [-0,001; -0,001] | -0,01 | n/a (Δ ≤ 0) | -0,001 | 0,370 | -0,00 | n/a (Δ ≤ 0) | -0,001 | 0,000 |
| 0 | LinUCB, rastreador BKT binário (sens.) | -0,245 [-0,249; -0,241] | -2,16 | n/a (Δ ≤ 0) | -0,274 | 0,439 | -0,62 | n/a (Δ ≤ 0) | -0,181 | 0,162 |
| 0,5 | Aleatória c/ pré-req. (P2) | -0,222 [-0,225; -0,218] | -1,97 | n/a (Δ ≤ 0) | -0,251 | 0,456 | -0,55 | n/a (Δ ≤ 0) | -0,165 | 0,071 |
| 0,5 | LinUCB (P3) | -0,001 [-0,002; -0,001] | -0,01 | n/a (Δ ≤ 0) | -0,002 | 0,416 | -0,00 | n/a (Δ ≤ 0) | -0,001 | 0,001 |
| 0,5 | Thompson linear (P3) | -0,002 [-0,003; -0,002] | -0,02 | n/a (Δ ≤ 0) | -0,003 | 0,416 | -0,01 | n/a (Δ ≤ 0) | -0,002 | 0,001 |
| 0,5 | DQN offline, λ por Top-1/1EP (P4a) | -0,000 [-0,000; 0,000] | -0,00 | n/a (Δ ≤ 0) | 0,000 | 0,416 | 0,00 | n/a (Δ ≤ 0) | -0,000 | 0,000 |
| 0,5 | DQN offline, λ por SNIS (P4a) | -0,017 [-0,017; -0,016] | -0,12 | n/a (Δ ≤ 0) | -0,019 | 0,419 | -0,05 | n/a (Δ ≤ 0) | -0,012 | 0,010 |
| 0,5 | DQN online (P4b) | -0,005 [-0,006; -0,004] | -0,04 | n/a (Δ ≤ 0) | -0,005 | 0,417 | -0,01 | n/a (Δ ≤ 0) | -0,004 | 0,004 |
| 0,5 | Guloso pelo rastreador (referência implementável) | -0,004 [-0,005; -0,004] | -0,03 | n/a (Δ ≤ 0) | -0,004 | 0,417 | -0,01 | n/a (Δ ≤ 0) | -0,003 | 0,004 |
| 0,5 | Oráculo míope (P6, teto) | 0,040 [0,038; 0,042] | 0,33 | 141 | 0,046 | 0,401 | 0,11 | 1213 | 0,030 | -0,025 |
| 0,5 | LinUCB, recompensa c/ revisão de crença (sens.) | -0,002 [-0,002; -0,001] | -0,01 | n/a (Δ ≤ 0) | -0,002 | 0,416 | -0,00 | n/a (Δ ≤ 0) | -0,001 | 0,002 |
| 0,5 | LinUCB, rastreador BKT binário (sens.) | -0,186 [-0,190; -0,182] | -1,36 | n/a (Δ ≤ 0) | -0,216 | 0,464 | -0,46 | n/a (Δ ≤ 0) | -0,138 | 0,110 |
| 1 | Aleatória c/ pré-req. (P2) | -0,178 [-0,182; -0,173] | -1,14 | n/a (Δ ≤ 0) | -0,201 | 0,471 | -0,43 | n/a (Δ ≤ 0) | -0,133 | 0,035 |
| 1 | LinUCB (P3) | 0,012 [0,010; 0,014] | 0,06 | 3957 | 0,011 | 0,441 | 0,03 | > 10.000 | 0,009 | -0,025 |
| 1 | Thompson linear (P3) | 0,005 [0,004; 0,007] | 0,03 | > 10.000 | 0,005 | 0,442 | 0,01 | > 10.000 | 0,004 | -0,012 |
| 1 | DQN offline, λ por Top-1/1EP (P4a) | -0,000 [-0,000; -0,000] | -0,00 | n/a (Δ ≤ 0) | -0,000 | 0,444 | -0,00 | n/a (Δ ≤ 0) | -0,000 | 0,000 |
| 1 | DQN offline, λ por SNIS (P4a) | -0,013 [-0,014; -0,011] | -0,07 | n/a (Δ ≤ 0) | -0,015 | 0,445 | -0,03 | n/a (Δ ≤ 0) | -0,009 | -0,007 |
| 1 | DQN online (P4b) | 0,023 [0,021; 0,026] | 0,13 | 1003 | 0,023 | 0,438 | 0,05 | 5846 | 0,018 | -0,029 |
| 1 | Guloso pelo rastreador (referência implementável) | 0,023 [0,021; 0,026] | 0,12 | 1049 | 0,024 | 0,438 | 0,06 | 5116 | 0,018 | -0,025 |
| 1 | Oráculo míope (P6, teto) | 0,091 [0,088; 0,094] | 0,52 | 58 | 0,096 | 0,419 | 0,23 | 299 | 0,068 | -0,078 |
| 1 | LinUCB, recompensa c/ revisão de crença (sens.) | 0,013 [0,011; 0,014] | 0,07 | 3639 | 0,011 | 0,439 | 0,03 | > 10.000 | 0,009 | -0,024 |
| 1 | LinUCB, rastreador BKT binário (sens.) | -0,088 [-0,092; -0,084] | -0,49 | n/a (Δ ≤ 0) | -0,098 | 0,456 | -0,21 | n/a (Δ ≤ 0) | -0,066 | 0,042 |

## 5. Validação da OPE (A2)

Logs: currículo com ε = 0,2 uniforme sobre as 11 tarefas, numa população própria por semente (1000 aprendizes). Estimadores de 10/10 por transição (SNIS e DR com r̂(s,a) = MLP de 10/10), com π_b CONHECIDA. Verdadeiro = recompensa média por passo com a política implantada no simulador. Médias entre sementes; IC = média dos IC por semente (bootstrap por aprendiz).

### h = 0

| Política | Verdadeiro (rec./passo) | SNIS [IC] | DR [IC] | Hake verdadeiro | ESS | concordância com a ação logada |
|---|---|---|---|---|---|---|
| Currículo fixo (P1) | 0,321 | 0,295 [0,291; 0,298] | 0,294 [0,291; 0,298] | 0,541 | 11494 | 0,819 |
| LinUCB (P3) | 0,320 | 0,338 [0,329; 0,346] | 0,338 [0,331; 0,345] | 0,540 | 874 | 0,606 |
| Thompson linear (P3) | 0,320 | 0,343 [0,334; 0,350] | 0,344 [0,335; 0,351] | 0,540 | 867 | 0,603 |
| DQN offline, λ por Top-1/1EP (P4a) | 0,321 | 0,294 [0,289; 0,299] | 0,294 [0,290; 0,298] | 0,541 | 4666 | 0,793 |
| DQN offline, λ por SNIS (P4a) | 0,313 | 0,318 [0,304; 0,331] | 0,319 [0,309; 0,327] | 0,528 | 721 | 0,556 |
| DQN online (P4b) | 0,319 | 0,328 [0,315; 0,339] | 0,330 [0,320; 0,338] | 0,538 | 727 | 0,557 |

τ de Kendall (ordenação das 6 políticas): SNIS × verdadeiro -0,14, DR × verdadeiro -0,14; SNIS × Hake -0,14, DR × Hake -0,14; recompensa verdadeira × Hake 1,00. Erro absoluto médio: SNIS 0,018, DR 0,019. A melhor política pela SNIS NÃO é a melhor pelo Hake; pela DR NÃO é.

### h = 0,5

| Política | Verdadeiro (rec./passo) | SNIS [IC] | DR [IC] | Hake verdadeiro | ESS | concordância com a ação logada |
|---|---|---|---|---|---|---|
| Currículo fixo (P1) | 0,316 | 0,291 [0,287; 0,295] | 0,291 [0,288; 0,295] | 0,500 | 11431 | 0,819 |
| LinUCB (P3) | 0,315 | 0,335 [0,326; 0,342] | 0,334 [0,327; 0,340] | 0,499 | 980 | 0,615 |
| Thompson linear (P3) | 0,315 | 0,344 [0,337; 0,351] | 0,342 [0,337; 0,348] | 0,498 | 902 | 0,597 |
| DQN offline, λ por Top-1/1EP (P4a) | 0,316 | 0,291 [0,285; 0,296] | 0,291 [0,288; 0,295] | 0,500 | 4072 | 0,787 |
| DQN offline, λ por SNIS (P4a) | 0,308 | 0,318 [0,306; 0,330] | 0,314 [0,306; 0,321] | 0,484 | 787 | 0,564 |
| DQN online (P4b) | 0,314 | 0,328 [0,317; 0,339] | 0,325 [0,316; 0,333] | 0,495 | 795 | 0,561 |

τ de Kendall (ordenação das 6 políticas): SNIS × verdadeiro -0,20, DR × verdadeiro -0,33; SNIS × Hake -0,20, DR × Hake -0,33; recompensa verdadeira × Hake 1,00. Erro absoluto médio: SNIS 0,021, DR 0,019. A melhor política pela SNIS NÃO é a melhor pelo Hake; pela DR NÃO é.

### h = 1

| Política | Verdadeiro (rec./passo) | SNIS [IC] | DR [IC] | Hake verdadeiro | ESS | concordância com a ação logada |
|---|---|---|---|---|---|---|
| Currículo fixo (P1) | 0,246 | 0,249 [0,244; 0,254] | 0,249 [0,244; 0,254] | 0,436 | 11306 | 0,819 |
| LinUCB (P3) | 0,268 | 0,292 [0,280; 0,303] | 0,290 [0,280; 0,298] | 0,448 | 706 | 0,538 |
| Thompson linear (P3) | 0,254 | 0,290 [0,277; 0,301] | 0,289 [0,278; 0,298] | 0,441 | 736 | 0,553 |
| DQN offline, λ por Top-1/1EP (P4a) | 0,246 | 0,249 [0,241; 0,257] | 0,250 [0,243; 0,256] | 0,436 | 2573 | 0,758 |
| DQN offline, λ por SNIS (P4a) | 0,245 | 0,268 [0,255; 0,281] | 0,269 [0,259; 0,278] | 0,423 | 686 | 0,533 |
| DQN online (P4b) | 0,289 | 0,283 [0,267; 0,297] | 0,282 [0,269; 0,294] | 0,459 | 521 | 0,440 |

τ de Kendall (ordenação das 6 políticas): SNIS × verdadeiro 0,33, DR × verdadeiro 0,33; SNIS × Hake 0,33, DR × Hake 0,33; recompensa verdadeira × Hake 1,00. Erro absoluto médio: SNIS 0,016, DR 0,016. A melhor política pela SNIS NÃO é a melhor pelo Hake; pela DR NÃO é.

## 6. Seleção de λ do DQN offline

Grade λ_reg ∈ [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0]. Dois critérios de seleção na validação (logs do currículo + ε): (i) como em 10/10, Top-1 contra a ação logada e regra de um erro-padrão binomial (menor λ com Top-1 ≥ máx − EP); (ii) maior SNIS de validação. Configuração do DQN: a de 10/10 (MLP 256×2, dropout 0,1, lr 3e-3, γ 0,9, Huber, alvo a cada 5 épocas, parada antecipada pelo Top-1 de validação) com lote 256 e 40 épocas/paciência 8 (há ~15× mais transições que em 10/10).

| h | semente | transições treino/val | λ_1EP | λ (Top-1 máx.) | λ_SNIS | Top-1 val por λ | SNIS val por λ |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 27980/7011 | 1.0 | 5.0 | 0.01 | 0.0: 0,533, 0.01: 0,553, 0.05: 0,681, 0.1: 0,722, 0.25: 0,738, 0.5: 0,771, 1.0: 0,782, 5.0: 0,784 | 0.0: 0,305, 0.01: 0,318, 0.05: 0,310, 0.1: 0,302, 0.25: 0,297, 0.5: 0,297, 1.0: 0,293, 5.0: 0,293 |
| 0 | 1 | 28042/7057 | 5.0 | 5.0 | 0.01 | 0.0: 0,554, 0.01: 0,557, 0.05: 0,686, 0.1: 0,728, 0.25: 0,744, 0.5: 0,777, 1.0: 0,786, 5.0: 0,794 | 0.0: 0,310, 0.01: 0,325, 0.05: 0,317, 0.1: 0,310, 0.25: 0,309, 0.5: 0,298, 1.0: 0,300, 5.0: 0,295 |
| 0 | 2 | 28093/7018 | 1.0 | 5.0 | 0.01 | 0.0: 0,544, 0.01: 0,557, 0.05: 0,690, 0.1: 0,731, 0.25: 0,753, 0.5: 0,783, 1.0: 0,793, 5.0: 0,797 | 0.0: 0,307, 0.01: 0,328, 0.05: 0,314, 0.1: 0,301, 0.25: 0,299, 0.5: 0,286, 1.0: 0,288, 5.0: 0,288 |
| 0 | 3 | 28201/7084 | 5.0 | 5.0 | 0.0 | 0.0: 0,525, 0.01: 0,541, 0.05: 0,666, 0.1: 0,721, 0.25: 0,744, 0.5: 0,779, 1.0: 0,790, 5.0: 0,795 | 0.0: 0,327, 0.01: 0,320, 0.05: 0,304, 0.1: 0,302, 0.25: 0,292, 0.5: 0,285, 1.0: 0,290, 5.0: 0,290 |
| 0 | 4 | 28009/6858 | 1.0 | 5.0 | 0.01 | 0.0: 0,530, 0.01: 0,538, 0.05: 0,680, 0.1: 0,717, 0.25: 0,749, 0.5: 0,773, 1.0: 0,793, 5.0: 0,797 | 0.0: 0,335, 0.01: 0,343, 0.05: 0,303, 0.1: 0,297, 0.25: 0,293, 0.5: 0,289, 1.0: 0,284, 5.0: 0,286 |
| 0,5 | 0 | 28147/6986 | 1.0 | 5.0 | 0.0 | 0.0: 0,564, 0.01: 0,583, 0.05: 0,689, 0.1: 0,719, 0.25: 0,747, 0.5: 0,768, 1.0: 0,789, 5.0: 0,793 | 0.0: 0,309, 0.01: 0,307, 0.05: 0,284, 0.1: 0,295, 0.25: 0,291, 0.5: 0,289, 1.0: 0,289, 5.0: 0,288 |
| 0,5 | 1 | 27984/7044 | 1.0 | 5.0 | 0.01 | 0.0: 0,537, 0.01: 0,553, 0.05: 0,673, 0.1: 0,719, 0.25: 0,743, 0.5: 0,771, 1.0: 0,777, 5.0: 0,782 | 0.0: 0,292, 0.01: 0,324, 0.05: 0,286, 0.1: 0,291, 0.25: 0,294, 0.5: 0,287, 1.0: 0,285, 5.0: 0,287 |
| 0,5 | 2 | 27936/6971 | 1.0 | 5.0 | 0.0 | 0.0: 0,559, 0.01: 0,586, 0.05: 0,698, 0.1: 0,736, 0.25: 0,752, 0.5: 0,771, 1.0: 0,786, 5.0: 0,791 | 0.0: 0,309, 0.01: 0,306, 0.05: 0,307, 0.1: 0,295, 0.25: 0,298, 0.5: 0,296, 1.0: 0,294, 5.0: 0,291 |
| 0,5 | 3 | 28023/6966 | 5.0 | 5.0 | 0.01 | 0.0: 0,552, 0.01: 0,568, 0.05: 0,691, 0.1: 0,730, 0.25: 0,753, 0.5: 0,781, 1.0: 0,788, 5.0: 0,794 | 0.0: 0,310, 0.01: 0,319, 0.05: 0,308, 0.1: 0,308, 0.25: 0,297, 0.5: 0,302, 1.0: 0,297, 5.0: 0,297 |
| 0,5 | 4 | 28121/7057 | 1.0 | 5.0 | 0.01 | 0.0: 0,555, 0.01: 0,578, 0.05: 0,705, 0.1: 0,733, 0.25: 0,750, 0.5: 0,776, 1.0: 0,791, 5.0: 0,795 | 0.0: 0,306, 0.01: 0,318, 0.05: 0,309, 0.1: 0,300, 0.25: 0,296, 0.5: 0,292, 1.0: 0,294, 5.0: 0,292 |
| 1 | 0 | 27527/6846 | 5.0 | 5.0 | 0.0 | 0.0: 0,500, 0.01: 0,549, 0.05: 0,691, 0.1: 0,721, 0.25: 0,734, 0.5: 0,748, 1.0: 0,754, 5.0: 0,761 | 0.0: 0,272, 0.01: 0,259, 0.05: 0,245, 0.1: 0,250, 0.25: 0,248, 0.5: 0,246, 1.0: 0,253, 5.0: 0,250 |
| 1 | 1 | 27466/6877 | 5.0 | 5.0 | 0.01 | 0.0: 0,500, 0.01: 0,527, 0.05: 0,684, 0.1: 0,722, 0.25: 0,731, 0.5: 0,745, 1.0: 0,750, 5.0: 0,758 | 0.0: 0,269, 0.01: 0,276, 0.05: 0,248, 0.1: 0,250, 0.25: 0,248, 0.5: 0,247, 1.0: 0,247, 5.0: 0,250 |
| 1 | 2 | 27549/6921 | 1.0 | 5.0 | 0.01 | 0.0: 0,467, 0.01: 0,547, 0.05: 0,690, 0.1: 0,719, 0.25: 0,731, 0.5: 0,744, 1.0: 0,752, 5.0: 0,757 | 0.0: 0,273, 0.01: 0,282, 0.05: 0,251, 0.1: 0,239, 0.25: 0,231, 0.5: 0,237, 1.0: 0,239, 5.0: 0,234 |
| 1 | 3 | 27637/7003 | 5.0 | 5.0 | 0.01 | 0.0: 0,479, 0.01: 0,538, 0.05: 0,690, 0.1: 0,724, 0.25: 0,733, 0.5: 0,748, 1.0: 0,754, 5.0: 0,760 | 0.0: 0,262, 0.01: 0,276, 0.05: 0,258, 0.1: 0,249, 0.25: 0,251, 0.5: 0,250, 1.0: 0,249, 5.0: 0,250 |
| 1 | 4 | 27561/6849 | 5.0 | 5.0 | 0.01 | 0.0: 0,477, 0.01: 0,530, 0.05: 0,680, 0.1: 0,732, 0.25: 0,742, 0.5: 0,751, 1.0: 0,760, 5.0: 0,768 | 0.0: 0,280, 0.01: 0,288, 0.05: 0,264, 0.1: 0,250, 0.25: 0,249, 0.5: 0,247, 1.0: 0,248, 5.0: 0,246 |

DQN online: Double DQN, MLP 256×2, lr 0.001, γ 0.9, lote 256, replay 200000, 24000 aprendizes de treino por (h, semente) em lotes de 256, 4 atualizações por passo, ε de 1.0 a 0.05 na primeira metade, checkpoint escolhido pelo retorno médio de treino numa população de validação (não pelo Hake). Bandits: 2.000 aprendizes de treino (lotes de 100 em paralelo, atualização a cada passo), LinUCB α = 1, TS v = 0,5 (um sorteio de θ por braço e por passo do lote), ridge 1; congelados = argmax da média posterior.

## 7. Planejador LLM (P5)

**NÃO executado.** Motivo: modelo indisponível localmente: Qwen/Qwen2.5-1.5B-Instruct: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files. | Qwen/Qwen2.5-0.5B-Instruct: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.

O código (`planejador_llm.py`) está completo e a mecânica (texto do estado, memorização por prompt, escolha restrita A–K por log-probabilidade) foi verificada com um pontuador determinístico de teste (`python planejador_llm.py`); a chamada ao modelo real não pôde ser exercitada.

Prompt do sistema (exato):

```
Você é um tutor de educação financeira que planeja a trilha de um aprendiz adulto num programa por mensagens. A cada passo você escolhe UMA tarefa do catálogo. Objetivo: maximizar o aprendizado medido num pós-teste sobre todas as tarefas e evitar que o aprendiz desista. Tarefas cujos pré-requisitos o aprendiz ainda não domina rendem pouco e frustram; tarefas que ele já domina entediam. Uma tarefa pode ser repetida para revisão. Responda apenas com a letra da tarefa.
```

Exemplo de mensagem de estado (aprendiz fora das populações de avaliação):

```
Aprendiz: segmento intermediário; sentimento declarado: neutro; passo 1 de 15.
Pré-teste: 11 acertos em 22 itens.
Tarefas (letra) nome [missão] | pré-requisito | vezes feita | domínio estimado:
(A) Introducao ao Raio-X [M1] | nenhum | 0 | baixo
(B) Custos Fixos [M1] | Introducao ao Raio-X | 0 | baixo
(C) Gastos Variaveis [M1] | Custos Fixos | 0 | baixo
(D) Informar Renda [M1] | Gastos Variaveis | 0 | baixo
(E) Reflexao do Resultado [M1] | Informar Renda | 0 | baixo
(F) Introducao Compra Inteligente [M2] | Reflexao do Resultado | 0 | baixo
(G) Decisao de Compra [M2] | Introducao Compra Inteligente | 0 | baixo
(H) Introducao Plano de Acao [M3] | Decisao de Compra | 0 | baixo
(I) Definir Objetivo [M3] | Introducao Plano de Acao | 0 | baixo
(J) Introducao Pesquisa de Precos [M6] | Definir Objetivo | 0 | baixo
(K) Pesquisa de Precos na Pratica [M6] | Introducao Pesquisa de Precos | 0 | baixo
Últimas respostas: nenhuma (início do programa).
Qual é a próxima tarefa? Responda só com a letra.
```

Few-shot: os três pares (estado, letra do currículo) abaixo entram como turnos user/assistant antes do estado:

```
Aprendiz: segmento intermediário; sentimento declarado: neutro; passo 1 de 15.
Pré-teste: 8 acertos em 22 itens.
Tarefas (letra) nome [missão] | pré-requisito | vezes feita | domínio estimado:
(A) Introducao ao Raio-X [M1] | nenhum | 0 | baixo
(B) Custos Fixos [M1] | Introducao ao Raio-X | 0 | baixo
(C) Gastos Variaveis [M1] | Custos Fixos | 0 | baixo
(D) Informar Renda [M1] | Gastos Variaveis | 0 | baixo
(E) Reflexao do Resultado [M1] | Informar Renda | 0 | baixo
(F) Introducao Compra Inteligente [M2] | Reflexao do Resultado | 0 | baixo
(G) Decisao de Compra [M2] | Introducao Compra Inteligente | 0 | baixo
(H) Introducao Plano de Acao [M3] | Decisao de Compra | 0 | baixo
(I) Definir Objetivo [M3] | Introducao Plano de Acao | 0 | baixo
(J) Introducao Pesquisa de Precos [M6] | Definir Objetivo | 0 | baixo
(K) Pesquisa de Precos na Pratica [M6] | Introducao Pesquisa de Precos | 0 | baixo
Últimas respostas: nenhuma (início do programa).
Qual é a próxima tarefa? Responda só com a letra.
→ A
```

```
Aprendiz: segmento intermediário; sentimento declarado: motivado; passo 6 de 15.
Pré-teste: 8 acertos em 22 itens.
Tarefas (letra) nome [missão] | pré-requisito | vezes feita | domínio estimado:
(A) Introducao ao Raio-X [M1] | nenhum | 1 | alto
(B) Custos Fixos [M1] | Introducao ao Raio-X | 1 | alto
(C) Gastos Variaveis [M1] | Custos Fixos | 1 | alto
(D) Informar Renda [M1] | Gastos Variaveis | 1 | alto
(E) Reflexao do Resultado [M1] | Informar Renda | 1 | alto
(F) Introducao Compra Inteligente [M2] | Reflexao do Resultado | 0 | baixo
(G) Decisao de Compra [M2] | Introducao Compra Inteligente | 0 | baixo
(H) Introducao Plano de Acao [M3] | Decisao de Compra | 0 | baixo
(I) Definir Objetivo [M3] | Introducao Plano de Acao | 0 | baixo
(J) Introducao Pesquisa de Precos [M6] | Definir Objetivo | 0 | baixo
(K) Pesquisa de Precos na Pratica [M6] | Introducao Pesquisa de Precos | 0 | baixo
Últimas respostas: Gastos Variaveis (acertou); Informar Renda (errou); Reflexao do Resultado (acertou).
Qual é a próxima tarefa? Responda só com a letra.
→ F
```

```
Aprendiz: segmento intermediário; sentimento declarado: neutro; passo 12 de 15.
Pré-teste: 8 acertos em 22 itens.
Tarefas (letra) nome [missão] | pré-requisito | vezes feita | domínio estimado:
(A) Introducao ao Raio-X [M1] | nenhum | 1 | alto
(B) Custos Fixos [M1] | Introducao ao Raio-X | 1 | alto
(C) Gastos Variaveis [M1] | Custos Fixos | 1 | alto
(D) Informar Renda [M1] | Gastos Variaveis | 1 | alto
(E) Reflexao do Resultado [M1] | Informar Renda | 1 | alto
(F) Introducao Compra Inteligente [M2] | Reflexao do Resultado | 1 | alto
(G) Decisao de Compra [M2] | Introducao Compra Inteligente | 1 | alto
(H) Introducao Plano de Acao [M3] | Decisao de Compra | 1 | alto
(I) Definir Objetivo [M3] | Introducao Plano de Acao | 1 | alto
(J) Introducao Pesquisa de Precos [M6] | Definir Objetivo | 1 | alto
(K) Pesquisa de Precos na Pratica [M6] | Introducao Pesquisa de Precos | 1 | alto
Últimas respostas: Definir Objetivo (errou); Introducao Pesquisa de Precos (acertou); Pesquisa de Precos na Pratica (errou).
Qual é a próxima tarefa? Responda só com a letra.
→ A
```

## 8. Interpretação, decisões e pendências

Texto escrito após a execução completa de 11/10/2026 (17,5 min); os números citados vêm das tabelas acima. Se o código for reexecutado com outra configuração, este texto precisa ser revisto.

### 8.1 O que os resultados mostram

1. **O parâmetro h funciona como previsto.** Em h = 0 o oráculo míope coincide com o currículo (divergência 0 %, Hake 0,541 nos dois), e nenhuma política adaptativa o supera: as diferenças vão de 0,000 a −0,012. Em h = 1 o teto do oráculo sobe para +0,091 de Hake (d = 0,52) e o abandono cai de 15,0 % para 7,3 %. O espaço para adaptação existe e cresce com a heterogeneidade.
2. **Em h = 0,5 (o cenário calibrado) nenhuma política implementável supera o currículo.** O oráculo ganha +0,040, mas todas as políticas que só veem o estado observável ficam entre −0,017 e −0,001. O rastreador guloso também perde (−0,004): o limite é a informação disponível, e não só o aprendizado da política.
3. **Em h = 1 o ganho das melhores políticas implementáveis é pequeno.** DQN online e o rastreador guloso ganham +0,023 [0,021; 0,026] (d ≈ 0,12–0,13). O LinUCB ganha +0,012 (d = 0,06) e o Thompson +0,005. O DQN online, com interação livre, só alcança a referência gulosa. Cerca de 75 % do teto do oráculo se perde por observabilidade parcial (0,091 contra 0,023).
4. **O DQN offline com logs do currículo + ε não supera o currículo em nenhum h.** Selecionado como em 10/10 (Top-1 com regra de 1 EP), λ fica em 1 ou 5 e a política imita o currículo (divergência ≤ 0,4 %, diferença ≈ 0). Selecionado por SNIS de validação, λ fica em 0 ou 0,01, a política diverge 22–34 % e perde de −0,012 a −0,017. É o mesmo padrão de 10/10, agora com um desfecho verdadeiro para conferir.
5. **A OPE por transição não ordena as políticas de forma confiável neste domínio.** O τ de Kendall entre SNIS/DR e o valor verdadeiro é −0,14 (h = 0), −0,20/−0,33 (h = 0,5) e 0,33 (h = 1). Nos três níveis a política de maior valor estimado (Thompson ou LinUCB) não é a de maior Hake. O viés tem direção quase sempre previsível. Os bandits e o DQN offline λ_SNIS são superestimados em +0,02 a +0,04 de recompensa por passo. O próprio currículo é subestimado em −0,025 em h = 0 e h = 0,5. A exceção é o DQN online em h = 1 (−0,006). A causa é que o estimador por transição avalia a ação nos estados visitados pelos logs do currículo + ε, e não nos que a política visitaria. O viés ocorre mesmo com π_b conhecida. O erro absoluto médio (0,016–0,021) é da mesma ordem das diferenças verdadeiras entre políticas (≤ 0,04).
6. **A recompensa de treino é um substituto imperfeito do Hake.** Em h = 1 o oráculo tem o maior Hake (0,527) e o menor retorno de treino (3,10, contra 3,43 do currículo). A ordenação pela recompensa verdadeira por passo coincide com a do Hake entre as políticas avaliadas por OPE (τ = 1), mas não entre todas.
7. **O rastreador de conhecimento é decisivo.** Com o BKT binário clássico (variante de sensibilidade) o LinUCB perde de −0,09 a −0,25 de Hake e abandona mais. O BKT binário é superconfiante num simulador de maestria contínua, e a política explora esse ruído. Com o rastreador contínuo, a variante de recompensa com revisão de crença não muda o resultado (diferença ≤ 0,001).

### 8.2 Implicação para o ensaio (A3)

O dp do Hake num teste de 10 itens simulado (0,37–0,44) é próximo do σ ≈ 0,36 implícito no plano (Δg = 0,15, n ≈ 90). O plano é coerente quanto à variância, mas a diferença simulada entre política adaptativa e currículo é 6 a 13 vezes menor que o MDE:

- h = 0,5: nenhuma política implementável tem efeito positivo. Não há n que detecte um efeito que o simulador não produz. Mesmo o oráculo (Δ = 0,046 no teste de 10 itens) exigiria cerca de 1.200 por braço.
- h = 1: DQN online e guloso, Δ ≈ 0,023: cerca de 1.000 por braço sem erro de medida e cerca de 5.100–5.800 com o teste de 10 itens. LinUCB, Δ = 0,012: cerca de 4.000 e mais de 10.000. O oráculo (Δ = 0,096 no teste de 10 itens) exigiria cerca de 300.
- Conclusão: sob este simulador, o plano A13 está **subdimensionado** para o contraste adaptativo × currículo em todos os cenários: por um fator de cerca de 60 em h = 1 e sem efeito a detectar em h = 0,5. O MDE de 0,15 só é plausível para contrastes de maior magnitude, como o de modalidade (tutor × cartilha, Δg = 0,30 no piloto), ou se o efeito real da adaptação for muito maior que o simulado. O simulador não estima esse efeito real; ele mostra que o valor do plano não está garantido pelas premissas atuais.

### 8.3 Decisões tomadas sem consulta (e por quê)

- **Catálogo sem bloqueio de ações.** Os pré-requisitos do arquivo formam uma cadeia linear. Se bloqueassem ações, toda política seria idêntica ao currículo ou repetiria tarefas. Os pré-requisitos agem pelo modelo cognitivo (P(T) × 0,3 e frustração). A política aleatória (P2) respeita os pré-requisitos concluídos, como pede a especificação.
- **Currículo com revisão.** Depois das 11 tarefas, o currículo revisa em ordem (1–4), para que todas as políticas tenham o mesmo orçamento de H = 15.
- **Maestria contínua e esquecimento até L0.** As razões estão no cabeçalho de `simulador.py`.
- **Rastreador contínuo (Beta) em vez do BKT binário.** A primeira versão usou o BKT binário. LinUCB e DQN aprenderam a explorar o ruído da estimativa: retorno de treino maior que o do currículo e Hake de 0,35 contra 0,44 em h = 1. O rastreador foi trocado por um coerente com o simulador. A versão binária ficou como sensibilidade e consta das tabelas.
- **Recompensa sem a revisão de crença.** A mudança de crença causada pela resposta é informação sobre o conhecimento prévio, e não aprendizado. A variante com o termo consta das tabelas e não muda o resultado.
- **Referência gulosa sobre o rastreador.** Não foi pedida. Foi incluída porque separa o limite de informação do limite de aprendizado da política, o que responde diretamente à pergunta científica.
- **DQN offline: lote 256 e 40 épocas** (em vez de 32 e 80), porque há cerca de 28 mil transições por treino. **Segundo critério de λ (SNIS).** Foi acrescentado para mostrar que o critério de 10/10 leva à imitação do currículo.
- **DQN online: Double DQN** e checkpoint escolhido pelo retorno de treino em validação, e não pelo Hake.
- **Bandits congelados = argmax da média.** No treino, o TS sorteia θ uma vez por passo do lote, por custo. A amostragem usa o fator de Cholesky de A, porque a decomposição de A⁻¹ falhou numericamente.
- **Pós-teste no relógio H para todos** (intenção de tratar). Quem abandona sofre esquecimento até H. O pré-teste observado pelo agente tem 2 itens por habilidade. O Hake de desfecho usa a probabilidade de acerto, como pede S5. Um teste de 10 itens amostrado foi acrescentado só para A3.
- **Calibração.** c_T = 0,6586 dá Hake de 0,500 em h = 0,5. α = 0,0108 dá abandono de 15,0 % em h = 1. Os dois alvos são ajustados por bisseções alternadas. O alvo de abandono (15 %) é escolha de desenho, e não dado.

### 8.4 Pendências

- **Planejador LLM (P5) não executado.** O ambiente bloqueou o acesso de rede ao Hugging Face (WinError 10013), e não há cópia local do Qwen2.5-1.5B nem do 0.5B. O código, o prompt e os exemplos few-shot estão prontos. Para executar: baixar o modelo uma vez (`huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct`) numa máquina com rede e rodar `python run_all.py`. O carregador tenta o cache local antes da rede, estima o custo com 16 prompts e reduz a população se a projeção passar de 90 min, registrando a redução. A memorização por prompt quase não reduz chamadas, porque o texto inclui passo e histórico. Sem o modelo, não há afirmação possível sobre o planejador LLM.
- A OPE do planejador LLM exigiria reconstruir o texto do estado a partir dos logs. Não foi implementada.
- Os parâmetros do simulador, exceto c_T, são plausíveis e não estimados. A sensibilidade ao multiplicador e ao limiar de pré-requisito e à dispersão está na seção 9. O "efeito pequeno" se mantém nas 27 combinações; a fração de 75 % do teto perdida não se mantém. A sensibilidade a α, S, G e F e às médias de L0 não foi testada.
- Planejador LLM: a decisão posterior do autor é executá-lo como subagente por rodadas de arquivos (`planejador_rodadas.py`; pastas `rodadas_h05/` e `rodadas_h1/`, 200 aprendizes, semente 42, pareados com o currículo). Os resultados, quando existirem, ficam em `rodadas_h*/resultado_llm_h*.json`, fora das tabelas acima.

## 9. Análise de sensibilidade (parâmetros não calibrados, h = 1)

Grade: multiplicador de P(T) com pré-requisito não dominado [0.2, 0.3, 0.5], limiar de pré-requisito [0.5, 0.6, 0.7], escala de todos os dp de heterogeneidade [0.75, 1.0, 1.25] (27 combinações). c_T recalibrado por bisseção em CADA combinação para Hake do currículo = 0,50 em h = 0,5; α fixo em 0,0108, por isso o abandono do currículo em h = 1 varia entre combinações. Em h = 1, 3 sementes × 1000 aprendizes, com as sementes e populações da execução principal; a combinação 0,3/0,6/1 corresponde às sementes 0–2 da tabela principal. Δg = diferença pareada vs currículo [IC 95 %, bootstrap de aprendizes]. Tempo: 16,9 min. Figura: fig_sim_sensibilidade.pdf. Reprodução: `python sensibilidade.py` depois de `python run_all.py`.

| mult. | limiar | escala | c_T | Hake currículo | aband. currículo | Δg LinUCB | Δg DQN online | Δg guloso | Δg oráculo | teto perdido (DQN on.) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0,2 | 0,5 | 0,75 | 0,607 | 0,431 | 0,118 | -0,001 [-0,002; -0,001] | -0,010 [-0,011; -0,008] | -0,009 | 0,067 | 115 % |
| 0,2 | 0,5 | 1 | 0,616 | 0,435 | 0,133 | -0,007 [-0,010; -0,005] | -0,000 [-0,004; 0,004] | -0,025 | 0,075 | 100 % |
| 0,2 | 0,5 | 1,25 | 0,623 | 0,430 | 0,142 | -0,021 [-0,024; -0,017] | -0,018 [-0,022; -0,014] | -0,022 | 0,087 | 120 % |
| 0,2 | 0,6 | 0,75 | 0,649 | 0,417 | 0,145 | 0,030 [0,028; 0,032] | 0,048 [0,045; 0,052] | 0,033 | 0,098 | 51 % |
| 0,2 | 0,6 | 1 | 0,663 | 0,426 | 0,145 | 0,018 [0,016; 0,021] | 0,025 [0,021; 0,029] | 0,023 | 0,104 | 76 % |
| 0,2 | 0,6 | 1,25 | 0,673 | 0,433 | 0,153 | 0,009 [0,006; 0,011] | 0,018 [0,014; 0,022] | 0,024 | 0,109 | 83 % |
| 0,2 | 0,7 | 0,75 | 0,729 | 0,428 | 0,159 | 0,043 [0,040; 0,045] | 0,060 [0,056; 0,064] | 0,064 | 0,122 | 51 % |
| 0,2 | 0,7 | 1 | 0,738 | 0,436 | 0,165 | 0,025 [0,022; 0,028] | 0,035 [0,031; 0,039] | 0,036 | 0,129 | 73 % |
| 0,2 | 0,7 | 1,25 | 0,747 | 0,436 | 0,173 | 0,014 [0,011; 0,017] | 0,028 [0,023; 0,032] | 0,033 | 0,135 | 79 % |
| 0,3 | 0,5 | 0,75 | 0,606 | 0,439 | 0,117 | -0,001 [-0,001; -0,001] | -0,007 [-0,008; -0,006] | -0,007 | 0,058 | 112 % |
| 0,3 | 0,5 | 1 | 0,615 | 0,441 | 0,132 | -0,011 [-0,014; -0,008] | -0,001 [-0,004; 0,002] | -0,020 | 0,068 | 101 % |
| 0,3 | 0,5 | 1,25 | 0,621 | 0,439 | 0,136 | -0,018 [-0,021; -0,015] | -0,009 [-0,012; -0,005] | -0,015 | 0,077 | 111 % |
| 0,3 | 0,6 | 0,75 | 0,647 | 0,428 | 0,145 | 0,020 [0,018; 0,022] | 0,040 [0,037; 0,043] | 0,030 | 0,086 | 53 % |
| 0,3 | 0,6 | 1 | 0,659 | 0,437 | 0,150 | 0,013 [0,010; 0,015] | 0,021 [0,018; 0,025] | 0,023 | 0,090 | 76 % |
| 0,3 | 0,6 | 1,25 | 0,667 | 0,438 | 0,159 | 0,003 [0,000; 0,006] | 0,019 [0,015; 0,022] | 0,022 | 0,099 | 81 % |
| 0,3 | 0,7 | 0,75 | 0,721 | 0,438 | 0,163 | 0,026 [0,024; 0,028] | 0,045 [0,041; 0,048] | 0,053 | 0,103 | 57 % |
| 0,3 | 0,7 | 1 | 0,730 | 0,445 | 0,169 | 0,013 [0,010; 0,015] | 0,022 [0,018; 0,026] | 0,029 | 0,103 | 79 % |
| 0,3 | 0,7 | 1,25 | 0,738 | 0,445 | 0,172 | 0,005 [0,002; 0,007] | 0,019 [0,016; 0,023] | 0,025 | 0,104 | 82 % |
| 0,5 | 0,5 | 0,75 | 0,606 | 0,453 | 0,115 | -0,001 [-0,001; -0,000] | -0,007 [-0,009; -0,006] | -0,003 | 0,038 | 119 % |
| 0,5 | 0,5 | 1 | 0,612 | 0,454 | 0,129 | -0,014 [-0,016; -0,011] | -0,005 [-0,008; -0,003] | -0,008 | 0,044 | 112 % |
| 0,5 | 0,5 | 1,25 | 0,616 | 0,449 | 0,139 | -0,010 [-0,012; -0,007] | -0,004 [-0,007; -0,001] | -0,003 | 0,052 | 107 % |
| 0,5 | 0,6 | 0,75 | 0,640 | 0,447 | 0,141 | 0,004 [0,002; 0,005] | 0,022 [0,020; 0,025] | 0,018 | 0,037 | 40 % |
| 0,5 | 0,6 | 1 | 0,649 | 0,452 | 0,149 | -0,000 [-0,002; 0,002] | 0,012 [0,009; 0,014] | 0,006 | 0,040 | 71 % |
| 0,5 | 0,6 | 1,25 | 0,654 | 0,450 | 0,159 | -0,008 [-0,011; -0,005] | 0,010 [0,007; 0,013] | 0,009 | 0,049 | 80 % |
| 0,5 | 0,7 | 0,75 | 0,703 | 0,460 | 0,160 | 0,002 [0,001; 0,003] | 0,020 [0,018; 0,022] | 0,001 | 0,019 | -7 % |
| 0,5 | 0,7 | 1 | 0,710 | 0,461 | 0,169 | -0,002 [-0,004; -0,000] | 0,010 [0,008; 0,013] | -0,008 | 0,022 | 53 % |
| 0,5 | 0,7 | 1,25 | 0,716 | 0,465 | 0,173 | -0,008 [-0,011; -0,005] | 0,008 [0,005; 0,010] | -0,003 | 0,029 | 73 % |

Faixas de Δg nas 27 combinações: LinUCB -0,021 a 0,043; DQN online -0,018 a 0,060; guloso -0,025 a 0,064; oráculo 0,019 a 0,135. Fração do teto do oráculo perdida pelo DQN online: -7 % a 120 %; pela melhor política implementável da combinação: -7 % a 120 %.

**Interpretação** (escrita após a execução de 11/10/2026; os números vêm da tabela acima).

- **O limiar de pré-requisito é o fator dominante.** Com limiar 0,5, nenhuma política implementável supera o currículo em h = 1 (Δg de −0,025 a 0,000). O oráculo ainda ganha de +0,038 a +0,087. Com limiar 0,6 ou 0,7, o DQN online ganha de +0,008 a +0,060, e o IC exclui zero nas 18 combinações com limiar ≥ 0,6. O multiplicador afeta sobretudo o teto: com multiplicador 0,5, o Δg do oráculo cai para 0,019–0,052.
- **"Efeito pequeno" se mantém nas 27 combinações.** O maior Δg implementável é +0,064 (guloso) e +0,060 (DQN online), com d ≤ 0,31. Isso fica abaixo do MDE de 0,15 em todas as combinações. O maior Δg do oráculo é +0,135 (d = 0,66), e também fica abaixo de 0,15.
- **"Cerca de 75 % do teto perdido" NÃO é robusto.** O número depende da escala de dispersão:
  - escala 1 ou 1,25 com limiar ≥ 0,6: a melhor política implementável perde 53–80 % do teto;
  - escala 0,75 com limiar ≥ 0,6: perde 40–53 %, e numa combinação (0,5/0,7/0,75) iguala o oráculo, cujo teto ali é só +0,019;
  - limiar 0,5: perde ≥ 100 %, porque nenhuma política implementável ganha algo.
  - Formulação sustentável: a maior parte do teto se perde por observabilidade parcial (≥ 50 % em 23 das 27 combinações). A proporção exata depende de parâmetros não calibrados.
- **Mais dispersão reduz o ganho implementável.** Com limiar ≥ 0,6, passar a escala de 0,75 para 1,25 aumenta o teto do oráculo e reduz o Δg do DQN online. Com mais dispersão, o rastreador, que usa parâmetros médios do segmento, erra mais sobre cada aprendiz.
- **Implicação para A3.** Mesmo no cenário mais favorável (Δg = 0,06, dp do teste de 10 itens ≈ 0,44), seriam necessários cerca de 850 aprendizes por braço para poder 0,80. O diagnóstico de subdimensionamento do plano (n ≈ 90) não depende dos três parâmetros testados.


## 10. Planejador LLM por subagente (executado em 11/10/2026)

O modelo local (Qwen) não pôde ser baixado (seção 7). Por decisão do autor, o planejador foi executado como subagente Claude, uma rodada por vez, com `planejador_rodadas.py` (init/step/finalize) e `rodada_para_texto.py`. Cada rodada foi decidida por um subagente novo, que leu apenas `INSTRUCOES_PLANEJADOR.md` e a visão em texto da rodada; nenhum teve acesso ao simulador, ao código ou à maestria verdadeira. Na rodada 0 de h = 1, o subagente usou um script que só lia o arquivo da rodada para resumi-lo; a partir da rodada 1, a visão em texto tornou o script desnecessário e foi proibido. Nenhuma das 30 rodadas teve decisão inválida.

200 aprendizes por cenário (semente 42), pareados com o currículo nos mesmos aprendizes e números aleatórios.

| Cenário | Hake LLM [IC 95 %] | Hake currículo | Δ Hake [IC 95 %] | d | Δ retenção [IC 95 %] | Abandono LLM / currículo | Divergência |
|---|---|---|---|---|---|---|---|
| h = 0,5 | 0,487 [0,471; 0,503] | 0,496 | −0,008 [−0,018; 0,001] | −0,07 | −0,006 [−0,013; 0,001] | 7,5 % / 8,0 % | 34,2 % |
| h = 1 | 0,452 [0,430; 0,474] | 0,435 | +0,017 [0,003; 0,030] | 0,10 | +0,014 [0,004; 0,025] | 11,0 % / 13,5 % | 42,8 % |

Arquivos: `rodadas_h05/resultado_llm_h05.json`, `rodadas_h1/resultado_llm_h1.json`, decisões em `rodadas_h*/decisoes_*.json`.

Leitura: sem treino, o planejador LLM acompanha as melhores políticas implementáveis (DQN online +0,023 e guloso +0,023 em h = 1, com 5 × 1.000 aprendizes). O resultado vale para um modelo de fronteira (Claude); não indica que um modelo de pesos abertos menor obtenha o mesmo.

# Resultados da reexecução dos experimentos de DRL — 2026-10-10

Ambiente: Python 3.12.10, torch 2.14.1+cpu (CPU), numpy 2.5.3. Sementes (split fixo): [42, 43, 44, 45, 46]; semente da validação cruzada: 42. Bootstrap: 1000 reamostras por transição. Tempo total: 7.4 min (preparação 0 s; split fixo 3.0 min; 5-fold 3.8 min; cultural 0.6 min).

Reprodução: `python run_all.py` nesta pasta (regenera resultados.json, este arquivo e as três figuras).

## 1. Partições por perfil (E2)

Os 25 perfis foram divididos em cinco grupos disjuntos, estratificados por segmento de `users.jsonl` (rodízio após embaralhar com semente 42). Split fixo: teste = grupo 0, validação = grupo 1, treino = grupos 2–4 (5/5/15 perfis). Validação cruzada: fold k usa teste = grupo k, validação = grupo k+1 (parada antecipada e seleção de λ), treino = os três grupos restantes.

- grupo 0: u_002 (int), u_004 (ava), u_006 (ini), u_007 (ini), u_008 (int)
- grupo 1: u_012 (int), u_014 (ini), u_016 (ini), u_017 (int), u_020 (ava)
- grupo 2: u_001 (ini), u_003 (int), u_015 (ava), u_018 (ini), u_024 (int)
- grupo 3: u_005 (ini), u_010 (ava), u_011 (ini), u_013 (int), u_023 (int)
- grupo 4: u_009 (int), u_019 (int), u_021 (ini), u_022 (ava), u_025 (int)

Split fixo — treino: u_001, u_003, u_005, u_009, u_010, u_011, u_013, u_015, u_018, u_019, u_021, u_022, u_023, u_024, u_025; validação: u_012, u_014, u_016, u_017, u_020; teste: u_002, u_004, u_006, u_007, u_008. Transições: {'train': 1207, 'val': 418, 'test': 375} (partição original por interação: {'train': 1581, 'val': 206, 'test': 213}).

## 2. Tabela principal — validação cruzada 5-fold por perfil (semente 42)

Média ± dp entre folds (n = 5 folds); IC 95 % por bootstrap das transições fora-da-amostra agrupadas (N = 2000). λ* reselecionado na validação de cada fold: fold 0: 2.0, fold 1: 0.5, fold 2: 1.0, fold 3: 1.0, fold 4: 1.0; λ_1EP por fold: fold 0: 0.25, fold 1: 0.5, fold 2: 0.25, fold 3: 0.25, fold 4: 0.25.

| Política | n | Top-1 | Top-3 | Rec. associada | Teto | % teto | SNIS [IC 95 %] | DR [IC 95 %] | DR-Q(CQL) | ESS |
|---|---|---|---|---|---|---|---|---|---|---|
| Regra sequencial (próxima do catálogo) | 5 | 0,648 ± 0,034 | 0,648 ± 0,034 | 0,701 ± 0,010 | 0,704 ± 0,009 | 99,5 ± 0,1 % | 0,699 ± 0,011 [0,694; 0,703] | 0,683 ± 0,008 [0,679; 0,687] | 0,59 ± 0,07 | 232 |
| Regra sequencial + repetir terminal (variante) | 5 | 0,776 ± 0,031 | 0,776 ± 0,031 | 0,627 ± 0,012 | 0,633 ± 0,011 | 99,1 ± 0,1 % | 0,625 ± 0,013 [0,614; 0,635] | 0,622 ± 0,012 [0,613; 0,630] | 0,55 ± 0,06 | 280 |
| BC | 5 | 0,841 ± 0,019 | 0,997 ± 0,002 | 0,573 ± 0,023 | 0,595 ± 0,020 | 96,4 ± 1,9 % | 0,553 ± 0,029 [0,539; 0,566] | 0,544 ± 0,029 [0,533; 0,556] | 0,55 ± 0,03 | 315 |
| DQN λ = 0 | 5 | 0,685 ± 0,014 | 0,723 ± 0,046 | 0,683 ± 0,013 | 0,691 ± 0,014 | 98,8 ± 0,7 % | 0,677 ± 0,013 [0,669; 0,685] | 0,632 ± 0,013 [0,624; 0,639] | -0,01 ± 0,22 | 245 |
| DQN λ* (Top-1 máx.) | 5 | 0,845 ± 0,020 | 0,997 ± 0,002 | 0,576 ± 0,022 | 0,593 ± 0,017 | 97,1 ± 1,7 % | 0,552 ± 0,026 [0,538; 0,566] | 0,554 ± 0,028 [0,544; 0,565] | 0,56 ± 0,03 | 308 |
| DQN λ_1EP | 5 | 0,841 ± 0,021 | 0,995 ± 0,006 | 0,582 ± 0,020 | 0,595 ± 0,016 | 97,8 ± 2,2 % | 0,555 ± 0,025 [0,541; 0,570] | 0,560 ± 0,028 [0,549; 0,571] | 0,56 ± 0,03 | 291 |
| CQL | 5 | 0,775 ± 0,034 | 0,998 ± 0,001 | 0,627 ± 0,014 | 0,634 ± 0,012 | 98,9 ± 0,4 % | 0,614 ± 0,017 [0,599; 0,626] | 0,603 ± 0,017 [0,593; 0,612] | 0,58 ± 0,02 | 263 |
| PPO offline | 5 | 0,234 ± 0,036 | 0,356 ± 0,021 | 0,668 ± 0,060 | 0,766 ± 0,004 | 87,1 ± 7,6 % | 0,539 ± 0,174 [0,387; 0,644] | 0,391 ± 0,062 [0,350; 0,416] | -2,43 ± 0,89 | 54 |
| DQN λ* (λ do split fixo) | 5 | 0,841 ± 0,018 | 0,996 ± 0,003 | 0,579 ± 0,019 | 0,594 ± 0,015 | 97,4 ± 1,4 % | 0,553 ± 0,020 [0,538; 0,566] | 0,557 ± 0,027 [0,546; 0,569] | 0,56 ± 0,03 | 300 |

Valores agrupados (OOF) de Top-1 / recompensa associada / SNIS / DR: Regra sequencial (próxima do catálogo): 0,648 / 0,701 / 0,699 / 0,683; Regra sequencial + repetir terminal (variante): 0,777 / 0,627 / 0,625 / 0,622; BC: 0,841 / 0,574 / 0,553 / 0,545; DQN λ = 0: 0,685 / 0,683 / 0,677 / 0,632; DQN λ* (Top-1 máx.): 0,845 / 0,576 / 0,552 / 0,554; DQN λ_1EP: 0,841 / 0,582 / 0,555 / 0,560; CQL: 0,775 / 0,627 / 0,614 / 0,603; PPO offline: 0,234 / 0,675 / 0,499 / 0,383.

## 3. Split fixo por perfil (5 sementes)

Média ± dp entre sementes (n = 5 execuções; a regra sequencial é determinística, dp = 0); IC 95 % por bootstrap das 375 transições de teste, com a estimativa média entre sementes. λ* = 5.0 (maior Top-1 médio de validação; desempate pelo menor λ). π_b por contagem em 72,5 % das transições de teste (BC nas demais).

| Política | n | Top-1 | Top-3 | Rec. associada | Teto | % teto | SNIS [IC 95 %] | DR [IC 95 %] | DR-Q(CQL) | ESS |
|---|---|---|---|---|---|---|---|---|---|---|
| Regra sequencial (próxima do catálogo) | 5 | 0,621 ± 0,000 | 0,621 ± 0,000 | 0,708 ± 0,000 | 0,710 ± 0,000 | 99,6 ± 0,0 % | 0,706 ± 0,001 [0,696; 0,714] | 0,682 ± 0,001 [0,673; 0,691] | 0,60 ± 0,01 | 216 |
| Regra sequencial + repetir terminal (variante) | 5 | 0,731 ± 0,000 | 0,731 ± 0,000 | 0,637 ± 0,000 | 0,643 ± 0,000 | 99,0 ± 0,0 % | 0,639 ± 0,000 [0,614; 0,662] | 0,630 ± 0,001 [0,610; 0,647] | 0,57 ± 0,01 | 256 |
| BC | 5 | 0,816 ± 0,005 | 0,998 ± 0,001 | 0,563 ± 0,005 | 0,590 ± 0,003 | 95,4 ± 0,4 % | 0,530 ± 0,008 [0,500; 0,561] | 0,532 ± 0,005 [0,505; 0,560] | 0,54 ± 0,00 | 279 |
| DQN λ = 0 | 5 | 0,673 ± 0,010 | 0,683 ± 0,014 | 0,681 ± 0,008 | 0,685 ± 0,007 | 99,4 ± 0,2 % | 0,680 ± 0,009 [0,665; 0,695] | 0,637 ± 0,008 [0,619; 0,653] | 0,19 ± 0,08 | 233 |
| DQN λ* (Top-1 máx.) | 5 | 0,822 ± 0,005 | 0,995 ± 0,003 | 0,570 ± 0,004 | 0,587 ± 0,003 | 97,1 ± 1,0 % | 0,539 ± 0,010 [0,510; 0,570] | 0,547 ± 0,007 [0,522; 0,574] | 0,55 ± 0,01 | 274 |
| DQN λ_1EP | 5 | 0,822 ± 0,005 | 0,995 ± 0,003 | 0,570 ± 0,004 | 0,587 ± 0,003 | 97,1 ± 1,0 % | 0,539 ± 0,010 [0,510; 0,570] | 0,547 ± 0,007 [0,522; 0,574] | 0,55 ± 0,01 | 274 |
| CQL | 5 | 0,746 ± 0,012 | 0,997 ± 0,000 | 0,627 ± 0,008 | 0,633 ± 0,008 | 99,0 ± 0,1 % | 0,617 ± 0,007 [0,593; 0,641] | 0,608 ± 0,008 [0,588; 0,629] | 0,58 ± 0,01 | 254 |
| PPO offline | 5 | 0,229 ± 0,051 | 0,373 ± 0,043 | 0,661 ± 0,030 | 0,772 ± 0,008 | 85,6 ± 3,3 % | 0,535 ± 0,129 [0,485; 0,625] | 0,407 ± 0,027 [0,387; 0,432] | -3,05 ± 1,56 | 40 |

## 4. Seleção de λ — grade de λ_reg (split fixo, 5 sementes)

Grade [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0] (estendida com 2, 5 e 10). λ* (Top-1 máximo de validação, desempate pelo menor λ) = 5.0. Regra de um erro-padrão: EP = dp entre sementes / √5 do λ de máximo = 0,0016; menor λ com Top-1 de validação ≥ máximo − EP: **λ_1EP = 5.0**. No 5-fold (uma execução por fold) o EP é o binomial √(p(1−p)/n_val) do máximo. Marcas: * = λ*, † = λ_1EP.

Com o EP binomial também no split fixo (EP = 0,0160), λ_1EP = 0.5: Top-1 teste 0,825 ± 0,006, Top-3 0,994 ± 0,002, rec. assoc. 0,573 ± 0,002, SNIS 0,542 ± 0,007, DR 0,554 ± 0,003. O EP entre sementes (0,0016) é muito menor que o binomial porque as sementes quase não alteram o Top-1 de validação; com ele a regra seleciona o próprio λ*.

| λ_reg | Top-1 val | Top-3 val | Rec. assoc. val | Top-1 teste | Top-3 teste | Rec. assoc. teste | SNIS teste | DR teste | épocas |
|---|---|---|---|---|---|---|---|---|---|
| 0,000 | 0,719 ± 0,007 | 0,736 ± 0,011 | 0,682 ± 0,005 | 0,673 ± 0,010 | 0,683 ± 0,014 | 0,681 ± 0,008 | 0,680 ± 0,009 | 0,637 ± 0,008 | 15 |
| 0,010 | 0,781 ± 0,023 | 0,931 ± 0,094 | 0,636 ± 0,016 | 0,709 ± 0,015 | 0,916 ± 0,106 | 0,651 ± 0,010 | 0,647 ± 0,011 | 0,625 ± 0,009 | 21 |
| 0,025 | 0,799 ± 0,000 | 0,945 ± 0,052 | 0,627 ± 0,000 | 0,731 ± 0,000 | 0,918 ± 0,077 | 0,637 ± 0,000 | 0,639 ± 0,000 | 0,630 ± 0,001 | 15 |
| 0,050 | 0,808 ± 0,008 | 0,984 ± 0,023 | 0,621 ± 0,005 | 0,743 ± 0,008 | 0,977 ± 0,033 | 0,628 ± 0,005 | 0,623 ± 0,011 | 0,619 ± 0,008 | 17 |
| 0,075 | 0,816 ± 0,006 | 0,998 ± 0,002 | 0,616 ± 0,003 | 0,745 ± 0,008 | 0,997 ± 0,001 | 0,627 ± 0,006 | 0,614 ± 0,010 | 0,608 ± 0,008 | 21 |
| 0,100 | 0,825 ± 0,004 | 0,998 ± 0,001 | 0,609 ± 0,006 | 0,767 ± 0,021 | 0,996 ± 0,001 | 0,610 ± 0,015 | 0,594 ± 0,023 | 0,590 ± 0,017 | 24 |
| 0,250 | 0,855 ± 0,008 | 0,998 ± 0,001 | 0,592 ± 0,004 | 0,821 ± 0,011 | 0,995 ± 0,002 | 0,577 ± 0,005 | 0,546 ± 0,012 | 0,557 ± 0,007 | 26 |
| 0,500 | 0,871 ± 0,003 | 0,998 ± 0,001 | 0,584 ± 0,002 | 0,825 ± 0,006 | 0,994 ± 0,002 | 0,573 ± 0,002 | 0,542 ± 0,007 | 0,554 ± 0,003 | 29 |
| 1,000 | 0,874 ± 0,003 | 0,997 ± 0,003 | 0,581 ± 0,001 | 0,828 ± 0,003 | 0,994 ± 0,002 | 0,569 ± 0,004 | 0,537 ± 0,006 | 0,551 ± 0,004 | 29 |
| 2,000 | 0,875 ± 0,005 | 0,997 ± 0,003 | 0,578 ± 0,003 | 0,824 ± 0,006 | 0,995 ± 0,002 | 0,565 ± 0,003 | 0,532 ± 0,012 | 0,542 ± 0,006 | 30 |
| 5,000 * † | 0,879 ± 0,004 | 0,996 ± 0,003 | 0,576 ± 0,004 | 0,822 ± 0,005 | 0,995 ± 0,003 | 0,570 ± 0,004 | 0,539 ± 0,010 | 0,547 ± 0,007 | 29 |
| 10,000 | 0,876 ± 0,003 | 0,997 ± 0,002 | 0,576 ± 0,004 | 0,820 ± 0,009 | 0,994 ± 0,003 | 0,567 ± 0,006 | 0,535 ± 0,012 | 0,543 ± 0,011 | 29 |

Top-1 de validação da grade em cada fold da validação cruzada: fold 0: 0.0→0,713, 0.01→0,742, 0.025→0,799, 0.05→0,821, 0.075→0,825, 0.1→0,821, 0.25→0,866, 0.5→0,873, 1.0→0,873, 2.0→0,880, 5.0→0,880, 10.0→0,876; fold 1: 0.0→0,678, 0.01→0,766, 0.025→0,776, 0.05→0,776, 0.075→0,794, 0.1→0,802, 0.25→0,819, 0.5→0,839, 1.0→0,829, 2.0→0,824, 5.0→0,829, 10.0→0,827; fold 2: 0.0→0,738, 0.01→0,761, 0.025→0,810, 0.05→0,810, 0.075→0,810, 0.1→0,838, 0.25→0,878, 0.5→0,878, 1.0→0,885, 2.0→0,878, 5.0→0,878, 10.0→0,883; fold 3: 0.0→0,652, 0.01→0,757, 0.025→0,770, 0.05→0,801, 0.075→0,826, 0.1→0,841, 0.25→0,870, 0.5→0,868, 1.0→0,875, 2.0→0,873, 5.0→0,875, 10.0→0,873; fold 4: 0.0→0,683, 0.01→0,731, 0.025→0,731, 0.05→0,749, 0.075→0,776, 0.1→0,776, 0.25→0,821, 0.5→0,829, 1.0→0,837, 2.0→0,832, 5.0→0,829, 10.0→0,829.

## 5. Experimento cultural (E6)

País no estado (one-hot BR/AO/PT, 3 dimensões) em ambas as configurações; recompensa R (sem) ou 0,85·R + 0,15·f_cult (com). DQN λ* = 5.0, split fixo por perfil, 5 sementes. Países no teste: {'AO': 150, 'BR': 150, 'PT': 75}.

| Configuração | Top-1 teste | Divergência por país |
|---|---|---|
| f_cult peso 0,00 | 0,810 ± 0,008 | 12,3 ± 2,3 % |
| f_cult peso 0,15 | 0,812 ± 0,009 | 15,0 ± 2,5 % |

Teste t de Welch (com vs sem): t = 1,83, p = 0,106; pareado por semente: t = 2,10, p = 0,103 (n = 5 por grupo).

f_cult só discrimina ações no par (AO, M6): BR vale 1,0 em todas as missões e PT vale 0,5 em todas; portanto a mistura 0,85·R + 0,15·f_cult altera a ordenação de ações apenas para aprendizes de Angola, e só entre M6 e as demais.

## 6. Diagnósticos (E7)

### Top-1 por segmento (DQN λ*, 5-fold OOF; dp entre folds)

| Segmento | Top-1 | n | dp entre folds |
|---|---|---|---|
| avancado | 0,843 | 369 | 0,043 |
| iniciante | 0,813 | 716 | 0,042 |
| intermediario | 0,871 | 915 | 0,045 |

Split fixo (5 sementes): avancado: 0,879 ± 0,022 (n = 76); iniciante: 0,785 ± 0,015 (n = 150); intermediario: 0,830 ± 0,008 (n = 149).

### Top-1 por missão da ação logada (DQN λ*, 5-fold OOF)

| Missão | Top-1 | n | dp entre folds |
|---|---|---|---|
| M1_RaioX | 0,800 | 619 | 0,022 |
| M2_Compra_Inteligente | 0,772 | 324 | 0,062 |
| M3_Plano_Acao | 0,827 | 423 | 0,051 |
| M6_Pesquisa_Precos | 0,938 | 634 | 0,026 |

Split fixo (5 sementes): M1_RaioX: 0,799 ± 0,009 (n = 145); M2_Compra_Inteligente: 0,771 ± 0,020 (n = 56); M3_Plano_Acao: 0,774 ± 0,022 (n = 76); M6_Pesquisa_Precos: 0,922 ± 0,006 (n = 98).

### Vazamento entre treino e teste

| Partição | estados de teste presentes no treino | pares (s,a) de teste presentes no treino | n teste | n treino |
|---|---|---|---|---|
| antes_particao_original | 66,2 % | 61,0 % | 213 | 1581 |
| depois_por_perfil_split_fixo | 1,6 % | 1,6 % | 375 | 1207 |

Recompensa refinada: r_miss = 1 em 3,1 % das 2000 transições (era 0 % com o codificador antigo); média 0,516; done = 1 em 8,3 %.

Concordância transição a transição entre o argmax do DQN λ* e a regra sequencial: split fixo 71,0 ± 1,3 % de todas as transições (80,2 ± 1,5 % das 88,5 % em que a regra tem predição); 5-fold OOF 71,7 % / 83,4 %.

Concordância DQN λ_1EP × regra sequencial: split fixo 71,0 ± 1,3 % (todas) / 80,2 ± 1,5 % (regra definida); 5-fold OOF 72,7 % / 84,5 %.

Regra sequencial no teste da partição ORIGINAL: Top-1 0,592 (com repetição no terminal: 0,812).

### Codificador (E1): médias por grupo de atributos

| grupo | s | s' corrigido | s' original |
|---|---|---|---|
| literacy | 0,547 | 0,547 | 0,000 |
| pontos | 0,326 | 0,374 | 0,374 |
| tasks_ratio | 0,524 | 0,578 | 0,578 |
| missions_ratio | 0,352 | 0,405 | 0,000 |
| mission_progress | 0,344 | 0,388 | 0,388 |
| cur_completed | 0,052 | 0,084 | 0,084 |
| cur_terminal | 0,140 | 0,208 | 0,208 |
| miss_flags | 0,352 | 0,405 | 0,000 |
| segmento | 0,333 | 0,333 | 0,333 |
| sentimento | 0,200 | 0,200 | 0,200 |
| missao_atual | 0,250 | 0,250 | 0,250 |
| tarefa_atual | 0,091 | 0,091 | 0,091 |

médias por grupo de atributos sobre as 2.000 transições; no codificador original s' tinha literacia 0, segmento sempre 'iniciante' e flags de missão zeradas.

## 7. Hiperparâmetros executados

| Política | hiperparâmetros efetivamente usados |
|---|---|
| BC | hidden=128, depth=2, lr=0.001, batch_size=32, epochs=40, patience=8, min_delta=0.0001, selecao=ce_val |
| r̂(s,a) (modelo direto do DR) | hidden=128, depth=2, lr=0.001, batch_size=32, epochs=40, patience=8, min_delta=0.0001, selecao=mse_val |
| CQL | hidden=128, depth=2, dropout=0.0, lr=0.001, gamma=0.95, alpha=0.2, batch_size=32, epochs=60, patience=8, min_delta=0.0001, target_update=20, weight_decay=0.0, loss=mse, selecao=loss_val |
| PPO offline | hidden=128, depth=2, lr=0.00025, gamma=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.001, batch_size=32, epochs=40, patience=8, min_delta=0.0001, m6_reward_mult=1.0, selecao=top1_val |
| DQN (todos os λ da grade) | hidden=256, depth=2, dropout=0.1, lr=0.003, gamma=0.9, batch_size=32, epochs=80, patience=12, target_update=5, loss=smooth_l1, selecao=top1_val |

Épocas efetivamente treinadas (parada antecipada), split fixo, média entre sementes: DQN λ=0.0: 15, DQN λ=0.01: 21, DQN λ=0.025: 15, DQN λ=0.05: 17, DQN λ=0.075: 21, DQN λ=0.1: 24, DQN λ=0.25: 26, DQN λ=0.5: 29, DQN λ=1.0: 29, DQN λ=2.0: 30, DQN λ=5.0: 29, DQN λ=10.0: 29; bc: 39, rmodel: 20, cql: 32, ppo: 12 (semente 42).

PPO offline: implementação de imitação ponderada por vantagem com recorte — a razão π/π_old é avaliada apenas nas ações logadas, a vantagem é r + γV(s') − V(s) com V ajustada nos mesmos dados, e não há correção off-policy em relação à política de comportamento.

OPE: SNIS com π_b por contagem empírica quando o estado codificado aparece ≥ 2 vezes no conjunto avaliado e pela BC (softmax) nos demais; π_b truncada em 0,01. DR = média do termo direto r̂(s, π(s)) + correção por importância auto-normalizada com r̂(s,a) (regressor de recompensa imediata, MLP 128, treinado no treino). DR-Q(CQL) usa a Q do CQL como modelo direto, como pede a especificação; está na escala de retorno descontado (γ = 0,95) e não na de recompensa imediata, por isso não é comparável às demais colunas.

## 8. O que mudou em relação a 2026-08-29 e por quê

(1) O codificador do estado seguinte recebia `next_state` sem `profile` e sem `missions_completed`: literacia 0, segmento 'iniciante' e flags de missão zeradas em TODOS os s'. Agora s' usa o profile da transição e missões derivadas de `tasks_completed` (também em s, para a mesma distribuição); r_miss, antes sempre 0, passa a ser ativo. (2) A partição original era por interação, com vazamento de estados entre treino e teste; a nova é por perfil (user_id), em split fixo e em 5-fold. (3) Entrou a regra sequencial determinística como baseline e OPE (SNIS, DR) com IC por bootstrap. (4) λ_reg foi reselecionado na validação, na nova grade, e a tabela principal passa a ser o 5-fold por perfil. Os números antigos (DQN 0,574/0,709; BC 0,901; CQL 0,809; λ = 0,05 → 0,832/1,000/0,563; divergência 8,7 % vs 7,2 %) foram obtidos com o codificador defeituoso e com vazamento, e não são comparáveis aos daqui.

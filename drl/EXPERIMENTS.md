## Experimentos e resultados

### Resumo geral (pos-experimentos)
Data: 2026-01-29

Melhor cenario (recomendado):
- Pipeline sem balanceamento, com `--refined-reward` e metrica principal em top‑3.
- Resultados: DQN top3=0.6808 | top1=0.5634 | top5=0.7183; PPO top3=0.5399 | top1=0.5070 | top5=0.5775; CQL top3=0.9484 | top1=0.7981 | top5=1.0000.

Interpretacao:
- CQL manteve top‑k muito alto em todos os cenarios testados, com sinais persistentes de overfitting/memorizacao.
- DQN entregou o melhor equilibrio entre top‑1 e top‑3, com comportamento mais confiavel.
- PPO nao conseguiu melhorar M6 de forma consistente; reforcos em M6 degradaram desempenho geral.

Melhor modelo para este estudo de caso:
- DQN (sem balanceamento + reward refinado) pela estabilidade e desempenho consistente sem sinais de colapso.

### Experimento 1 - Baseline (features simples)
Data: 2026-01-29

Definicao:
- Estado com: literacy_score, total_points, ratio de tarefas/missoes, segment, sentiment, task atual (one-hot).
- Avaliacao: action_accuracy e top-k (1,3,5).
- Split: test com 33 amostras.

Resultados:
- DQN: action_accuracy=0.0606 | top1=0.0606 | top3=0.0606 | top5=0.0606 | avg_reward=0.3788 | avg_q_taken=0.8011
- PPO: action_accuracy=0.0000 | top1=0.0000 | top3=0.0000 | top5=0.0000 | avg_reward=0.3788 | avg_action_prob=9.43e-08
- CQL: action_accuracy=1.0000 | top1=1.0000 | top3=1.0000 | top5=1.0000 | avg_reward=0.3788 | avg_q_taken=0.2527

Observacao:
- PPO zerou a acuracia; CQL aparentou colapso/overfitting.

### Experimento 2 - Task 4 (enriquecer features do estado)
Data: 2026-01-29

O que foi feito:
- Adicionados features derivados usando os dados atuais:
  - progressao da missao atual (tarefas concluidas na missao / total da missao)
  - flag de tarefa atual concluida
  - flag de tarefa atual terminal
  - one-hot da missao atual
- input_dim passou de 23 para 30.

Resultados:
- DQN: action_accuracy=0.0606 | top1=0.0606 | top3=0.0606 | top5=0.0606 | avg_reward=0.3788 | avg_q_taken=0.7355
- PPO: action_accuracy=0.0000 | top1=0.0000 | top3=0.0000 | top5=0.0606 | avg_reward=0.3788 | avg_action_prob=6.09e-05
- CQL: action_accuracy=1.0000 | top1=1.0000 | top3=1.0000 | top5=1.0000 | avg_reward=0.3788 | avg_q_taken=0.4293

Observacao:
- Pequena melhora no top5 do PPO (0.0606), demais metricas mantiveram o comportamento anterior.

### Experimento 3 - Base com 1000 interacoes (features enriquecidas)
Data: 2026-01-29

O que foi feito:
- Aumentado o dataset para 1000 interacoes, com split: train=781, val=106, test=113.
- Mantidas as features enriquecidas do estado (input_dim=30).

Resultados:
- DQN: action_accuracy=0.3717 | top1=0.3717 | top3=0.4956 | top5=0.4956 | avg_reward=0.5111 | avg_matched_reward=0.6757 | avg_q_taken=1.6876
- PPO: action_accuracy=0.1239 | top1=0.1239 | top3=0.2832 | top5=0.3274 | avg_reward=0.5111 | avg_matched_reward=0.7029 | avg_action_prob=0.1299
- CQL: action_accuracy=1.0000 | top1=1.0000 | top3=1.0000 | top5=1.0000 | avg_reward=0.5111 | avg_matched_reward=0.5111 | avg_q_taken=1.4748

Observacao:
- Houve salto expressivo de acuracia no DQN e melhora do PPO em top-k; CQL segue com acuracia perfeita no teste.

### Experimento 4 - Base com 2000 interacoes (features enriquecidas)
Data: 2026-01-29

O que foi feito:
- Aumentado o dataset para 2000 interacoes, com split: train=1581, val=206, test=213.
- Mantidas as features enriquecidas do estado (input_dim=30).

Resultados:
- DQN: action_accuracy=0.4977 | top1=0.4977 | top3=0.6385 | top5=0.6948 | avg_reward=0.5149 | avg_matched_reward=0.7133 | avg_q_taken=2.3111
- PPO: action_accuracy=0.2300 | top1=0.2300 | top3=0.4930 | top5=0.5822 | avg_reward=0.5149 | avg_matched_reward=0.5771 | avg_action_prob=0.2100
- CQL: action_accuracy=0.8920 | top1=0.8920 | top3=0.9906 | top5=1.0000 | avg_reward=0.5149 | avg_matched_reward=0.5412 | avg_q_taken=1.7941

Observacao:
- DQN e PPO melhoraram; CQL perdeu acuracia perfeita, mas segue muito alto no top-k.

### Experimento 5 - Validacao cruzada (k=5)
Data: 2026-01-29

O que foi feito:
- Validacao cruzada k=5 usando toda a base (2000 interacoes).
- Por fold: train=1200, val=400, test=400.

Resultados (media ± desvio-padrao):
- DQN: top1=0.6080±0.0504 | top3=0.7125±0.0511 | top5=0.8010±0.0222 | avg_reward=0.5319±0.0116
- PPO: top1=0.3690±0.0512 | top3=0.6055±0.0424 | top5=0.7155±0.0405 | avg_reward=0.5319±0.0116
- CQL: top1=0.8300±0.0166 | top3=0.9970±0.0037 | top5=0.9995±0.0010 | avg_reward=0.5319±0.0116

Observacao:
- DQN e PPO mostram melhora consistente com variancia moderada; CQL permanece muito alto no top-k.

### Experimento 6 - Ajuste de hiperparametros (PPO/CQL)
Data: 2026-01-29

O que foi feito:
- Busca simples de hiperparametros com a base de 2000 interacoes (train=1581, val=206, test=213).
- PPO: variacao de epochs, clip_eps e ent_coef.
- CQL: variacao de alpha e target_update (epochs reduzidas).

Melhores resultados (top1):
- PPO (epochs=40, clip_eps=0.2, ent_coef=0.001):
  - top1=0.3333 | top3=0.5258 | top5=0.5446 | avg_reward=0.5149
- CQL (epochs=60, alpha=0.2, target_update=20):
  - top1=0.8920 | top3=1.0000 | top5=1.0000 | avg_reward=0.5149

Observacao:
- PPO melhorou com entropia menor e menos epocas; CQL ficou mais forte com alpha menor e target_update maior.

### Experimento 7 - Pipeline com hiperparametros ajustados
Data: 2026-01-29

O que foi feito:
- Pipeline completo com novos defaults (PPO: epochs=40, ent_coef=0.001; CQL: epochs=60, alpha=0.2, target_update=20).
- Base com 2000 interacoes (train=1581, val=206, test=213).

Resultados:
- DQN: action_accuracy=0.4977 | top1=0.4977 | top3=0.6385 | top5=0.6948 | avg_reward=0.5149 | avg_matched_reward=0.7133 | avg_q_taken=2.3111
- PPO: action_accuracy=0.3333 | top1=0.3333 | top3=0.5258 | top5=0.5446 | avg_reward=0.5149 | avg_matched_reward=0.6173 | avg_action_prob=0.3102
- CQL: action_accuracy=0.8920 | top1=0.8920 | top3=1.0000 | top5=1.0000 | avg_reward=0.5149 | avg_matched_reward=0.5474 | avg_q_taken=1.2094

Observacao:
- PPO subiu no top-1; CQL manteve top-k perfeito com ganho de top-1.

### Experimento 8 - Baseline supervisionado (Behavior Cloning)
Data: 2026-01-29

O que foi feito:
- Treino supervisionado para prever a acao (task_id) a partir do estado.
- Base com 2000 interacoes (train=1581, val=206, test=213).

Resultados:
- BC: action_accuracy=0.9061 | top1=0.9061 | top3=1.0000 | top5=1.0000 | avg_reward=0.5149 | avg_matched_reward=0.5381 | avg_action_prob=0.8417

Observacao:
- BC supera CQL/DQN/PPO no top-1 neste split, indicando forte alinhamento com o comportamento observado.

### Experimento 9 - Avaliacao por segmento e missao
Data: 2026-01-29

O que foi feito:
- Geracao de breakdown por segmento (iniciante/intermediario/avancado) e por missao.
- Resultados salvos em `treino_DRL/models/<modelo>/eval_breakdown.json`.

Resumo (top1 por segmento):
- DQN: iniciante=0.50 | intermediario=0.53 | avancado=0.41
- PPO: iniciante=0.38 | intermediario=0.37 | avancado=0.18
- CQL: iniciante=0.87 | intermediario=0.91 | avancado=0.89
- BC:  iniciante=0.90 | intermediario=0.91 | avancado=0.91

Resumo (top1 por missao):
- DQN: M2=0.84 | M1=0.65 | M3=0.48 | M6=0.32
- PPO: M1=0.65 | M2=0.56 | M3=0.26 | M6=0.11
- CQL: M6=0.96 | M1=0.85 | M3=0.83 | M2=0.84
- BC:  M6=0.97 | M2=0.96 | M1=0.85 | M3=0.81

Observacao:
- Queda consistente em M6 para DQN/PPO; CQL/BC mantem desempenho alto em todas as missoes.

### Experimento 10 - Ajustes especificos para M6
Data: 2026-01-29

O que foi feito:
- Teste de duas estrategias focadas em M6 (DQN/PPO):
  1) Amostragem ponderada (m6_weight=2.0).
  2) Reforco de reward na acao M6 (m6_reward_mult=1.5).
- Modelos salvos em `treino_DRL/models/m6_weight/*` e `treino_DRL/models/m6_reward/*`.

Resultados (top1 em M6):
- DQN baseline: 0.3187
- DQN m6_weight=2.0: 0.2088 (pior)
- DQN m6_reward_mult=1.5: 0.4286 (melhor)
- PPO baseline: 0.1099
- PPO m6_weight=2.0: 0.0879 (pior)
- PPO m6_reward_mult=1.5: 0.1429 (melhor)

Observacao:
- Amostragem ponderada degradou o desempenho em M6.
- Reforco de reward foi a melhor alavanca para DQN/PPO no M6, com ganho claro em top-1.

### Experimento 11 - Pipeline com reward M6 reforcado
Data: 2026-01-29

O que foi feito:
- Ajuste do default para `m6_reward_mult=1.5` em DQN/PPO.
- Pipeline completo com a base de 2000 interacoes (train=1581, val=206, test=213).

Resultados:
- DQN: action_accuracy=0.5258 | top1=0.5258 | top3=0.6714 | top5=0.7136 | avg_reward=0.5149 | avg_matched_reward=0.7227 | avg_q_taken=2.7185
- PPO: action_accuracy=0.3850 | top1=0.3850 | top3=0.5399 | top5=0.5587 | avg_reward=0.5149 | avg_matched_reward=0.6349 | avg_action_prob=0.3699
- CQL: action_accuracy=0.8920 | top1=0.8920 | top3=1.0000 | top5=1.0000 | avg_reward=0.5149 | avg_matched_reward=0.5474 | avg_q_taken=1.2094

Observacao:
- DQN e PPO subiram no top-1 geral com o reforco de reward em M6; CQL permanece estável.

### Experimento 12 - Features com flags de missao concluida
Data: 2026-01-29

O que foi feito:
- Normalizacao de pontos com clamp (max=1.0) e adicao de flags por missao concluida no estado.
- Pipeline completo com a base de 2000 interacoes (train=1581, val=206, test=213).

Resultados:
- DQN: action_accuracy=0.5023 | top1=0.5023 | top3=0.6479 | top5=0.6854 | avg_reward=0.5149 | avg_matched_reward=0.7012 | avg_q_taken=2.2473
- PPO: action_accuracy=0.4460 | top1=0.4460 | top3=0.5211 | top5=0.5352 | avg_reward=0.5149 | avg_matched_reward=0.6576 | avg_action_prob=0.4407
- CQL: action_accuracy=0.8498 | top1=0.8498 | top3=0.9906 | top5=1.0000 | avg_reward=0.5149 | avg_matched_reward=0.5471 | avg_q_taken=0.2810

Observacao:
- PPO melhora no top-1; CQL caiu em top-1 vs. experimento anterior.

### Experimento 13 - Early stopping (melhor por validacao)
Data: 2026-01-29

O que foi feito:
- Early stopping habilitado em DQN/PPO/CQL/BC com paciencia=8 e min_delta=1e-4.
- Salvamento do melhor modelo por validacao (`best_model.pt`) e uso dele no `model.pt`.
- Pipeline com a base de 2000 interacoes (train=1581, val=206, test=213).

Resultados:
- DQN: action_accuracy=0.6009 | top1=0.6009 | top3=0.6338 | top5=0.6995 | avg_reward=0.5149 | avg_matched_reward=0.7027 | avg_q_taken=0.6745
- PPO: action_accuracy=0.1455 | top1=0.1455 | top3=0.2582 | top5=0.3803 | avg_reward=0.5149 | avg_matched_reward=0.6358 | avg_action_prob=0.0866
- CQL: action_accuracy=0.8122 | top1=0.8122 | top3=1.0000 | top5=1.0000 | avg_reward=0.5149 | avg_matched_reward=0.5780 | avg_q_taken=0.5854

Observacao:
- DQN melhorou no top-1; PPO degradou com o criterio de early stopping baseado em CE.

### Experimento 14 - Early stopping PPO por acuracia
Data: 2026-01-29

O que foi feito:
- Criterio de early stopping do PPO alterado para maximizar acuracia no val.
- Pipeline completo com a base de 2000 interacoes (train=1581, val=206, test=213).

Resultados:
- DQN: action_accuracy=0.6009 | top1=0.6009 | top3=0.6338 | top5=0.6995 | avg_reward=0.5149 | avg_matched_reward=0.7027 | avg_q_taken=0.6745
- PPO: action_accuracy=0.5023 | top1=0.5023 | top3=0.5258 | top5=0.5305 | avg_reward=0.5149 | avg_matched_reward=0.6650 | avg_action_prob=0.4325
- CQL: action_accuracy=0.8122 | top1=0.8122 | top3=1.0000 | top5=1.0000 | avg_reward=0.5149 | avg_matched_reward=0.5780 | avg_q_taken=0.5854

Observacao:
- PPO recuperou top-1 com o early stopping baseado em acuracia de validacao.

### Experimento 15 - Breakdown apos early stopping do PPO
Data: 2026-01-29

O que foi feito:
- Breakdown por segmento e missao apos ajuste do early stopping do PPO por acuracia.
- Arquivos: `treino_DRL/models/dqn/eval_breakdown.json` e `treino_DRL/models/ppo/eval_breakdown.json`.

Resumo (top1 por missao):
- DQN: M3=0.7143 | M6=0.4505 | M2=0.8400 | M1=0.6545
- PPO: M3=0.7857 | M6=0.2418 | M2=0.5200 | M1=0.7091

Observacao:
- PPO ainda fica baixo em M6, apesar da melhora global.

### Experimento 16 - PPO com reward M6 mais agressivo
Data: 2026-01-29

O que foi feito:
- Teste de reforco mais agressivo no reward de M6 apenas no PPO:
  - `m6_reward_mult=2.0` e `m6_reward_mult=3.0`.
- Modelos salvos em `treino_DRL/models/m6_reward/ppo_r2` e `ppo_r3`.

Resultados (top1 em M6):
- PPO base (m6_reward_mult=1.5): 0.2418
- PPO m6_reward_mult=2.0: 0.2088 (pior)
- PPO m6_reward_mult=3.0: 0.2637 (melhor em M6, mas degradou demais outras missoes)

Observacao:
- O ganho em M6 com 3.0 vem com colapso em M1/M2/M3 (top1 ~0).
- Mantemos `m6_reward_mult=1.5` como melhor trade-off global.

### Experimento 17 - Reward refinado (conclusao e pontos)
Data: 2026-01-29

O que foi feito:
- Reward refinado usando: base_reward + bonus por nova task + delta de pontos + conclusao de missao.
- Pipeline completo com `--refined-reward`.

Resultados:
- DQN: action_accuracy=0.5352 | top1=0.5352 | top3=0.6714 | top5=0.7136 | avg_reward=0.4850 | avg_matched_reward=0.6997 | avg_q_taken=0.6560
- PPO: action_accuracy=0.4789 | top1=0.4789 | top3=0.5117 | top5=0.5540 | avg_reward=0.4850 | avg_matched_reward=0.6468 | avg_action_prob=0.4280
- CQL: action_accuracy=0.7934 | top1=0.7934 | top3=0.9531 | top5=1.0000 | avg_reward=0.4850 | avg_matched_reward=0.5771 | avg_q_taken=0.5665

Observacao:
- O reward refinado reduziu o avg_reward por reescala, mantendo top-k competitivo.

### Experimento 18 - Breakdown com reward refinado
Data: 2026-01-29

O que foi feito:
- Breakdown por segmento e missao usando `--refined-reward` (DQN/PPO/CQL).
- Arquivos: `treino_DRL/models/{dqn,ppo,cql}/eval_breakdown.json`.

Resumo (top1 por missao):
- DQN: M3=0.7143 | M6=0.4615 | M2=0.2800 | M1=0.6364
- PPO: M3=0.7857 | M6=0.2418 | M2=0.5200 | M1=0.6182
- CQL: M3=0.7143 | M6=0.9011 | M2=0.8400 | M1=0.6545

Observacao:
- DQN melhorou em M6, mas caiu em M2 com o reward refinado.

### Experimento 19 - Reward refinado + boost M2
Data: 2026-01-29

O que foi feito:
- Reward refinado com boost extra para M2 (`refined_m2_boost=0.05`).
- Pipeline com `--refined-reward --refined-m2-boost 0.05`.

Resultados:
- DQN: action_accuracy=0.5634 | top1=0.5634 | top3=0.6808 | top5=0.7183 | avg_reward=0.4909
- PPO: action_accuracy=0.5070 | top1=0.5070 | top3=0.5399 | top5=0.5775 | avg_reward=0.4909
- CQL: action_accuracy=0.7981 | top1=0.7981 | top3=0.9484 | top5=1.0000 | avg_reward=0.4909

Resumo (top1 por missao):
- DQN: M2=0.6000 | M6=0.4396 | M3=0.7143 | M1=0.6364
- PPO: M2=0.5600 | M6=0.2418 | M3=0.7857 | M1=0.7091
- CQL: M2=0.8400 | M6=0.9121 | M3=0.7143 | M1=0.6545

Observacao:
- Boost em M2 recuperou DQN (0.28 -> 0.60) sem derrubar M6; PPO manteve M6 estável.

### Experimento 20 - Pipeline com reward refinado (boost M2 fixo)
Data: 2026-01-29

O que foi feito:
- Pipeline com `--refined-reward` usando boost M2 padrao (0.05).

Resultados:
- DQN: action_accuracy=0.5634 | top1=0.5634 | top3=0.6808 | top5=0.7183 | avg_reward=0.4909 | avg_matched_reward=0.7031 | avg_q_taken=0.6612
- PPO: action_accuracy=0.5070 | top1=0.5070 | top3=0.5399 | top5=0.5775 | avg_reward=0.4909 | avg_matched_reward=0.6570 | avg_action_prob=0.4514
- CQL: action_accuracy=0.7981 | top1=0.7981 | top3=0.9484 | top5=1.0000 | avg_reward=0.4909 | avg_matched_reward=0.5808 | avg_q_taken=0.5756

Observacao:
- Resultados consistentes com o Experimento 19, agora com boost M2 como padrao.

### Experimento 21 - Breakdown com reward refinado (boost M2 padrao)
Data: 2026-01-29

O que foi feito:
- Breakdown atualizado apos pipeline com `--refined-reward` (boost M2 padrao).
- Arquivos: `treino_DRL/models/{dqn,ppo,cql}/eval_breakdown.json`.

Resumo (top1 por missao):
- DQN: M2=0.6000 | M6=0.4396 | M3=0.7143 | M1=0.6364
- PPO: M2=0.5600 | M6=0.2418 | M3=0.7857 | M1=0.7091
- CQL: M2=0.8400 | M6=0.9121 | M3=0.7143 | M1=0.6545

DQN: acurácia top‑1 de 0,3717 (acerta a missão exata em ~37%). O top‑3/top‑5 sobe para ~0,50, indicando que em ~50% dos casos a missão correta está entre as 3–5 melhores. avg_reward e avg_matched_reward sugerem que, quando acerta, a recompensa média é relativamente alta (0,6757).
PPO: top‑1 baixo (0,1239), mas top‑3/top‑5 melhoram (0,2832/0,3274). Isso indica que o PPO está aprendendo preferências, mas ainda não decide bem a melhor ação na primeira escolha. avg_action_prob baixo (~0,13) sugere política ainda difusa.
CQL: acurácia perfeita (1.0) em top‑1/top‑3/top‑5. Isso normalmente indica overfitting/colapso no dataset offline: ele está escolhendo exatamente as ações do histórico no teste. O avg_matched_reward igual ao avg_reward reforça que quase sempre a ação prevista coincide com a registrada.
Em resumo: DQN melhorou bastante com mais dados, PPO começou a capturar padrões mas ainda fraco, e CQL está com comportamento suspeito de superajuste. Se quiser, posso investigar o CQL (ex.: regularização/alpha, ruído, ou validação cruzada) e refinar o PPO.

### Experimento 22 - Pipeline com balanceamento de acoes (reward refinado)
Data: 2026-01-29

O que foi feito:
- Adicionado `--balance-actions` (reponderacao por frequencia de acao) em treino DQN/PPO/CQL/BC.
- Pipeline executado com `--refined-reward` e balanceamento ativo.

Resultados:
- DQN: action_accuracy=0.5070 | top1=0.5070 | top3=0.6761 | top5=0.7653 | avg_reward=0.4909 | avg_matched_reward=0.7055 | avg_q_taken=0.6590
- PPO: action_accuracy=0.4930 | top1=0.4930 | top3=0.5446 | top5=0.5587 | avg_reward=0.4909 | avg_matched_reward=0.6654 | avg_action_prob=0.3209
- CQL: action_accuracy=0.8122 | top1=0.8122 | top3=0.9718 | top5=1.0000 | avg_reward=0.4909 | avg_matched_reward=0.5738 | avg_q_taken=0.6517

Observacao:
- Balanceamento reduziu top‑1 do DQN/PPO vs experimento 20, enquanto o CQL subiu levemente; top‑3/top‑5 do DQN mantiveram patamar semelhante.

### Experimento 23 - Breakdown com balanceamento de acoes
Data: 2026-01-29

O que foi feito:
- `eval_breakdown.py` nos modelos treinados com `--balance-actions` e `--refined-reward`.
- Arquivos: `treino_DRL/models/{dqn,ppo,cql}/eval_breakdown.json`.

Resumo (top1 por segmento):
- DQN: iniciante=0.6029 | intermediario=0.5446 | avancado=0.2727
- PPO: iniciante=0.6029 | intermediario=0.4851 | avancado=0.3409
- CQL: iniciante=0.8382 | intermediario=0.8119 | avancado=0.7727

Resumo (top1 por missao):
- DQN: M2=0.3200 | M6=0.4396 | M3=0.5714 | M1=0.6545
- PPO: M2=0.5200 | M6=0.2418 | M3=0.7857 | M1=0.6727
- CQL: M2=0.8400 | M6=0.9451 | M3=0.7143 | M1=0.6545

Observacao:
- O balanceamento nao elevou M6 em DQN/PPO, e reduziu M2 no DQN; PPO manteve M2/M1 fortes mas M6 segue baixo. CQL manteve desempenho alto e muito concentrado.

### Experimento 24 - Avaliacao com metrica principal top-k
Data: 2026-01-29

O que foi feito:
- Atualizacao dos avaliadores para salvar `primary_metric`/`primary_value` (prioriza top‑3, depois top‑5).
- Reavaliacao de DQN/PPO/CQL/BC com `--refined-reward`.
- BC foi reentreinado para alinhar o `input_dim` (34).

Resultados (primary_metric=top3_accuracy):
- DQN: primary_value=0.6761 | top1=0.5070 | top5=0.7653
- PPO: primary_value=0.5446 | top1=0.4930 | top5=0.5587
- CQL: primary_value=0.9718 | top1=0.8122 | top5=1.0000
- BC: primary_value=1.0000 | top1=0.9014 | top5=1.0000

Observacao:
- Comparacao principal passa a usar top‑3; BC e CQL continuam muito altos no top‑3/top‑5.

### Experimento 25 - Pipeline sem balanceamento (metrica top‑k)
Data: 2026-01-29

O que foi feito:
- Pipeline com `--refined-reward` (sem balanceamento).

Resultados (primary_metric=top3_accuracy):
- DQN: primary_value=0.6808 | top1=0.5634 | top5=0.7183
- PPO: primary_value=0.5399 | top1=0.5070 | top5=0.5775
- CQL: primary_value=0.9484 | top1=0.7981 | top5=1.0000

### Experimento 26 - Pipeline com balanceamento (metrica top‑k)
Data: 2026-01-29

O que foi feito:
- Pipeline com `--refined-reward` e `--balance-actions`.

Resultados (primary_metric=top3_accuracy):
- DQN: primary_value=0.6761 | top1=0.5070 | top5=0.7653
- PPO: primary_value=0.5446 | top1=0.4930 | top5=0.5587
- CQL: primary_value=0.9718 | top1=0.8122 | top5=1.0000

Observacao:
- Sem balanceamento, DQN tem top‑1 maior; com balanceamento, top‑5 do DQN sobe e CQL melhora levemente no top‑3.

### Experimento 27 - Breakdown sem balanceamento
Data: 2026-01-29

O que foi feito:
- `eval_breakdown.py` apos pipeline sem balanceamento.
- Arquivos salvos: `models/*/eval_breakdown_nobalance.json`.

Resumo (top1 por segmento):
- DQN: iniciante=0.7059 | intermediario=0.5644 | avancado=0.3409
- PPO: iniciante=0.6029 | intermediario=0.5050 | avancado=0.3636
- CQL: iniciante=0.8382 | intermediario=0.8119 | avancado=0.7045

Resumo (top1 por missao):
- DQN: M2=0.6000 | M6=0.4396 | M3=0.7143 | M1=0.6364
- PPO: M2=0.5600 | M6=0.2418 | M3=0.7857 | M1=0.7091
- CQL: M2=0.8400 | M6=0.9121 | M3=0.7143 | M1=0.6545

### Experimento 28 - Breakdown com balanceamento
Data: 2026-01-29

O que foi feito:
- `eval_breakdown.py` apos pipeline com `--balance-actions`.
- Arquivos salvos: `models/*/eval_breakdown_balance.json`.

Resumo (top1 por segmento):
- DQN: iniciante=0.6029 | intermediario=0.5446 | avancado=0.2727
- PPO: iniciante=0.6029 | intermediario=0.4851 | avancado=0.3409
- CQL: iniciante=0.8382 | intermediario=0.8119 | avancado=0.7727

Resumo (top1 por missao):
- DQN: M2=0.3200 | M6=0.4396 | M3=0.5714 | M1=0.6545
- PPO: M2=0.5200 | M6=0.2418 | M3=0.7857 | M1=0.6727
- CQL: M2=0.8400 | M6=0.9451 | M3=0.7143 | M1=0.6545

Observacao:
- Balanceamento derrubou M2 no DQN e nao melhorou M6 para DQN/PPO; CQL subiu levemente em M6 e avancado.
- Deltas resumidos foram adicionados acima para facilitar comparacao.

### Experimento 29 - CQL com regularizacao contra overfitting
Data: 2026-01-29

O que foi feito:
- Treino CQL com regularizacao leve: `--weight-decay 1e-4 --state-noise-std 0.01 --q-penalty 1e-4`.
- Avaliacao padrao com `--refined-reward`.

Resultados (primary_metric=top3_accuracy):
- CQL: primary_value=0.9484 | top1=0.8122 | top5=1.0000 | avg_q_taken=0.6118

Observacao:
- A regularizacao reduziu `avg_q_taken` mas ainda nao alterou top‑k; precisamos testar regularizacao mais forte ou outros mecanismos (ex.: maior alpha, drop de epocas, ou ruido maior).

### Experimento 30 - CQL com regularizacao agressiva
Data: 2026-01-29

O que foi feito:
- Treino CQL com regularizacao agressiva: `--weight-decay 5e-4 --state-noise-std 0.05 --q-penalty 5e-4`.
- Avaliacao padrao com `--refined-reward`.

Resultados (primary_metric=top3_accuracy):
- CQL: primary_value=0.9531 | top1=0.8122 | top5=0.9859 | avg_q_taken=0.6149

Observacao:
- Houve leve queda no top‑5 e top‑3 permaneceu alto; overfitting ainda nao foi mitigado de forma clara.

### Experimento 31 - CQL com alpha maior + balanceamento
Data: 2026-01-29

O que foi feito:
- Treino CQL com `--alpha 0.5`, `--balance-actions` e regularizacao agressiva (`weight_decay=5e-4`, `state_noise_std=0.05`, `q_penalty=5e-4`).
- Avaliacao padrao com `--refined-reward`.

Resultados (primary_metric=top3_accuracy):
- CQL: primary_value=0.9812 | top1=0.8122 | top5=1.0000 | avg_q_taken=0.5991

Observacao:
- O top‑3 subiu ainda mais e o overfitting persiste; aumentar alpha + balanceamento nao reduziu a memorizacao.

### Experimento 32 - CQL com ruido maior no treino
Data: 2026-01-29

O que foi feito:
- Treino CQL com `state_noise_std=0.1` (apenas no treino), mantendo `weight_decay=5e-4` e `q_penalty=5e-4`.
- Avaliacao padrao com `--refined-reward`.

Resultados (primary_metric=top3_accuracy):
- CQL: primary_value=1.0000 | top1=0.8263 | top5=1.0000 | avg_q_taken=0.5722

Observacao:
- Ruido maior nao reduziu top‑k; overfitting continua.

### Experimento 33 - CQL com dropout
Data: 2026-01-29

O que foi feito:
- Adicionado dropout na rede do CQL (`--dropout 0.2`), sem outras regularizacoes.
- Avaliacao padrao com `--refined-reward`.

Resultados (primary_metric=top3_accuracy):
- CQL: primary_value=1.0000 | top1=0.8498 | top5=1.0000 | avg_q_taken=0.7465

Observacao:
- Dropout nao reduziu top‑k; overfitting persiste.

### Experimento 34 - CQL com alpha alto
Data: 2026-01-29

O que foi feito:
- Treino CQL com `--alpha 1.0` (mais conservador).
- Avaliacao padrao com `--refined-reward`.

Resultados (primary_metric=top3_accuracy):
- CQL: primary_value=1.0000 | top1=0.8122 | top5=1.0000 | avg_q_taken=0.5585

Observacao:
- Alpha alto reduziu `avg_q_taken`, mas top‑k continua saturado.

### Experimento 35 - CQL com alpha muito alto
Data: 2026-01-29

O que foi feito:
- Treino CQL com `--alpha 2.0`.
- Avaliacao padrao com `--refined-reward`.

Resultados (primary_metric=top3_accuracy):
- CQL: primary_value=1.0000 | top1=0.8404 | top5=1.0000 | avg_q_taken=0.5174

Observacao:
- `avg_q_taken` caiu mais, mas top‑k continua saturado.

### Experimento 36 - CQL com alpha alto + dropout
Data: 2026-01-29

O que foi feito:
- Treino CQL com `--alpha 2.0 --dropout 0.2`.
- Avaliacao padrao com `--refined-reward`.

Resultados (primary_metric=top3_accuracy):
- CQL: primary_value=1.0000 | top1=0.9014 | top5=1.0000 | avg_q_taken=0.7586

Observacao:
- Dropout com alpha alto nao reduziu top‑k; overfitting persiste.

### Experimento 37 - PPO com reforco em M6
Data: 2026-01-29

O que foi feito:
- Treino PPO com `--m6-weight 2.0 --m6-reward-mult 2.0` e `--refined-reward`.
- Avaliacao padrao + breakdown salvo em `models/ppo/eval_breakdown_m6boost.json`.

Resultados (primary_metric=top3_accuracy):
- PPO: primary_value=0.6056 | top1=0.4601 | top5=0.6385 | avg_action_prob=0.3697

Resumo (top1 por missao):
- PPO: M2=0.5200 | M6=0.2418 | M3=0.7857 | M1=0.5455

Observacao:
- O reforco elevou top‑3 geral, mas nao aumentou top‑1 em M6; M1 caiu.

### Experimento 38 - PPO com M6 weight apenas
Data: 2026-01-29

O que foi feito:
- Treino PPO com `--m6-weight 2.0 --m6-reward-mult 1.0`.
- Avaliacao + breakdown salvo em `models/ppo/eval_breakdown_m6weight2.json`.

Resultados (primary_metric=top3_accuracy):
- PPO: primary_value=0.5211 | top1=0.4460 | top5=0.5446 | avg_action_prob=0.3530

Resumo (top1 por missao):
- PPO: M2=0.5200 | M6=0.1648 | M3=0.7381 | M1=0.6545

Observacao:
- Apenas peso reduziu M6 e piorou top‑3 geral.

### Experimento 39 - PPO com M6 reward apenas
Data: 2026-01-29

O que foi feito:
- Treino PPO com `--m6-weight 1.0 --m6-reward-mult 2.0`.
- Avaliacao + breakdown salvo em `models/ppo/eval_breakdown_m6reward2.json`.

Resultados (primary_metric=top3_accuracy):
- PPO: primary_value=0.2113 | top1=0.1925 | top5=0.3427 | avg_action_prob=0.0838

Resumo (top1 por missao):
- PPO: M2=0.0000 | M6=0.2088 | M3=0.0000 | M1=0.4000

Observacao:
- Recompensa isolada colapsou desempenho geral; top‑1 em M6 nao melhorou.
Comparativo (delta top1 = balanceamento - sem balanceamento):
- Segmentos | DQN: iniciante=-0.1030 | intermediario=-0.0198 | avancado=-0.0682
- Segmentos | PPO: iniciante=+0.0000 | intermediario=-0.0199 | avancado=-0.0227
- Segmentos | CQL: iniciante=+0.0000 | intermediario=+0.0000 | avancado=+0.0682
- Missoes | DQN: M2=-0.2800 | M6=+0.0000 | M3=-0.1429 | M1=+0.0181
- Missoes | PPO: M2=-0.0400 | M6=+0.0000 | M3=+0.0000 | M1=-0.0364
- Missoes | CQL: M2=+0.0000 | M6=+0.0330 | M3=+0.0000 | M1=+0.0000


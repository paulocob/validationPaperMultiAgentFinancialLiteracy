## Treino DRL - base mockada

Esta pasta contém um conjunto de dados mockado para treinar um motor de
recomendação de próxima missão (DRL) no contexto da Jornada da Tranquilidade.

Estrutura:
- `data/users.jsonl`: perfis de usuários e contexto atual.
- `data/tasks.json`: catálogo de missões/tarefas disponíveis.
- `data/interactions.jsonl`: histórico de interações (estado, ação, recompensa).
- `data/splits.json`: divisões de treino/validação/teste por `interaction_id`.
- `train_utils.py`: utilitários de carga e codificação de estado.
- `train_dqn.py`: treino DQN offline.
- `train_ppo.py`: treino PPO offline.
- `train_cql.py`: treino Offline RL (CQL).
- `eval_utils.py`: utilitários de avaliação.
- `eval_dqn.py`: avaliação DQN.
- `eval_ppo.py`: avaliação PPO.
- `eval_cql.py`: avaliação CQL.
- `pipeline.py`: treino + avaliação sequenciais.

### Campos principais (MDP)
- Estado (`state`): perfil, etapa atual, progresso, histórico resumido, sentimento.
- Ação (`action`): id da missão/tarefa recomendada.
- Recompensa (`reward`): proxy de conclusão, ganho de conhecimento e tempo inativo.
- Próximo estado (`next_state`): estado após a ação.

### Observação
Os dados são sintéticos e servem como base inicial. Ajuste conforme seu fluxo
real (missões, etapas, métricas de sucesso, etc.).

### Treinamento (primeiro baseline)
Pré-requisito:
- Python 3.10+
- `pip install torch`

Executar:
- `python treino_DRL/train_dqn.py`
- `python treino_DRL/train_ppo.py`
- `python treino_DRL/train_cql.py`

Uso de GPU no Mac (MPS):
- Por padrão usa `mps` se disponível (`--device auto`).
- Forçar CPU: `--device cpu`

Os modelos e o metadata são salvos em `treino_DRL/models/<estrategia>/`.
Os logs ficam em `treino_DRL/models/<estrategia>/train.log`.

### Avaliação
- `python treino_DRL/eval_dqn.py`
- `python treino_DRL/eval_ppo.py`
- `python treino_DRL/eval_cql.py`

Métricas salvas em `treino_DRL/models/<estrategia>/eval_metrics.json`.
Suporte a top-k: `--topk 1,3,5`
Métrica principal: top‑k (prioriza top‑3, depois top‑5, senão top‑1), salva como `primary_metric` e `primary_value`.

### Pipeline (treino + avaliação)
- `python treino_DRL/pipeline.py`

Log do pipeline em `treino_DRL/models/pipeline.log`.

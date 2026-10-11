"""Carga de dados, codificador corrigido, partição por perfil e regra sequencial.

Reutiliza `load_tasks`, `load_interactions`, `load_splits` e o vocabulário de
`train_utils.py` (pasta-mãe), sem modificá-lo. O codificador aqui difere do
original em dois pontos (item E1 da especificação):

1. o estado seguinte s' recebe o `profile` da própria transição (o `next_state`
   gravado em disco não traz `profile`);
2. `missions_completed` é DERIVADO de `tasks_completed` (missão concluída quando
   todas as suas tarefas do catálogo estão em `tasks_completed`), tanto em s
   quanto em s', para que os dois vetores tenham a mesma distribuição.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent  # treino_DRL
sys.path.insert(0, str(RAIZ))

from train_utils import (  # noqa: E402
    SEGMENTS,
    TransitionDataset,
    Vocab,
    _one_hot,
    load_interactions,
    load_splits,
    load_tasks,
)

DATA_DIR = RAIZ / "data"
# Mesmas 2.000 transições, acrescidas do campo profile.country (BR/AO/PT).
DATA_DIR_CULTURAL = RAIZ / "experimentos_2026-08-29" / "cultural" / "data"

PAISES = ["BR", "AO", "PT"]
PAIS_TO_IDX = {p: i for i, p in enumerate(PAISES)}

# Copiado de experimentos_2026-08-29/cultural/train_utils_com_fcult.py.
# Só o par (AO, M6) difere do restante do país: BR vale 1,0 em todas as missões
# e PT vale 0,5 em todas; logo f_cult só discrimina ações no par (AO, M6).
ADEQUACAO_CULTURAL = {
    "BR": {"M1_RaioX": 1.0, "M2_Compra_Inteligente": 1.0, "M3_Plano_Acao": 1.0, "M6_Pesquisa_Precos": 1.0},
    "AO": {"M1_RaioX": 1.0, "M2_Compra_Inteligente": 1.0, "M3_Plano_Acao": 1.0, "M6_Pesquisa_Precos": 0.5},
    "PT": {"M1_RaioX": 0.5, "M2_Compra_Inteligente": 0.5, "M3_Plano_Acao": 0.5, "M6_Pesquisa_Precos": 0.5},
}
DEFAULT_ADEQUACAO = 0.5


# ---------------------------------------------------------------------------
# Codificador corrigido
# ---------------------------------------------------------------------------
def derive_missions_completed(tasks_completed: Sequence[str], vocab: Vocab) -> List[str]:
    """Missão concluída quando todas as suas tarefas do catálogo foram feitas."""
    done = Counter(vocab.task_to_mission[t] for t in tasks_completed if t in vocab.task_to_mission)
    return sorted(m for m, n in vocab.mission_task_counts.items() if done[m] == n)


def encode_state_corrigido(
    state: Dict,
    profile: Dict,
    vocab: Vocab,
    total_tasks: int,
    with_country: bool = False,
) -> torch.Tensor:
    """Mesmo layout do `encode_state` original; profile explícito e missões derivadas."""
    literacy_score = float(profile.get("literacy_score", 0.0))
    total_points = min(float(state.get("total_points", 0.0)) / 1000.0, 1.0)

    tasks_completed = state.get("tasks_completed", [])
    missions_completed = derive_missions_completed(tasks_completed, vocab)
    tasks_ratio = len(tasks_completed) / max(1, total_tasks)
    missions_ratio = len(missions_completed) / max(1, len(vocab.mission_to_idx))
    missions_set = set(missions_completed)

    segment = profile.get("segment", "iniciante")
    sentiment = state.get("sentiment", "neutral")
    if sentiment not in vocab.sentiment_to_idx:
        sentiment = "other"

    current_task = state.get("current_task")
    current_task_idx = vocab.task_to_idx.get(current_task, -1)
    current_task_completed = 1.0 if current_task in tasks_completed else 0.0
    current_task_terminal = 1.0 if current_task in vocab.terminal_tasks else 0.0

    current_mission = vocab.task_to_mission.get(current_task)
    current_mission_idx = vocab.mission_to_idx.get(current_mission, -1)
    if current_mission and current_mission in vocab.mission_task_counts:
        total_in_mission = max(1, vocab.mission_task_counts[current_mission])
        completed_in_mission = sum(1 for t in tasks_completed if vocab.task_to_mission.get(t) == current_mission)
        mission_progress = completed_in_mission / total_in_mission
    else:
        mission_progress = 0.0

    mission_flags = [
        1.0 if mid in missions_set else 0.0
        for mid, _ in sorted(vocab.mission_to_idx.items(), key=lambda kv: kv[1])
    ]

    features = [
        literacy_score,
        total_points,
        tasks_ratio,
        missions_ratio,
        mission_progress,
        current_task_completed,
        current_task_terminal,
    ]
    features += mission_flags
    features += _one_hot(vocab.segment_to_idx.get(segment, 0), len(vocab.segment_to_idx))
    features += _one_hot(vocab.sentiment_to_idx.get(sentiment, 0), len(vocab.sentiment_to_idx))
    features += _one_hot(current_mission_idx, len(vocab.mission_to_idx))
    features += _one_hot(current_task_idx, len(vocab.task_to_idx))
    if with_country:
        features += _one_hot(PAIS_TO_IDX.get(profile.get("country"), -1), len(PAISES))
    return torch.tensor(features, dtype=torch.float32)


FEATURE_GROUPS = [
    ("literacy", 1), ("pontos", 1), ("tasks_ratio", 1), ("missions_ratio", 1),
    ("mission_progress", 1), ("cur_completed", 1), ("cur_terminal", 1),
    ("miss_flags", 4), ("segmento", 3), ("sentimento", 5), ("missao_atual", 4), ("tarefa_atual", 11),
]


# ---------------------------------------------------------------------------
# Recompensa refinada (idêntica ao código, com missões derivadas em s e s')
# ---------------------------------------------------------------------------
def refined_reward_components(item: Dict, vocab: Vocab, m2_boost: float = 0.05) -> Dict[str, float]:
    state, next_state = item["state"], item["next_state"]
    tasks_before = set(state.get("tasks_completed", []))
    tasks_after = set(next_state.get("tasks_completed", []))
    missions_before = set(derive_missions_completed(tasks_before, vocab))
    missions_after = set(derive_missions_completed(tasks_after, vocab))

    action_id = item["action"]["task_id"]
    action_mission = vocab.task_to_mission.get(action_id)
    r_miss = 1.0 if (action_mission in missions_after and action_mission not in missions_before) else 0.0
    r_nov = 1.0 if len(tasks_after - tasks_before) > 0 else 0.0
    points_delta = max(0.0, float(next_state.get("total_points", 0.0)) - float(state.get("total_points", 0.0)))
    r_prog = min(points_delta / 100.0, 1.0)
    r_base = float(item.get("reward", 0.0))
    m2 = m2_boost if action_mission == "M2_Compra_Inteligente" else 0.0
    refined = 0.45 * r_base + 0.25 * r_nov + 0.2 * r_prog + 0.1 * r_miss + m2
    return {
        "r_base": r_base, "r_nov": r_nov, "r_prog": r_prog, "r_miss": r_miss, "m2": m2,
        "refined": max(0.0, min(refined, 1.0)),
    }


def cultural_reward(item: Dict, vocab: Vocab) -> float:
    pais = (item.get("state", {}).get("profile", {}) or {}).get("country")
    if not pais:
        return DEFAULT_ADEQUACAO
    missao = vocab.task_to_mission.get(item["action"]["task_id"])
    return ADEQUACAO_CULTURAL.get(pais, {}).get(missao, DEFAULT_ADEQUACAO)


# ---------------------------------------------------------------------------
# Conjunto de transições com metadados
# ---------------------------------------------------------------------------
@dataclass
class Conjunto:
    ids: List[str]
    dataset: TransitionDataset
    user_ids: List[str]
    segments: List[str]
    action_missions: List[str]
    current_tasks: List[str]
    rewards: np.ndarray            # recompensa usada no treino/avaliação (refinada)
    r_miss: np.ndarray
    seq_pred: np.ndarray           # ação da regra sequencial (−1 se indefinida)
    seq_rank: np.ndarray           # ranking da regra (N×3), −1 nas posições indefinidas
    countries: List[Optional[str]] = field(default_factory=list)

    def __len__(self):
        return len(self.ids)

    @property
    def states(self) -> torch.Tensor:
        return self.dataset.states

    @property
    def actions(self) -> np.ndarray:
        return self.dataset.actions.numpy()

    def state_keys(self) -> List[bytes]:
        return [s.numpy().tobytes() for s in self.dataset.states]


def catalog_next(vocab: Vocab, tasks: List[Dict]) -> Dict[str, Optional[str]]:
    return {t["task_id"]: (t["next_tasks"][0] if t.get("next_tasks") else None) for t in tasks}


def sequential_rank(current_task: str, nxt: Dict[str, Optional[str]], vocab: Vocab, k: int = 3,
                    repeat_terminal: bool = False) -> List[int]:
    """Ranking da regra "próxima tarefa do catálogo": encadeia next_tasks k vezes.

    Com `repeat_terminal`, uma tarefa terminal (sem próxima) é repetida como
    recomendação (variante documentada no RESULTADOS.md); sem ela, a regra não
    tem predição e a transição conta como erro, como verificou o coordenador.
    """
    rank: List[int] = []
    t = current_task
    while len(rank) < k and t is not None:
        n = nxt.get(t)
        if n is None:
            if repeat_terminal and vocab.task_to_idx[t] not in rank:
                rank.append(vocab.task_to_idx[t])
            break
        rank.append(vocab.task_to_idx[n])
        t = n
    while len(rank) < k:
        rank.append(-1)
    return rank


def build_conjunto(
    items: List[Dict],
    vocab: Vocab,
    nxt: Dict[str, Optional[str]],
    with_country: bool = False,
    cultural_weight: float = 0.0,
    m2_boost: float = 0.05,
    repeat_terminal: bool = False,
) -> Conjunto:
    total_tasks = len(vocab.task_to_idx)
    states, next_states, actions, rewards, dones = [], [], [], [], []
    r_miss, seq_pred, seq_rank = [], [], []
    for it in items:
        profile = it["state"].get("profile", {})
        states.append(encode_state_corrigido(it["state"], profile, vocab, total_tasks, with_country))
        next_states.append(encode_state_corrigido(it["next_state"], profile, vocab, total_tasks, with_country))
        a = it["action"]["task_id"]
        actions.append(vocab.task_to_idx[a])
        comp = refined_reward_components(it, vocab, m2_boost)
        r = comp["refined"]
        if cultural_weight:
            r = (1.0 - cultural_weight) * r + cultural_weight * cultural_reward(it, vocab)
        rewards.append(r)
        r_miss.append(comp["r_miss"])
        # Terminal como no original: ação terminal já concluída em s'.
        dones.append(1.0 if (a in vocab.terminal_tasks and a in it["next_state"].get("tasks_completed", [])) else 0.0)
        rk = sequential_rank(it["state"]["current_task"], nxt, vocab, 3, repeat_terminal)
        seq_rank.append(rk)
        seq_pred.append(rk[0])
    ds = TransitionDataset(
        torch.stack(states),
        torch.tensor(actions, dtype=torch.int64),
        torch.tensor(rewards, dtype=torch.float32),
        torch.stack(next_states),
        torch.tensor(dones, dtype=torch.float32),
    )
    return Conjunto(
        ids=[it["interaction_id"] for it in items],
        dataset=ds,
        user_ids=[it["user_id"] for it in items],
        segments=[it["state"]["profile"].get("segment", "iniciante") for it in items],
        action_missions=[vocab.task_to_mission[it["action"]["task_id"]] for it in items],
        current_tasks=[it["state"]["current_task"] for it in items],
        rewards=np.array(rewards, dtype=np.float64),
        r_miss=np.array(r_miss, dtype=np.float64),
        seq_pred=np.array(seq_pred, dtype=np.int64),
        seq_rank=np.array(seq_rank, dtype=np.int64),
        countries=[it["state"]["profile"].get("country") for it in items],
    )


# ---------------------------------------------------------------------------
# Partição por perfil (user_id), estratificada por segmento de users.jsonl
# ---------------------------------------------------------------------------
def load_users(data_dir: Path = DATA_DIR) -> Dict[str, Dict]:
    users = {}
    with (data_dir / "users.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                u = json.loads(line)
                users[u["user_id"]] = u
    return users


def profile_folds(users: Dict[str, Dict], n_folds: int = 5, seed: int = 42) -> List[List[str]]:
    """Cinco grupos disjuntos de perfis, com segmentos distribuídos por rodízio.

    Os grupos servem tanto ao split fixo (teste = grupo 0, validação = grupo 1,
    treino = grupos 2–4, isto é 5/5/15) quanto à validação cruzada (teste =
    grupo k, validação = grupo k+1, treino = os outros três).
    """
    rng = random.Random(seed)
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    i = 0
    for seg in SEGMENTS:
        uids = sorted(u for u, d in users.items() if d["profile"]["segment"] == seg)
        rng.shuffle(uids)
        for u in uids:
            folds[i % n_folds].append(u)
            i += 1
    return [sorted(f) for f in folds]


def fold_partition(folds: List[List[str]], k: int) -> Dict[str, List[str]]:
    n = len(folds)
    test = folds[k]
    val = folds[(k + 1) % n]
    train = [u for j in range(n) if j not in (k, (k + 1) % n) for u in folds[j]]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def items_by_users(interactions: Dict[str, Dict], users: Sequence[str]) -> List[Dict]:
    us = set(users)
    return [it for it in interactions.values() if it["user_id"] in us]


def items_by_ids(interactions: Dict[str, Dict], ids: Sequence[str]) -> List[Dict]:
    return [interactions[i] for i in ids]


@dataclass
class Base:
    vocab: Vocab
    tasks: List[Dict]
    nxt: Dict[str, Optional[str]]
    interactions: Dict[str, Dict]
    users: Dict[str, Dict]
    original_splits: Dict[str, List[str]]
    data_dir: Path


def load_base(data_dir: Path = DATA_DIR) -> Base:
    tasks, vocab = load_tasks(DATA_DIR)  # catálogo único; a pasta cultural só tem interações/usuários
    interactions = load_interactions(data_dir)
    users = load_users(DATA_DIR)  # segmento de referência (users.jsonl da base principal)
    return Base(vocab, tasks, catalog_next(vocab, tasks), interactions, users, load_splits(DATA_DIR), data_dir)


def set_all_seeds(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g

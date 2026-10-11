import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

SEGMENTS = ["iniciante", "intermediario", "avancado"]
SENTIMENTS = ["neutral", "motivated", "anxious", "tired", "other"]


@dataclass
class Vocab:
    task_to_idx: Dict[str, int]
    idx_to_task: Dict[int, str]
    mission_to_idx: Dict[str, int]
    segment_to_idx: Dict[str, int]
    sentiment_to_idx: Dict[str, int]
    terminal_tasks: List[str]
    task_to_mission: Dict[str, str]
    mission_task_counts: Dict[str, int]


class TransitionDataset(Dataset):
    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones

    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def setup_logger(name: str, output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _one_hot(index: int, size: int) -> List[float]:
    vec = [0.0] * size
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def load_tasks(data_dir: Path) -> Tuple[List[Dict], Vocab]:
    tasks_path = data_dir / "tasks.json"
    with tasks_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    tasks = payload["tasks"]
    task_ids = [t["task_id"] for t in tasks]
    mission_ids = sorted({t["mission_id"] for t in tasks})
    terminal_tasks = [t["task_id"] for t in tasks if not t.get("next_tasks")]
    task_to_mission = {t["task_id"]: t["mission_id"] for t in tasks}
    mission_task_counts: Dict[str, int] = {mid: 0 for mid in mission_ids}
    for t in tasks:
        mission_task_counts[t["mission_id"]] += 1

    task_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    idx_to_task = {i: tid for tid, i in task_to_idx.items()}
    mission_to_idx = {mid: i for i, mid in enumerate(mission_ids)}
    segment_to_idx = {seg: i for i, seg in enumerate(SEGMENTS)}
    sentiment_to_idx = {sent: i for i, sent in enumerate(SENTIMENTS)}

    return tasks, Vocab(
        task_to_idx=task_to_idx,
        idx_to_task=idx_to_task,
        mission_to_idx=mission_to_idx,
        segment_to_idx=segment_to_idx,
        sentiment_to_idx=sentiment_to_idx,
        terminal_tasks=terminal_tasks,
        task_to_mission=task_to_mission,
        mission_task_counts=mission_task_counts,
    )


def load_interactions(data_dir: Path) -> Dict[str, Dict]:
    interactions_path = data_dir / "interactions.jsonl"
    interactions = {}
    with interactions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            interactions[item["interaction_id"]] = item
    return interactions


def load_splits(data_dir: Path) -> Dict[str, List[str]]:
    splits_path = data_dir / "splits.json"
    with splits_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def encode_state(state: Dict, vocab: Vocab, total_tasks: int) -> torch.Tensor:
    profile = state.get("profile", {})
    literacy_score = float(profile.get("literacy_score", 0.0))
    total_points_raw = float(state.get("total_points", 0.0))
    total_points = min(total_points_raw / 1000.0, 1.0)

    tasks_completed = state.get("tasks_completed", [])
    missions_completed = state.get("missions_completed", [])
    tasks_ratio = len(tasks_completed) / max(1, total_tasks)
    missions_ratio = len(missions_completed) / max(1, len(vocab.mission_to_idx))
    missions_completed_set = set(missions_completed)

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
        completed_in_mission = sum(
            1 for t in tasks_completed if vocab.task_to_mission.get(t) == current_mission
        )
        mission_progress = completed_in_mission / total_in_mission
    else:
        mission_progress = 0.0

    mission_completed_flags = [
        1.0 if mission_id in missions_completed_set else 0.0
        for mission_id, _ in sorted(vocab.mission_to_idx.items(), key=lambda item: item[1])
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
    features += mission_completed_flags
    features += _one_hot(vocab.segment_to_idx.get(segment, 0), len(vocab.segment_to_idx))
    features += _one_hot(vocab.sentiment_to_idx.get(sentiment, 0), len(vocab.sentiment_to_idx))
    features += _one_hot(current_mission_idx, len(vocab.mission_to_idx))
    features += _one_hot(current_task_idx, len(vocab.task_to_idx))

    return torch.tensor(features, dtype=torch.float32)


def is_terminal(action_task_id: str, next_state: Dict, vocab: Vocab) -> float:
    if action_task_id not in vocab.terminal_tasks:
        return 0.0
    completed = next_state.get("tasks_completed", [])
    return 1.0 if action_task_id in completed else 0.0


def compute_refined_reward(item: Dict, vocab: Vocab, m2_boost: float = 0.05) -> float:
    base_reward = float(item.get("reward", 0.0))
    state = item.get("state", {})
    next_state = item.get("next_state", {})

    tasks_before = set(state.get("tasks_completed", []))
    tasks_after = set(next_state.get("tasks_completed", []))
    new_tasks = len(tasks_after - tasks_before)

    missions_before = set(state.get("missions_completed", []))
    missions_after = set(next_state.get("missions_completed", []))

    action_id = item.get("action", {}).get("task_id")
    action_mission = vocab.task_to_mission.get(action_id)
    mission_completed = 1.0 if action_mission in missions_after and action_mission not in missions_before else 0.0

    points_before = float(state.get("total_points", 0.0))
    points_after = float(next_state.get("total_points", points_before))
    points_delta = max(0.0, points_after - points_before)
    points_norm = min(points_delta / 100.0, 1.0)

    task_bonus = 1.0 if new_tasks > 0 else 0.0

    refined = (
        0.45 * base_reward
        + 0.25 * task_bonus
        + 0.2 * points_norm
        + 0.1 * mission_completed
    )
    if action_mission == "M2_Compra_Inteligente":
        refined += m2_boost
    return max(0.0, min(refined, 1.0))


def build_split_dataset(
    data_dir: Path,
    split_name: str,
    use_refined_reward: bool = False,
    refined_m2_boost: float = 0.05,
) -> Tuple[TransitionDataset, Vocab, int]:
    _, vocab = load_tasks(data_dir)
    interactions = load_interactions(data_dir)
    splits = load_splits(data_dir)

    ids = splits[split_name]
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    total_tasks = len(vocab.task_to_idx)

    for interaction_id in ids:
        item = interactions[interaction_id]
        state = encode_state(item["state"], vocab, total_tasks)
        next_state = encode_state(item["next_state"], vocab, total_tasks)
        action_id = item["action"]["task_id"]
        action_idx = vocab.task_to_idx[action_id]
        reward = (
            compute_refined_reward(item, vocab, refined_m2_boost)
            if use_refined_reward
            else float(item["reward"])
        )
        done = is_terminal(action_id, item["next_state"], vocab)

        states.append(state)
        actions.append(action_idx)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.stack(next_states)
    dones = torch.tensor(dones, dtype=torch.float32)

    dataset = TransitionDataset(states, actions, rewards, next_states, dones)
    return dataset, vocab, states.size(1)


def build_mission_weights(
    data_dir: Path,
    split_name: str,
    vocab: Vocab,
    mission_id: str,
    boost: float,
) -> torch.Tensor:
    interactions = load_interactions(data_dir)
    splits = load_splits(data_dir)
    ids = splits[split_name]

    weights = []
    for interaction_id in ids:
        item = interactions[interaction_id]
        action_id = item["action"]["task_id"]
        action_mission = vocab.task_to_mission.get(action_id)
        weight = boost if action_mission == mission_id else 1.0
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.double)


def build_action_weights(
    data_dir: Path,
    split_name: str,
    vocab: Vocab,
) -> torch.Tensor:
    interactions = load_interactions(data_dir)
    splits = load_splits(data_dir)
    ids = splits[split_name]

    counts = {i: 0 for i in range(len(vocab.task_to_idx))}
    for interaction_id in ids:
        item = interactions[interaction_id]
        action_id = item["action"]["task_id"]
        action_idx = vocab.task_to_idx[action_id]
        counts[action_idx] += 1

    weights = []
    for interaction_id in ids:
        item = interactions[interaction_id]
        action_id = item["action"]["task_id"]
        action_idx = vocab.task_to_idx[action_id]
        count = max(1, counts[action_idx])
        weights.append(1.0 / count)

    weight_tensor = torch.tensor(weights, dtype=torch.double)
    mean = float(weight_tensor.mean().item()) if len(weights) else 1.0
    if mean > 0:
        weight_tensor = weight_tensor / mean
    return weight_tensor


def save_metadata(output_dir: Path, vocab: Vocab, input_dim: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_dim": input_dim,
        "tasks": [vocab.idx_to_task[i] for i in range(len(vocab.idx_to_task))],
        "segments": SEGMENTS,
        "sentiments": SENTIMENTS,
        "terminal_tasks": vocab.terminal_tasks,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

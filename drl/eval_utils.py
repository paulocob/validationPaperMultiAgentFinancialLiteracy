import json
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        x = self.net[0](x)
        x = self.net[1](x)
        if self.dropout_p > 0:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.net[2](x)
        x = self.net[3](x)
        if self.dropout_p > 0:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.net[4](x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(1)
        return logits, value


def load_metadata(model_dir: Path) -> Dict:
    metadata_path = model_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_metrics(model_dir: Path, metrics: Dict) -> Path:
    output_path = model_dir / "eval_metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return output_path


def parse_topk(value: str, action_dim: int) -> list[int]:
    if not value:
        return [1, 3, 5]
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    items = sorted(set(items))
    return [k for k in items if 1 <= k <= action_dim]


def pick_primary_topk(topk_list: list[int]) -> int:
    if not topk_list:
        return 1
    for k in (3, 5, 1):
        if k in topk_list:
            return k
    return max(topk_list)

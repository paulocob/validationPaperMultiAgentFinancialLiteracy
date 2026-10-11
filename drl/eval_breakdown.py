import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from eval_utils import ActorCritic, QNetwork, parse_topk
from train_utils import encode_state, load_interactions, load_splits, load_tasks, resolve_device, set_seed


def init_bucket(topk_list: list[int]) -> dict:
    return {
        "total": 0,
        "correct": 0,
        "reward_sum": 0.0,
        "matched_reward_sum": 0.0,
        "topk_hits": {k: 0 for k in topk_list},
    }


def update_bucket(bucket: dict, correct: bool, reward: float, topk_hits: dict[int, bool]) -> None:
    bucket["total"] += 1
    bucket["correct"] += int(correct)
    bucket["reward_sum"] += reward
    if correct:
        bucket["matched_reward_sum"] += reward
    for k, hit in topk_hits.items():
        bucket["topk_hits"][k] += int(hit)


def finalize_bucket(bucket: dict) -> dict:
    total = max(1, bucket["total"])
    correct = bucket["correct"]
    accuracy = correct / total
    avg_reward = bucket["reward_sum"] / total
    avg_matched_reward = bucket["matched_reward_sum"] / max(1, correct) if correct else 0.0
    topk_accuracy = {f"top{k}_accuracy": bucket["topk_hits"][k] / total for k in bucket["topk_hits"]}
    return {
        "dataset_size": bucket["total"],
        "action_accuracy": accuracy,
        "avg_reward": avg_reward,
        "avg_matched_reward": avg_matched_reward,
        **topk_accuracy,
    }


def load_model(model_name: str, input_dim: int, action_dim: int, model_path: Path, device: torch.device):
    if model_name in {"dqn", "cql"}:
        model = QNetwork(input_dim, action_dim).to(device)
    elif model_name == "ppo":
        model = ActorCritic(input_dim, action_dim).to(device)
    elif model_name == "bc":
        from train_bc import BCNetwork

        model = BCNetwork(input_dim, action_dim).to(device)
    else:
        raise SystemExit(f"Modelo desconhecido: {model_name}")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliacao por segmento e missao")
    parser.add_argument("--model", type=str, required=True, choices=["dqn", "ppo", "cql", "bc"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--topk", type=str, default="1,3,5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--refined-reward", action="store_true")
    parser.add_argument("--refined-m2-boost", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    model_dir = Path(args.model_dir) if args.model_dir else base_dir / "models" / args.model
    model_path = model_dir / "model.pt"

    tasks, vocab = load_tasks(data_dir)
    interactions = load_interactions(data_dir)
    splits = load_splits(data_dir)
    ids = splits[args.split]

    total_tasks = len(vocab.task_to_idx)
    input_dim = len(encode_state(interactions[ids[0]]["state"], vocab, total_tasks))
    action_dim = len(vocab.task_to_idx)

    topk_list = parse_topk(args.topk, action_dim)
    device = resolve_device(args.device)
    model = load_model(args.model, input_dim, action_dim, model_path, device)

    buckets = {"overall": init_bucket(topk_list)}
    by_segment = defaultdict(lambda: init_bucket(topk_list))
    by_mission = defaultdict(lambda: init_bucket(topk_list))

    for interaction_id in ids:
        item = interactions[interaction_id]
        state = encode_state(item["state"], vocab, total_tasks).to(device)
        action_id = item["action"]["task_id"]
        action_idx = vocab.task_to_idx[action_id]
        reward = float(item["reward"])
        if args.refined_reward:
            from train_utils import compute_refined_reward

            reward = compute_refined_reward(item, vocab, args.refined_m2_boost)

        segment = item.get("state", {}).get("profile", {}).get("segment", "iniciante")
        mission = vocab.task_to_mission.get(action_id, "desconhecida")

        with torch.no_grad():
            if args.model in {"dqn", "cql"}:
                scores = model(state.unsqueeze(0))
            else:
                logits, _ = model(state.unsqueeze(0)) if args.model == "ppo" else (model(state.unsqueeze(0)), None)
                scores = logits

            preds = torch.argmax(scores, dim=1)
            correct = int(preds.item()) == action_idx

            topk_hits = {}
            if topk_list:
                _, topk_idx = torch.topk(scores, k=max(topk_list), dim=1)
                for k in topk_list:
                    hit = topk_idx[:, :k].eq(torch.tensor([action_idx], device=device).unsqueeze(1)).any(dim=1)
                    topk_hits[k] = bool(hit.item())

        update_bucket(buckets["overall"], correct, reward, topk_hits)
        update_bucket(by_segment[segment], correct, reward, topk_hits)
        update_bucket(by_mission[mission], correct, reward, topk_hits)

    payload = {
        "model": args.model,
        "split": args.split,
        "overall": finalize_bucket(buckets["overall"]),
        "by_segment": {seg: finalize_bucket(bucket) for seg, bucket in by_segment.items()},
        "by_mission": {mid: finalize_bucket(bucket) for mid, bucket in by_mission.items()},
    }

    output_path = model_dir / "eval_breakdown.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eval_utils import QNetwork, load_metadata, parse_topk, pick_primary_topk, write_metrics
from train_utils import build_split_dataset, resolve_device, set_seed, setup_logger


def main():
    parser = argparse.ArgumentParser(description="Avaliacao CQL")
    parser.add_argument("--batch-size", type=int, default=64)
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
    model_dir = Path(args.model_dir) if args.model_dir else base_dir / "models" / "cql"
    model_path = model_dir / "model.pt"

    logger = setup_logger("eval_cql", model_dir)
    logger.info("Inicio da avaliacao CQL")

    dataset, vocab, input_dim = build_split_dataset(
        data_dir,
        "test",
        use_refined_reward=args.refined_reward,
        refined_m2_boost=args.refined_m2_boost,
    )
    action_dim = len(vocab.task_to_idx)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    topk_list = parse_topk(args.topk, action_dim)

    device = resolve_device(args.device)
    logger.info("Device: %s", device)
    logger.info("Dataset test: %s amostras", len(dataset))

    metadata = load_metadata(model_dir)
    if metadata.get("input_dim") != input_dim:
        logger.warning(
            "input_dim do metadata (%s) difere do dataset (%s)",
            metadata.get("input_dim"),
            input_dim,
        )

    model = QNetwork(input_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    reward_sum = 0.0
    matched_reward_sum = 0.0
    q_taken_sum = 0.0
    topk_hits = {k: 0 for k in topk_list}

    with torch.no_grad():
        for states, actions, rewards, _, _ in loader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)

            q_values = model(states)
            preds = torch.argmax(q_values, dim=1)
            correct_mask = preds.eq(actions)

            if topk_list:
                topk_vals, topk_idx = torch.topk(q_values, k=max(topk_list), dim=1)
                for k in topk_list:
                    hits = topk_idx[:, :k].eq(actions.unsqueeze(1)).any(dim=1)
                    topk_hits[k] += int(hits.sum().item())

            correct += int(correct_mask.sum().item())
            total += int(actions.size(0))
            reward_sum += float(rewards.sum().item())
            matched_reward_sum += float(rewards[correct_mask].sum().item())

            q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            q_taken_sum += float(q_taken.sum().item())

    accuracy = correct / max(1, total)
    avg_reward = reward_sum / max(1, total)
    avg_matched_reward = matched_reward_sum / max(1, correct) if correct else 0.0
    avg_q_taken = q_taken_sum / max(1, total)

    topk_accuracy = {f"top{k}_accuracy": topk_hits[k] / max(1, total) for k in topk_list}
    primary_k = pick_primary_topk(topk_list)
    primary_metric = f"top{primary_k}_accuracy"
    primary_value = topk_accuracy.get(primary_metric, accuracy)

    metrics = {
        "dataset_size": total,
        "action_accuracy": accuracy,
        "avg_reward": avg_reward,
        "avg_matched_reward": avg_matched_reward,
        "avg_q_taken": avg_q_taken,
        "device": str(device),
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        **topk_accuracy,
    }

    output_path = write_metrics(model_dir, metrics)
    logger.info("Metricas: %s", metrics)
    logger.info("Metricas salvas em %s", output_path)


if __name__ == "__main__":
    main()

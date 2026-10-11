import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_utils import (
    build_action_weights,
    build_split_dataset,
    build_mission_weights,
    resolve_device,
    save_metadata,
    set_seed,
    setup_logger,
)


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


def ppo_loss(
    model,
    old_model,
    states,
    actions,
    rewards,
    next_states,
    dones,
    gamma,
    clip_eps,
    vf_coef,
    ent_coef,
):
    logits, values = model(states)
    with torch.no_grad():
        old_logits, _ = old_model(states)
        _, next_values = model(next_states)
        targets = rewards + gamma * (1.0 - dones) * next_values

    log_probs = torch.log_softmax(logits, dim=1)
    old_log_probs = torch.log_softmax(old_logits, dim=1)
    logp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    old_logp = old_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    advantages = (targets - values).detach()
    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False) + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    ratio = torch.exp(logp - old_logp)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = nn.functional.mse_loss(values, targets.detach())
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean()

    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return total_loss, policy_loss.item(), value_loss.item(), entropy.item()


def evaluate_policy_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for states, actions, _, _, _ in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            logits, _ = model(states)
            preds = torch.argmax(logits, dim=1)
            correct += int(preds.eq(actions).sum().item())
            total += int(actions.size(0))
    model.train()
    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Treino PPO offline")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--m6-weight", type=float, default=1.0)
    parser.add_argument("--m6-reward-mult", type=float, default=1.5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--refined-reward", action="store_true")
    parser.add_argument("--refined-m2-boost", type=float, default=0.05)
    parser.add_argument("--balance-actions", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "models" / "ppo"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("ppo", output_dir)
    logger.info("Inicio do treino PPO")

    train_dataset, vocab, input_dim = build_split_dataset(
        data_dir,
        "train",
        use_refined_reward=args.refined_reward,
        refined_m2_boost=args.refined_m2_boost,
    )
    val_dataset, _, _ = build_split_dataset(
        data_dir,
        "val",
        use_refined_reward=args.refined_reward,
        refined_m2_boost=args.refined_m2_boost,
    )
    action_dim = len(vocab.task_to_idx)
    weights = None
    if args.balance_actions:
        weights = build_action_weights(data_dir, "train", vocab)
    if args.m6_weight != 1.0:
        m6_weights = build_mission_weights(
            data_dir,
            "train",
            vocab,
            "M6_Pesquisa_Precos",
            args.m6_weight,
        )
        weights = m6_weights if weights is None else weights * m6_weights
    if weights is not None:
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    device = resolve_device(args.device)
    logger.info("Device: %s", device)
    logger.info(
        "Hiperparametros: epochs=%s batch=%s gamma=%s lr=%s clip_eps=%s vf_coef=%s ent_coef=%s m6_weight=%s m6_reward_mult=%s",
        args.epochs,
        args.batch_size,
        args.gamma,
        args.lr,
        args.clip_eps,
        args.vf_coef,
        args.ent_coef,
        args.m6_weight,
        args.m6_reward_mult,
    )
    logger.info("Balance actions: %s", args.balance_actions)
    logger.info(
        "Early stopping: enabled=%s patience=%s min_delta=%s",
        not args.no_early_stopping,
        args.patience,
        args.min_delta,
    )
    logger.info(
        "Dimensoes: input_dim=%s action_dim=%s train=%s val=%s",
        input_dim,
        action_dim,
        len(train_dataset),
        len(val_dataset),
    )
    model = ActorCritic(input_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    m6_mask = torch.tensor(
        [
            1.0 if vocab.task_to_mission.get(vocab.idx_to_task[i]) == "M6_Pesquisa_Precos" else 0.0
            for i in range(action_dim)
        ],
        device=device,
    )
    best_val_metric = 0.0
    epochs_no_improve = 0
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        old_model = deepcopy(model).eval()
        epoch_loss = []
        for states, actions, rewards, next_states, dones in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            if args.m6_reward_mult != 1.0:
                reward_mult = 1.0 + (args.m6_reward_mult - 1.0) * m6_mask[actions]
                rewards = rewards * reward_mult

            loss, p_loss, v_loss, entropy = ppo_loss(
                model,
                old_model,
                states,
                actions,
                rewards,
                next_states,
                dones,
                args.gamma,
                args.clip_eps,
                args.vf_coef,
                args.ent_coef,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        avg_loss = sum(epoch_loss) / max(1, len(epoch_loss))
        val_acc = evaluate_policy_accuracy(model, val_loader, device)
        logger.info("Epoch %03d | loss=%.4f | val_acc=%.4f", epoch, avg_loss, val_acc)

        if not args.no_early_stopping:
            if val_acc > best_val_metric + args.min_delta:
                best_val_metric = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                logger.info("Melhor modelo atualizado (val_acc=%.4f)", best_val_metric)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    logger.info(
                        "Early stopping ativado em epoch %s (best_val_acc=%.4f)",
                        epoch,
                        best_val_metric,
                    )
                    break

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    torch.save(model.state_dict(), output_dir / "model.pt")
    save_metadata(output_dir, vocab, input_dim)
    logger.info("Modelo salvo em %s", output_dir)


if __name__ == "__main__":
    main()

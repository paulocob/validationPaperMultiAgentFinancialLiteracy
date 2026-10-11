import argparse
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


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def evaluate_td_loss(model, dataloader, gamma, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for states, actions, rewards, next_states, dones in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            q_values = model(states)
            q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            q_next = model(next_states).max(1)[0]
            target = rewards + gamma * (1.0 - dones) * q_next
            loss = nn.functional.mse_loss(q_action, target)
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    parser = argparse.ArgumentParser(description="Treino DQN offline")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-update", type=int, default=10)
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
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "models" / "dqn"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("dqn", output_dir)
    logger.info("Inicio do treino DQN")

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
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = resolve_device(args.device)
    logger.info("Device: %s", device)
    logger.info(
        "Hiperparametros: epochs=%s batch=%s gamma=%s lr=%s target_update=%s m6_weight=%s m6_reward_mult=%s",
        args.epochs,
        args.batch_size,
        args.gamma,
        args.lr,
        args.target_update,
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
    model = QNetwork(input_dim, action_dim).to(device)
    target = QNetwork(input_dim, action_dim).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    m6_mask = torch.tensor(
        [
            1.0 if vocab.task_to_mission.get(vocab.idx_to_task[i]) == "M6_Pesquisa_Precos" else 0.0
            for i in range(action_dim)
        ],
        device=device,
    )
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        losses = []
        for states, actions, rewards, next_states, dones in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            if args.m6_reward_mult != 1.0:
                reward_mult = 1.0 + (args.m6_reward_mult - 1.0) * m6_mask[actions]
                rewards = rewards * reward_mult

            q_values = model(states)
            q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = target(next_states).max(1)[0]
                target_q = rewards + args.gamma * (1.0 - dones) * q_next

            loss = nn.functional.mse_loss(q_action, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch % args.target_update == 0:
            target.load_state_dict(model.state_dict())

        val_loss = evaluate_td_loss(model, val_loader, args.gamma, device)
        avg_loss = sum(losses) / max(1, len(losses))
        logger.info(
            "Epoch %03d | train_loss=%.4f | val_loss=%.4f",
            epoch,
            avg_loss,
            val_loss,
        )

        if not args.no_early_stopping:
            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                logger.info("Melhor modelo atualizado (val_loss=%.4f)", best_val_loss)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    logger.info(
                        "Early stopping ativado em epoch %s (best_val_loss=%.4f)",
                        epoch,
                        best_val_loss,
                    )
                    break

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    torch.save(model.state_dict(), output_dir / "model.pt")
    save_metadata(output_dir, vocab, input_dim)
    logger.info("Modelo salvo em %s", output_dir)


if __name__ == "__main__":
    main()

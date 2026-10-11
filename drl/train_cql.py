import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_utils import (
    build_action_weights,
    build_split_dataset,
    resolve_device,
    save_metadata,
    set_seed,
    setup_logger,
)


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


def cql_loss(q_values, actions):
    q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    logsumexp = torch.logsumexp(q_values, dim=1)
    return (logsumexp - q_action).mean()


def main():
    parser = argparse.ArgumentParser(description="Treino Offline RL (CQL)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target-update", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--state-noise-std", type=float, default=0.0)
    parser.add_argument("--q-penalty", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
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
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "models" / "cql"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("cql", output_dir)
    logger.info("Inicio do treino CQL")

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

    if args.balance_actions:
        weights = build_action_weights(data_dir, "train", vocab)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = resolve_device(args.device)
    logger.info("Device: %s", device)
    logger.info(
        "Hiperparametros: epochs=%s batch=%s gamma=%s lr=%s alpha=%s target_update=%s",
        args.epochs,
        args.batch_size,
        args.gamma,
        args.lr,
        args.alpha,
        args.target_update,
    )
    logger.info(
        "Regularizacao: weight_decay=%s state_noise_std=%s q_penalty=%s",
        args.weight_decay,
        args.state_noise_std,
        args.q_penalty,
    )
    logger.info("Dropout: %s", args.dropout)
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
    model = QNetwork(input_dim, action_dim, dropout_p=args.dropout).to(device)
    target = QNetwork(input_dim, action_dim, dropout_p=args.dropout).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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

            if args.state_noise_std > 0:
                noise = torch.randn_like(states) * args.state_noise_std
                states = states + noise
                next_states = next_states + noise

            q_values = model(states)
            q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = target(next_states).max(1)[0]
                target_q = rewards + args.gamma * (1.0 - dones) * q_next

            td_loss = nn.functional.mse_loss(q_action, target_q)
            conservative_loss = cql_loss(q_values, actions)
            loss = td_loss + args.alpha * conservative_loss
            if args.q_penalty > 0:
                loss = loss + args.q_penalty * (q_values**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch % args.target_update == 0:
            target.load_state_dict(model.state_dict())

        val_loss = 0.0
        with torch.no_grad():
            for states, actions, rewards, next_states, dones in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                q_values = model(states)
                q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                q_next = model(next_states).max(1)[0]
                target_q = rewards + args.gamma * (1.0 - dones) * q_next
                td_loss = nn.functional.mse_loss(q_action, target_q)
                conservative_loss = cql_loss(q_values, actions)
                val_loss += (td_loss + args.alpha * conservative_loss).item()
        val_loss /= max(1, len(val_loader))

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

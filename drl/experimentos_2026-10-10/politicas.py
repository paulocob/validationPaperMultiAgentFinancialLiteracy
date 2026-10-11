"""Treino das políticas (BC, DQN com regularização λ, CQL, PPO offline) e do
modelo direto de recompensa usado no estimador duplamente robusto.

Os laços reproduzem os scripts da pasta-mãe (train_bc.py, train_cql.py,
train_ppo.py) e o DQN da "configuração atual" (experimentos_2026-08-29/
cultural/sweep_cult.py + dropout de segmento_sementes.py). Os scripts originais
são programas com argparse e partição fixa em disco; por isso os laços foram
copiados aqui, sem alteração de lógica, para aceitar conjuntos em memória. Cada
função devolve o modelo e um dicionário com os hiperparâmetros EFETIVAMENTE
usados (a tabela de hiperparâmetros do RESULTADOS.md é gerada deles).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dados import Conjunto, set_all_seeds

# --------------------------------------------------------------------------
# Configurações (E3). DQN: "configuração atual"; demais: defaults dos scripts.
# --------------------------------------------------------------------------
CFG_DQN = dict(hidden=256, depth=2, dropout=0.1, lr=3e-3, gamma=0.9, batch_size=32,
               epochs=80, patience=12, target_update=5, loss="smooth_l1",
               selecao="top1_val", lambda_reg=0.0)
CFG_BC = dict(hidden=128, depth=2, lr=1e-3, batch_size=32, epochs=40, patience=8,
              min_delta=1e-4, selecao="ce_val")
CFG_CQL = dict(hidden=128, depth=2, dropout=0.0, lr=1e-3, gamma=0.95, alpha=0.2, batch_size=32,
               epochs=60, patience=8, min_delta=1e-4, target_update=20, weight_decay=0.0,
               loss="mse", selecao="loss_val")
CFG_PPO = dict(hidden=128, depth=2, lr=2.5e-4, gamma=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.001,
               batch_size=32, epochs=40, patience=8, min_delta=1e-4, m6_reward_mult=1.0, selecao="top1_val")
CFG_RMODEL = dict(hidden=128, depth=2, lr=1e-3, batch_size=32, epochs=40, patience=8,
                  min_delta=1e-4, selecao="mse_val")


def mlp(input_dim: int, output_dim: int, hidden: int, depth: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Cópia de train_ppo.ActorCritic (h = 128)."""

    def __init__(self, input_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h).squeeze(1)


def _loaders(train: Conjunto, val: Conjunto, batch_size: int, g: torch.Generator):
    tl = DataLoader(train.dataset, batch_size=batch_size, shuffle=True, generator=g)
    vl = DataLoader(val.dataset, batch_size=256, shuffle=False)
    return tl, vl


@torch.no_grad()
def scores(model: nn.Module, states: torch.Tensor) -> np.ndarray:
    """Pontuação por ação (Q, logits...) para todo o conjunto, em modo eval."""
    model.eval()
    out = model(states)
    if isinstance(out, tuple):
        out = out[0]
    return out.numpy().astype(np.float64)


def _top1(model, states, actions) -> float:
    return float((scores(model, states).argmax(1) == actions).mean())


def _clone_state(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------
# DQN com regularização por comportamento: loss = Huber(TD) + λ · CE
# --------------------------------------------------------------------------
def train_dqn(train: Conjunto, val: Conjunto, seed: int, lambda_reg: float = 0.0,
              cfg: Dict = None) -> Tuple[nn.Module, Dict]:
    cfg = dict(CFG_DQN if cfg is None else cfg)
    cfg["lambda_reg"] = lambda_reg
    g = set_all_seeds(seed)
    idim, adim = train.states.shape[1], int(train.actions.max()) + 1
    adim = max(adim, 11)
    tl, _ = _loaders(train, val, cfg["batch_size"], g)
    model = mlp(idim, adim, cfg["hidden"], cfg["depth"], cfg["dropout"])
    target = mlp(idim, adim, cfg["hidden"], cfg["depth"], cfg["dropout"])
    target.load_state_dict(model.state_dict())
    target.eval()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best, best_state, bad, ep_run = -1.0, None, 0, 0
    for ep in range(cfg["epochs"]):
        model.train()
        for s, a, r, ns, d in tl:
            logits = model(s)
            q = logits.gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                tgt = r + cfg["gamma"] * (1 - d) * target(ns).max(1).values
            loss = F.smooth_l1_loss(q, tgt)
            if lambda_reg > 0:
                loss = loss + lambda_reg * F.cross_entropy(logits, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % cfg["target_update"] == 0:
            target.load_state_dict(model.state_dict())
        acc = _top1(model, val.states, val.actions)
        ep_run = ep + 1
        if acc > best:
            best, best_state, bad = acc, _clone_state(model), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {**cfg, "seed": seed, "epochs_run": ep_run, "best_val_top1": best}


# --------------------------------------------------------------------------
# BC (train_bc.py): CE, seleção pela perda de validação
# --------------------------------------------------------------------------
def train_bc(train: Conjunto, val: Conjunto, seed: int, cfg: Dict = None) -> Tuple[nn.Module, Dict]:
    cfg = dict(CFG_BC if cfg is None else cfg)
    g = set_all_seeds(seed)
    idim = train.states.shape[1]
    tl, vl = _loaders(train, val, cfg["batch_size"], g)
    model = mlp(idim, 11, cfg["hidden"], cfg["depth"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best, best_state, bad, ep_run = float("inf"), None, 0, 0
    for ep in range(cfg["epochs"]):
        model.train()
        for s, a, _, _, _ in tl:
            loss = F.cross_entropy(model(s), a)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(np.mean([F.cross_entropy(model(s), a).item() for s, a, _, _, _ in vl]))
        ep_run = ep + 1
        if vloss < best - cfg["min_delta"]:
            best, best_state, bad = vloss, _clone_state(model), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {**cfg, "seed": seed, "epochs_run": ep_run, "best_val_loss": best}


# --------------------------------------------------------------------------
# CQL (train_cql.py): MSE(TD) + α · (logsumexp Q − Q(s,a)), seleção pela perda
# --------------------------------------------------------------------------
def _cql_term(q, a):
    return (torch.logsumexp(q, dim=1) - q.gather(1, a.unsqueeze(1)).squeeze(1)).mean()


def train_cql(train: Conjunto, val: Conjunto, seed: int, cfg: Dict = None) -> Tuple[nn.Module, Dict]:
    cfg = dict(CFG_CQL if cfg is None else cfg)
    g = set_all_seeds(seed)
    idim = train.states.shape[1]
    tl, vl = _loaders(train, val, cfg["batch_size"], g)
    model = mlp(idim, 11, cfg["hidden"], cfg["depth"], cfg["dropout"])
    target = mlp(idim, 11, cfg["hidden"], cfg["depth"], cfg["dropout"])
    target.load_state_dict(model.state_dict())
    target.eval()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, best_state, bad, ep_run = float("inf"), None, 0, 0
    for ep in range(cfg["epochs"]):
        model.train()
        for s, a, r, ns, d in tl:
            q = model(s)
            qa = q.gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                tgt = r + cfg["gamma"] * (1 - d) * target(ns).max(1).values
            loss = F.mse_loss(qa, tgt) + cfg["alpha"] * _cql_term(q, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % cfg["target_update"] == 0:
            target.load_state_dict(model.state_dict())
        # Validação como no original: alvo com a própria rede (não a target).
        model.eval()
        with torch.no_grad():
            vl_sum = 0.0
            for s, a, r, ns, d in vl:
                q = model(s)
                qa = q.gather(1, a.unsqueeze(1)).squeeze(1)
                tgt = r + cfg["gamma"] * (1 - d) * model(ns).max(1).values
                vl_sum += (F.mse_loss(qa, tgt) + cfg["alpha"] * _cql_term(q, a)).item()
            vloss = vl_sum / max(1, len(vl))
        ep_run = ep + 1
        if vloss < best - cfg["min_delta"]:
            best, best_state, bad = vloss, _clone_state(model), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {**cfg, "seed": seed, "epochs_run": ep_run, "best_val_loss": best}


# --------------------------------------------------------------------------
# PPO offline (train_ppo.py). Documentação (E3-f): o "PPO" é imitação
# ponderada por vantagem com recorte — a razão π/π_old é avaliada só nas ações
# LOGADAS, a vantagem é r + γV(s') − V(s) com V ajustada aos mesmos dados, e não
# há correção off-policy em relação à política de comportamento.
# --------------------------------------------------------------------------
def train_ppo(train: Conjunto, val: Conjunto, seed: int, cfg: Dict = None,
              m6_mask: torch.Tensor = None) -> Tuple[nn.Module, Dict]:
    cfg = dict(CFG_PPO if cfg is None else cfg)
    g = set_all_seeds(seed)
    idim = train.states.shape[1]
    tl, _ = _loaders(train, val, cfg["batch_size"], g)
    model = ActorCritic(idim, 11, cfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best, best_state, bad, ep_run = 0.0, None, 0, 0
    for ep in range(cfg["epochs"]):
        old_model = deepcopy(model).eval()
        model.train()
        for s, a, r, ns, d in tl:
            if cfg["m6_reward_mult"] != 1.0 and m6_mask is not None:
                r = r * (1.0 + (cfg["m6_reward_mult"] - 1.0) * m6_mask[a])
            logits, values = model(s)
            with torch.no_grad():
                old_logits, _ = old_model(s)
                _, next_values = model(ns)
                targets = r + cfg["gamma"] * (1.0 - d) * next_values
            logp = torch.log_softmax(logits, 1).gather(1, a.unsqueeze(1)).squeeze(1)
            old_logp = torch.log_softmax(old_logits, 1).gather(1, a.unsqueeze(1)).squeeze(1)
            adv = (targets - values).detach()
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            ratio = torch.exp(logp - old_logp)
            policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv).mean()
            value_loss = F.mse_loss(values, targets.detach())
            probs = torch.softmax(logits, 1)
            entropy = -(probs * torch.log_softmax(logits, 1)).sum(1).mean()
            loss = policy_loss + cfg["vf_coef"] * value_loss - cfg["ent_coef"] * entropy
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = _top1(model, val.states, val.actions)
        ep_run = ep + 1
        if acc > best + cfg["min_delta"]:
            best, best_state, bad = acc, _clone_state(model), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {**cfg, "seed": seed, "epochs_run": ep_run, "best_val_top1": best}


# --------------------------------------------------------------------------
# Modelo direto de recompensa imediata r̂(s, a) para o estimador DR
# --------------------------------------------------------------------------
def train_reward_model(train: Conjunto, val: Conjunto, seed: int, cfg: Dict = None) -> Tuple[nn.Module, Dict]:
    cfg = dict(CFG_RMODEL if cfg is None else cfg)
    g = set_all_seeds(seed)
    idim = train.states.shape[1]
    tl, vl = _loaders(train, val, cfg["batch_size"], g)
    model = mlp(idim, 11, cfg["hidden"], cfg["depth"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best, best_state, bad, ep_run = float("inf"), None, 0, 0
    for ep in range(cfg["epochs"]):
        model.train()
        for s, a, r, _, _ in tl:
            pred = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(pred, r)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(np.mean([F.mse_loss(model(s).gather(1, a.unsqueeze(1)).squeeze(1), r).item()
                                   for s, a, r, _, _ in vl]))
        ep_run = ep + 1
        if vloss < best - cfg["min_delta"]:
            best, best_state, bad = vloss, _clone_state(model), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {**cfg, "seed": seed, "epochs_run": ep_run, "best_val_mse": best}

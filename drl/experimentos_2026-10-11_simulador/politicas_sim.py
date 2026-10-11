"""Políticas comparadas no simulador (Parte P) e o laço de simulação.

Interface comum: `agir(amb, linhas, obs) -> ações` para os aprendizes ativos
`linhas`; `obs` é a observação parcial (S7). Só o oráculo míope lê o estado
verdadeiro do ambiente, e por isso não é implementável fora do simulador.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from simulador import DIM_OBS, Ambiente, Registro

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "experimentos_2026-10-10"))
sys.path.insert(0, str(AQUI.parent))
import politicas as P1010  # noqa: E402  (reaproveita mlp, train_dqn e train_reward_model de 10/10)

N_ACOES = 11


class Politica:
    nome = "base"
    deterministica = True

    def agir(self, amb: Ambiente, linhas: np.ndarray, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def probs(self, obs: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """π(a|s) para a OPE, a partir da observação (e da ação do currículo)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# P1, P2, P6 e a política de comportamento dos logs
# ---------------------------------------------------------------------------
class Curriculo(Politica):
    nome = "curriculo"

    def agir(self, amb, linhas, obs):
        return amb.acao_curriculo(linhas)

    def acao_obs(self, obs, curr):
        return curr


class CurriculoEps(Politica):
    """Currículo com exploração ε uniforme sobre as 11 tarefas (logs de P4a).
    A exploração é uniforme sobre TODAS as tarefas para que qualquer política
    avaliada tenha suporte nos logs (condição da OPE por importância)."""
    nome = "curriculo_eps"
    deterministica = False

    def __init__(self, eps: float, seed: int):
        self.eps, self.rng = eps, np.random.default_rng(seed)

    def agir(self, amb, linhas, obs):
        a = amb.acao_curriculo(linhas)
        explora = self.rng.random(len(linhas)) < self.eps
        return np.where(explora, self.rng.integers(0, N_ACOES, len(linhas)), a)

    def prob_acao(self, a: np.ndarray, curr: np.ndarray) -> np.ndarray:
        return (1 - self.eps) * (a == curr) + self.eps / N_ACOES


class Aleatoria(Politica):
    """P2: uniforme entre as tarefas cujos pré-requisitos do catálogo já foram concluídos."""
    nome = "aleatoria"
    deterministica = False

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def agir(self, amb, linhas, obs):
        ok = amb.prereq_concluidos(linhas)
        sc = self.rng.random(ok.shape) + ok * 2.0
        return sc.argmax(1)


class OraculoMiope(Politica):
    """P6: maior ganho esperado de maestria TOTAL no passo, com o estado verdadeiro:
    ganho_k = (1 − L_k)·T_k·mult_k + F·(L_k − L0_k), onde o segundo termo é o
    esquecimento evitado por exercitar k. Míope: ignora efeitos em passos futuros
    (inclusive o risco de abandono)."""
    nome = "oraculo"

    def agir(self, amb, linhas, obs):
        p, pop = amb.p, amb.pop
        L = amb.L[linhas]
        mult = np.ones_like(L)
        for k, pre in enumerate(amb.cat.prereq):
            if pre:
                mult[:, k] = np.where(L[:, pre].mean(1) < p.limiar_prereq, p.mult_prereq, 1.0)
        ganho = (1 - L) * pop.T[linhas] * mult + pop.F[linhas, None] * (L - pop.L0[linhas])
        return ganho.argmax(1)


class GulosoRastreador(Politica):
    """Referência implementável (não pedida na especificação): a mesma regra do
    oráculo míope, mas sobre a maestria ESTIMADA pelo rastreador e com os
    parâmetros populacionais do segmento. Separa o limite imposto pela
    observabilidade parcial (oráculo − guloso) do limite do aprendizado da
    política (guloso − bandit/DQN)."""
    nome = "guloso_kt"

    def agir(self, amb, linhas, obs):
        kt, p = amb.kt, amb.p
        L = kt.L[linhas]
        mult = np.ones_like(L)
        for k, pre in enumerate(amb.cat.prereq):
            if pre:
                mult[:, k] = np.where(L[:, pre].mean(1) < p.limiar_prereq, p.mult_prereq, 1.0)
        ganho = (1 - L) * kt.T[linhas, None] * mult + kt.F * (L - kt.L_ini[linhas])
        return ganho.argmax(1)


# ---------------------------------------------------------------------------
# Laço de simulação
# ---------------------------------------------------------------------------
def simular(amb: Ambiente, pol: Politica, registro: Optional[Registro] = None,
            aprendiz_online=None, rec_alt: bool = False) -> Dict[str, np.ndarray]:
    """Roda os episódios da população até todos terminarem.

    `aprendiz_online(obs, a, rec, obs2, fim)` é chamado a cada passo (bandits
    e DQN online aprendem com os dados que eles próprios geram). Com `rec_alt`,
    o aprendiz recebe a recompensa com revisão de crença (análise de sensibilidade).
    Devolve o desfecho por aprendiz, acrescido do retorno de treino acumulado."""
    retorno = np.zeros(amb.n)
    while amb.ativo.any():
        linhas = np.flatnonzero(amb.ativo)
        obs = amb.observar(linhas)
        a_lin = pol.agir(amb, linhas, obs)
        acoes = np.zeros(amb.n, dtype=np.int64)
        acoes[linhas] = a_lin
        rec = amb.passo(acoes, registro)
        retorno[linhas] += rec
        if rec_alt:
            rec = amb.ultima_rec_alt
        if aprendiz_online is not None:
            fim = ~amb.ativo[linhas]
            obs2 = amb.observar(linhas)
            aprendiz_online(obs, a_lin, rec, obs2, fim)
    d = amb.desfecho()
    d["retorno"] = retorno
    return d


# ---------------------------------------------------------------------------
# P3: bandits contextuais lineares (braços disjuntos)
# ---------------------------------------------------------------------------
class BanditLinear(Politica):
    """LinUCB (Li et al., 2010) e Thompson sampling linear (Agrawal & Goyal, 2013).

    Um modelo de regressão ridge por braço sobre x = [obs, 1]. Durante o treino:
    LinUCB escolhe argmax xθ_a + α√(x A_a⁻¹ x); TS sorteia θ̃_a ~ N(θ_a, v² A_a⁻¹)
    uma vez por passo do lote (compartilhado entre os aprendizes do lote, para
    custo viável; documentado). Congelado (avaliação), ambos escolhem argmax xθ_a,
    o que torna a política determinística e avaliável por OPE.
    """
    deterministica = True

    def __init__(self, tipo: str, seed: int, alfa: float = 1.0, v: float = 0.5, ridge: float = 1.0):
        self.tipo, self.alfa, self.v = tipo, alfa, v
        self.nome = "linucb" if tipo == "ucb" else "ts"
        d = DIM_OBS + 1
        self.A = np.stack([np.eye(d) * ridge for _ in range(N_ACOES)])
        self.b = np.zeros((N_ACOES, d))
        self.rng = np.random.default_rng(seed)
        self.congelado = False
        self._atualizar_cache()

    def _atualizar_cache(self):
        self.Ainv = np.linalg.inv(self.A)
        self.theta = np.einsum("aij,aj->ai", self.Ainv, self.b)

    @staticmethod
    def _x(obs):
        return np.concatenate([obs.astype(np.float64), np.ones((len(obs), 1))], 1)

    def pontuar(self, obs: np.ndarray) -> np.ndarray:
        x = self._x(obs)
        media = x @ self.theta.T
        if self.congelado:
            return media
        if self.tipo == "ucb":
            var = np.einsum("ni,aij,nj->na", x, self.Ainv, x)
            return media + self.alfa * np.sqrt(np.maximum(var, 0))
        # θ̃ = θ + v·L⁻ᵀz com A = LLᵀ: tem covariância v²A⁻¹ e evita decompor A⁻¹, que
        # é mal condicionada (atributos one-hot colineares com o intercepto).
        z = self.rng.standard_normal((N_ACOES, x.shape[1]))
        Lc = np.linalg.cholesky(self.A)
        th = self.theta + self.v * np.stack([np.linalg.solve(Lc[a].T, z[a]) for a in range(N_ACOES)])
        return x @ th.T

    def agir(self, amb, linhas, obs):
        return self.pontuar(obs).argmax(1)

    def acao_obs(self, obs, curr):
        return self._x(obs).dot(self.theta.T).argmax(1)

    def aprender(self, obs, a, rec, obs2, fim):
        x = self._x(obs)
        for k in np.unique(a):
            sel = a == k
            xs = x[sel]
            self.A[k] += xs.T @ xs
            self.b[k] += xs.T @ rec[sel]
        self._atualizar_cache()


# ---------------------------------------------------------------------------
# P4: DQN (rede da "configuração atual" de 10/10: MLP 256×2)
# ---------------------------------------------------------------------------
class PoliticaQ(Politica):
    """Política gulosa sobre uma rede Q treinada (offline ou online)."""

    def __init__(self, modelo: nn.Module, nome: str):
        self.modelo, self.nome = modelo, nome
        self.modelo.eval()

    @torch.no_grad()
    def acao_obs(self, obs, curr=None):
        return self.modelo(torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))).argmax(1).numpy()

    def agir(self, amb, linhas, obs):
        return self.acao_obs(obs)


class ConjuntoLog:
    """Adaptador mínimo para reaproveitar politicas.train_dqn/train_reward_model de
    10/10, que esperam um objeto com .dataset, .states e .actions."""

    def __init__(self, log: Dict[str, np.ndarray]):
        from train_utils import TransitionDataset
        self.dataset = TransitionDataset(
            torch.from_numpy(log["obs"]), torch.from_numpy(log["acao"]), torch.from_numpy(log["rec"]),
            torch.from_numpy(log["obs2"]), torch.from_numpy(log["fim"]))
        self.states = self.dataset.states
        self.actions = log["acao"]


# Hiperparâmetros do DQN offline: os da "configuração atual" de 10/10, exceto
# batch_size (32 → 256) e épocas (80 → 40, paciência 12 → 8), porque aqui há ~15×
# mais transições; com lote 32 o custo em CPU inviabilizaria a grade de λ.
CFG_DQN_OFF = dict(P1010.CFG_DQN, batch_size=256, epochs=40, patience=8)


def treinar_dqn_offline(log_tr, log_va, seed: int, lam: float):
    modelo, hp = P1010.train_dqn(ConjuntoLog(log_tr), ConjuntoLog(log_va), seed, lambda_reg=lam, cfg=CFG_DQN_OFF)
    return modelo, hp


CFG_DQN_ON = dict(hidden=256, depth=2, lr=1e-3, gamma=0.9, batch=256, buffer=200_000, envs=256,
                  aprendizes=24_000, updates_por_passo=4, eps_ini=1.0, eps_fim=0.05, frac_decaimento=0.5,
                  target_update=250, avaliar_cada=10)


def treinar_dqn_online(cat, p, h, seed, pop_val, amostrar, cfg=None):
    """P4b: DQN com interação livre no simulador (limite superior de DRL).

    Lotes de `envs` aprendizes novos (sementes de treino) rodam episódios
    completos; cada passo alimenta o replay e dispara `updates_por_passo`
    atualizações. A seleção do checkpoint usa o RETORNO MÉDIO de treino (a mesma
    recompensa S8) numa população de validação, não o Hake, para não usar o
    desfecho de avaliação no treino."""
    cfg = dict(CFG_DQN_ON if cfg is None else cfg)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q = P1010.mlp(DIM_OBS, N_ACOES, cfg["hidden"], cfg["depth"])
    alvo = P1010.mlp(DIM_OBS, N_ACOES, cfg["hidden"], cfg["depth"])
    alvo.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg["lr"])
    cap = cfg["buffer"]
    buf = {k: np.zeros((cap, DIM_OBS), np.float32) for k in ("s", "s2")}
    buf.update(a=np.zeros(cap, np.int64), r=np.zeros(cap, np.float32), d=np.zeros(cap, np.float32))
    nbuf, pos, n_upd = 0, 0, 0
    n_lotes = max(1, cfg["aprendizes"] // cfg["envs"])
    melhor, melhor_estado, curva = -np.inf, None, []

    def retorno_val():
        amb = Ambiente(cat, p, pop_val)
        tot = np.zeros(pop_val.n)
        pol = PoliticaQ(q, "dqn_on")
        while amb.ativo.any():
            linhas = np.flatnonzero(amb.ativo)
            acoes = np.zeros(amb.n, dtype=np.int64)
            acoes[linhas] = pol.acao_obs(amb.observar(linhas))
            tot[linhas] += amb.passo(acoes)
        q.train()
        return float(tot.mean())

    for lote in range(n_lotes):
        eps = max(cfg["eps_fim"], cfg["eps_ini"] - (cfg["eps_ini"] - cfg["eps_fim"]) * lote / (cfg["frac_decaimento"] * n_lotes))
        pop = amostrar(cfg["envs"], h, 10_000_000 + seed * 100_000 + lote)
        amb = Ambiente(cat, p, pop)
        while amb.ativo.any():
            linhas = np.flatnonzero(amb.ativo)
            obs = amb.observar(linhas)
            with torch.no_grad():
                q.eval()
                gulosa = q(torch.from_numpy(obs)).argmax(1).numpy()
            a = np.where(rng.random(len(linhas)) < eps, rng.integers(0, N_ACOES, len(linhas)), gulosa)
            acoes = np.zeros(amb.n, dtype=np.int64)
            acoes[linhas] = a
            rec = amb.passo(acoes)
            obs2 = amb.observar(linhas)
            fim = (~amb.ativo[linhas]).astype(np.float32)
            m = len(linhas)
            idx = (pos + np.arange(m)) % cap
            buf["s"][idx], buf["s2"][idx], buf["a"][idx], buf["r"][idx], buf["d"][idx] = obs, obs2, a, rec, fim
            pos, nbuf = (pos + m) % cap, min(cap, nbuf + m)
            q.train()
            for _ in range(cfg["updates_por_passo"]):
                j = rng.integers(0, nbuf, cfg["batch"])
                s, s2 = torch.from_numpy(buf["s"][j]), torch.from_numpy(buf["s2"][j])
                aa, rr, dd = torch.from_numpy(buf["a"][j]), torch.from_numpy(buf["r"][j]), torch.from_numpy(buf["d"][j])
                qa = q(s).gather(1, aa[:, None]).squeeze(1)
                with torch.no_grad():
                    # Double DQN: a rede online escolhe, a alvo avalia (reduz superestimação).
                    a2 = q(s2).argmax(1, keepdim=True)
                    tgt = rr + cfg["gamma"] * (1 - dd) * alvo(s2).gather(1, a2).squeeze(1)
                loss = F.smooth_l1_loss(qa, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                n_upd += 1
                if n_upd % cfg["target_update"] == 0:
                    alvo.load_state_dict(q.state_dict())
        if (lote + 1) % cfg["avaliar_cada"] == 0 or lote == n_lotes - 1:
            rv = retorno_val()
            curva.append((lote + 1, eps, rv))
            # Só checkpoints da fase de baixa exploração são candidatos.
            if lote >= cfg["frac_decaimento"] * n_lotes and rv > melhor:
                melhor, melhor_estado = rv, {k: v.clone() for k, v in q.state_dict().items()}
    if melhor_estado is not None:
        q.load_state_dict(melhor_estado)
    q.eval()
    return q, {**cfg, "seed": seed, "atualizacoes": n_upd, "retorno_val_melhor": melhor, "curva": curva}

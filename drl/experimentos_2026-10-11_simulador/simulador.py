"""Simulador de aprendiz com modelo cognitivo (Parte S da especificação da 7ª rodada).

Visão geral
-----------
Cada aprendiz tem uma habilidade por tarefa do catálogo (11 tarefas, 11
habilidades). A maestria verdadeira L[i, k] é contínua em [0, 1] e evolui pelo
Bayesian Knowledge Tracing estendido (Corbett & Anderson, 1995):

    resposta   y ~ Bernoulli( L_k (1 − S) + (1 − L_k) G )        (antes do aprendizado)
    aprendizado L_k ← L_k + (1 − L_k) · T_eff,   T_eff = T_k · (0,3 se maestria média dos pré-req. < 0,6)
    esquecimento L_j ← L_j − F · (L_j − L0_j)  para toda habilidade j não exercitada no passo

Por que a maestria é contínua e não binária: a especificação define limiares sobre
"maestria" (0,6 para pré-requisitos, 0,9 para tédio, 0,95 para término), o que só
faz sentido para uma probabilidade de domínio por aprendiz. As observações (respostas)
NÃO alteram a maestria verdadeira; servem só ao rastreador do agente.

Por que o esquecimento puxa para L0 e não para zero: o conhecimento prévio medido no
pré-teste é tratado como consolidado. Se o esquecimento levasse a zero, um aprendiz
avançado que não fosse exposto a uma habilidade teria ganho negativo só pela
passagem do tempo, o que é artefato do modelo e não fenômeno de interesse.

Heterogeneidade h ∈ [0, 1] (S3)
------------------------------
Para cada parâmetro θ:
    média do segmento   μ_seg(h) = μ̄ + h · (μ_seg − μ̄)
    média do aprendiz   m_i ~ Beta(média μ_seg(h), dp = h · σ_aprendiz)
    valor por habilidade θ_ik ~ Beta(média m_i, dp = h · σ_habilidade)   (só L0 e T)
Em h = 0 todos os aprendizes têm exatamente os mesmos parâmetros (variância zero);
em h = 1 as médias dos segmentos ficam nos valores nominais e a dispersão individual e
entre habilidades é máxima. A parametrização Beta por média e dp usa
κ = m(1 − m)/dp² − 1, α = mκ, β = (1 − m)κ, com dp limitado a 0,95·√(m(1 − m)).

Engajamento e abandono (S4)
---------------------------
Frustração: a tarefa tem pré-requisitos com maestria média verdadeira < 0,6.
Tédio: a habilidade da tarefa já tem maestria > 0,9.
Acumuladores com decaimento 0,5 por passo (Fr, Bo) fazem a frustração persistente
pesar mais do que um episódio isolado. Probabilidade de abandono no passo:
    p = base + α · Fr + α · Bo
com base = 0,005 e α calibrado para o currículo fixo abandonar ~15 % em h = 1.
O sentimento observável (neutro, motivado, ansioso, cansado) deriva de Fr, Bo e do
último acerto, com 25 % de ruído (substituição uniforme), porque o sentimento real é
autodeclarado e ruidoso.

Desfecho (S5)
-------------
Pré e pós-teste: média, sobre as 11 habilidades, de P(correto) = L(1 − S) + (1 − L)G.
O pós-teste é aplicado no "relógio" H para todos: quem abandona ou termina antes
sofre esquecimento nos passos restantes até H (análise por intenção de tratar).
Hake g_i = (pós − pré)/(1 − pré). Retenção: pós-teste após mais 30 passos de
esquecimento. Para o cálculo de n (A3) há também um teste de 10 itens amostrado
(Binomial), que acrescenta o erro de medida de um instrumento real.

Todos os números aleatórios que dependem do aprendiz (parâmetros, respostas,
abandono, ruído do sentimento, itens do pré-teste observado) são sorteados de
antemão por aprendiz e por passo. Assim, políticas diferentes enfrentam os mesmos
aprendizes e o mesmo ruído (números aleatórios comuns), e as diferenças entre
políticas podem ser estimadas de forma pareada.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

AQUI = Path(__file__).resolve().parent
CATALOGO_PATH = AQUI.parent / "data" / "tasks.json"

SEGMENTOS = ["iniciante", "intermediario", "avancado"]
# Proporções dos segmentos nos 25 perfis de users.jsonl (9 / 11 / 5).
PROP_SEG = np.array([9, 11, 5], dtype=np.float64) / 25.0
SENTIMENTOS = ["neutral", "motivated", "anxious", "tired", "other"]  # vocabulário de train_utils.py

H_PASSOS = 15            # S5: dose aproximada de M1–M3 em 2–3 sessões
PASSOS_RETENCAO = 30     # S5
ITENS_PRE_OBS = 2        # itens por habilidade no pré-teste OBSERVADO pelo agente
ITENS_TESTE_REAL = 10    # instrumento do piloto (10 itens dicotômicos), usado só em A3


# ---------------------------------------------------------------------------
# Catálogo
# ---------------------------------------------------------------------------
@dataclass
class Catalogo:
    ids: List[str]
    nomes: List[str]
    missao: np.ndarray          # índice da missão de cada tarefa
    missoes: List[str]
    pontos: np.ndarray
    prereq: List[List[int]]
    proxima: List[int]          # índice da próxima tarefa (−1 se terminal)
    terminal: np.ndarray        # bool

    @property
    def K(self) -> int:
        return len(self.ids)


def carregar_catalogo(path: Path = CATALOGO_PATH) -> Catalogo:
    tarefas = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    ids = [t["task_id"] for t in tarefas]
    idx = {t: i for i, t in enumerate(ids)}
    missoes = sorted({t["mission_id"] for t in tarefas})  # mesma ordem de train_utils.Vocab
    return Catalogo(
        ids=ids,
        nomes=[t["name"] for t in tarefas],
        missao=np.array([missoes.index(t["mission_id"]) for t in tarefas]),
        missoes=missoes,
        pontos=np.array([t["points"] for t in tarefas], dtype=np.float64),
        prereq=[[idx[p] for p in t["prerequisites"]] for t in tarefas],
        proxima=[idx[t["next_tasks"][0]] if t["next_tasks"] else -1 for t in tarefas],
        terminal=np.array([not t["next_tasks"] for t in tarefas]),
    )


# ---------------------------------------------------------------------------
# Parâmetros do simulador
# ---------------------------------------------------------------------------
@dataclass
class ParamsSim:
    """Parâmetros base. Só c_T (escala de P(T)) e alfa_abandono são calibrados;
    os demais são valores plausíveis, NÃO estimados de dados (S6)."""
    L0_seg: tuple = (0.15, 0.35, 0.60)       # S3 (exemplo da especificação)
    T_seg_rel: tuple = (0.75, 1.00, 1.25)    # P(T) relativo por segmento; P(T) = c_T · rel
    c_T: float = 0.45                         # calibrado (Hake do currículo ≈ 0,50 em h = 0,5)
    S_media: float = 0.10
    G_media: float = 0.20
    F_media: float = 0.01                     # esquecimento por passo sem exposição
    # dp em h = 1 (escalam linearmente com h)
    sd_L0_aprendiz: float = 0.10
    sd_L0_habilidade: float = 0.20
    sd_T_aprendiz: float = 0.10
    sd_T_habilidade: float = 0.05
    sd_S: float = 0.04
    sd_G: float = 0.06
    sd_F: float = 0.005
    # pré-requisitos (S2)
    limiar_prereq: float = 0.6
    mult_prereq: float = 0.3
    # engajamento (S4)
    limiar_tedio: float = 0.9
    abandono_base: float = 0.005
    alfa_abandono: float = 0.05               # calibrado (abandono do currículo ≈ 15 % em h = 1)
    decaimento_afeto: float = 0.5
    ruido_sentimento: float = 0.25
    limiar_maestria: float = 0.95

    def T_seg(self) -> np.ndarray:
        return self.c_T * np.asarray(self.T_seg_rel)

    def as_dict(self) -> Dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def beta_media_dp(rng: np.random.Generator, media: np.ndarray, dp: float, shape=None) -> np.ndarray:
    """Amostra Beta parametrizada por média e desvio-padrão (dp = 0 → constante)."""
    media = np.clip(np.asarray(media, dtype=np.float64), 0.005, 0.995)
    if shape is None:
        shape = media.shape
    media = np.broadcast_to(media, shape)
    if dp <= 0:
        return media.copy()
    dp_max = 0.95 * np.sqrt(media * (1 - media))
    dp_ef = np.minimum(dp, dp_max)
    kappa = media * (1 - media) / dp_ef ** 2 - 1.0
    return rng.beta(media * kappa, (1 - media) * kappa)


def medias_segmento(valores_seg: np.ndarray, h: float) -> np.ndarray:
    """μ_seg(h) = μ̄ + h(μ_seg − μ̄); μ̄ ponderada pelas proporções dos segmentos."""
    valores_seg = np.asarray(valores_seg, dtype=np.float64)
    mu = float((PROP_SEG * valores_seg).sum())
    return mu + h * (valores_seg - mu)


@dataclass
class Populacao:
    h: float
    seg: np.ndarray       # (n,)
    L0: np.ndarray        # (n, K)
    T: np.ndarray         # (n, K)
    S: np.ndarray         # (n,)
    G: np.ndarray         # (n,)
    F: np.ndarray         # (n,)
    # números aleatórios comuns (CRN)
    U_resp: np.ndarray    # (n, H)
    U_aband: np.ndarray   # (n, H)
    U_sent: np.ndarray    # (n, H+1, 2)
    U_pre_obs: np.ndarray  # (n, K, ITENS_PRE_OBS)
    U_teste: np.ndarray   # (n, 3, ITENS_TESTE_REAL) — pré, pós, retenção

    @property
    def n(self) -> int:
        return len(self.seg)


def amostrar_populacao(n: int, h: float, seed: int, p: ParamsSim, K: int = 11) -> Populacao:
    rng = np.random.default_rng(seed)
    seg = rng.choice(3, size=n, p=PROP_SEG)
    # L0: aprendiz → habilidade (duas camadas, para que haja aprendizes fortes com
    # lacunas pontuais e aprendizes fracos com uma ou outra habilidade já dominada).
    m_L0 = beta_media_dp(rng, medias_segmento(p.L0_seg, h)[seg], h * p.sd_L0_aprendiz)
    L0 = beta_media_dp(rng, m_L0[:, None], h * p.sd_L0_habilidade, (n, K))
    m_T = beta_media_dp(rng, medias_segmento(p.T_seg(), h)[seg], h * p.sd_T_aprendiz)
    T = beta_media_dp(rng, m_T[:, None], h * p.sd_T_habilidade, (n, K))
    S = beta_media_dp(rng, np.full(n, p.S_media), h * p.sd_S)
    G = beta_media_dp(rng, np.full(n, p.G_media), h * p.sd_G)
    F = beta_media_dp(rng, np.full(n, p.F_media), h * p.sd_F)
    return Populacao(
        h=h, seg=seg, L0=L0, T=T, S=S, G=G, F=F,
        U_resp=rng.random((n, H_PASSOS)), U_aband=rng.random((n, H_PASSOS)),
        U_sent=rng.random((n, H_PASSOS + 1, 2)), U_pre_obs=rng.random((n, K, ITENS_PRE_OBS)),
        U_teste=rng.random((n, 3, ITENS_TESTE_REAL)),
    )


def p_correto(L: np.ndarray, S: np.ndarray, G: np.ndarray) -> np.ndarray:
    return L * (1 - S) + (1 - L) * G


# ---------------------------------------------------------------------------
# Rastreador BKT do agente (S7): parâmetros POPULACIONAIS do segmento
# ---------------------------------------------------------------------------
class RastreadorBKT:
    """Estima a maestria a partir do pré-teste observado e das respostas (S7).

    Parâmetros POPULACIONAIS do segmento no nível h (como se o rastreador tivesse
    sido ajustado à população, por segmento) e a mesma estrutura do simulador
    (pré-requisitos e esquecimento). É uma hipótese FAVORÁVEL às políticas
    adaptativas: a estrutura está correta; só os parâmetros individuais são
    desconhecidos.

    modo = "beta" (principal): BKT com maestria latente CONTÍNUA, coerente com o
    simulador. A crença sobre L de cada habilidade é uma Beta(média m, variância v);
    a resposta tem verossimilhança linear em L, P(y = 1 | L) = G + (1 − S − G)L, e a
    posterior é reprojetada numa Beta por casamento de momentos. A variância a priori
    é a variância populacional de L0 dentro do segmento no nível h; em h = 0 ela é
    nula e o rastreador é exato (todos os aprendizes são iguais).

    modo = "binario" (sensibilidade): BKT clássico de Corbett & Anderson com estado
    latente binário. Num simulador de maestria contínua ele é superconfiante (uma
    resposta certa leva a estimativa de 0,5 a 0,82), e na execução exploratória as
    políticas aprendidas exploraram esse ruído. Mantido só como variante.
    """

    VAR_MIN = 1e-6

    def __init__(self, cat: Catalogo, p: ParamsSim, pop: Populacao, modo: str = "beta"):
        self.cat, self.p, self.modo = cat, p, modo
        h = pop.h
        self.l0 = medias_segmento(p.L0_seg, h)[pop.seg]          # (n,)
        self.T = medias_segmento(p.T_seg(), h)[pop.seg]          # (n,)
        self.S, self.G, self.F = p.S_media, p.G_media, p.F_media
        n, K = pop.n, cat.K
        m = np.broadcast_to(self.l0[:, None], (n, K)).copy()
        v0 = (h * p.sd_L0_aprendiz) ** 2 + (h * p.sd_L0_habilidade) ** 2
        v = np.minimum(np.full((n, K), v0), 0.9 * m * (1 - m))
        self.V = np.maximum(v, self.VAR_MIN)
        self.L = m
        # Pré-teste observado: ITENS_PRE_OBS itens por habilidade, sorteados da maestria inicial verdadeira.
        pc = p_correto(pop.L0, pop.S[:, None], pop.G[:, None])
        itens = (pop.U_pre_obs < pc[:, :, None])
        for j in range(ITENS_PRE_OBS):
            self.L, self.V = self._posterior(self.L, self.V, itens[:, :, j])
        self.itens_pre = itens
        self.L_ini = self.L.copy()

    def _posterior(self, m, v, y):
        S, G = self.S, self.G
        if self.modo == "binario":
            num = np.where(y, m * (1 - S), m * S)
            den = np.where(y, m * (1 - S) + (1 - m) * G, m * S + (1 - m) * (1 - G))
            return num / np.maximum(den, 1e-12), v
        c = 1 - S - G
        m2 = v + m * m                                   # E[L²]
        kappa = np.maximum(m * (1 - m) / np.maximum(v, 1e-12) - 1, 1e-6)
        a = m * kappa
        m3 = m2 * (a + 2) / (kappa + 2)                  # E[L³] da Beta
        p1 = G + c * m
        e1 = np.where(y, (G * m + c * m2) / p1, ((1 - G) * m - c * m2) / (1 - p1))
        e2 = np.where(y, (G * m2 + c * m3) / p1, ((1 - G) * m2 - c * m3) / (1 - p1))
        e1 = np.clip(e1, 1e-4, 1 - 1e-4)
        v2 = np.clip(e2 - e1 * e1, self.VAR_MIN, 0.9 * e1 * (1 - e1))
        return e1, v2

    def atualizar(self, linhas: np.ndarray, a: np.ndarray, y: np.ndarray):
        """Incorpora a resposta e o passo de aprendizado/esquecimento.

        Devolve (revisao, aprendizado) por aprendiz: `revisao` é a mudança de
        crença causada pela resposta (posterior − prior na habilidade praticada);
        `aprendizado` é a mudança prevista pelo modelo de transição (aprendizado na
        habilidade praticada menos esquecimento nas demais). Só o segundo é ganho
        de maestria; o primeiro é informação sobre o conhecimento prévio."""
        L, V = self.L, self.V
        r = np.arange(len(linhas))
        Lr, Vr = L[linhas], V[linhas]
        antes = Lr.sum(1)
        La, Va = self._posterior(Lr[r, a], Vr[r, a], y)
        revisao = La - Lr[r, a]
        mult = np.ones(len(linhas))
        for k, pre in enumerate(self.cat.prereq):
            if pre:
                sel = a == k
                if sel.any():
                    mult[sel] = np.where(Lr[sel][:, pre].mean(1) < self.p.limiar_prereq, self.p.mult_prereq, 1.0)
        Tef = self.T[linhas] * mult
        # Esquecimento (transformação afim de L): média puxada para a estimativa inicial.
        Lr = Lr - self.F * (Lr - self.L_ini[linhas])
        Vr = Vr * (1 - self.F) ** 2
        # Aprendizado na habilidade praticada: L ← L + (1 − L)T, afim em L.
        Lr[r, a] = La + (1 - La) * Tef
        Vr[r, a] = np.maximum(Va * (1 - Tef) ** 2, self.VAR_MIN)
        L[linhas], V[linhas] = Lr, Vr
        return revisao, Lr.sum(1) - antes - revisao


# ---------------------------------------------------------------------------
# Ambiente vetorizado
# ---------------------------------------------------------------------------
RECOMPENSA_CUSTO_PASSO = 0.01
RECOMPENSA_PENAL_ABANDONO = 1.0
DIM_BASE = 34                 # mesma família de atributos do estado de 34 dimensões (10/10)
DIM_OBS = DIM_BASE + 11 + 1   # + maestria estimada pelo BKT (11) + passo/H


@dataclass
class Registro:
    obs: List[np.ndarray] = field(default_factory=list)
    acao: List[np.ndarray] = field(default_factory=list)
    rec: List[np.ndarray] = field(default_factory=list)
    obs2: List[np.ndarray] = field(default_factory=list)
    fim: List[np.ndarray] = field(default_factory=list)
    curr: List[np.ndarray] = field(default_factory=list)   # ação do currículo no estado
    aprendiz: List[np.ndarray] = field(default_factory=list)
    passo: List[np.ndarray] = field(default_factory=list)

    def empilhar(self) -> Dict[str, np.ndarray]:
        if not self.obs:
            return {}
        return {
            "obs": np.concatenate(self.obs).astype(np.float32),
            "acao": np.concatenate(self.acao).astype(np.int64),
            "rec": np.concatenate(self.rec).astype(np.float32),
            "obs2": np.concatenate(self.obs2).astype(np.float32),
            "fim": np.concatenate(self.fim).astype(np.float32),
            "curr": np.concatenate(self.curr).astype(np.int64),
            "aprendiz": np.concatenate(self.aprendiz).astype(np.int64),
            "passo": np.concatenate(self.passo).astype(np.int64),
        }


class Ambiente:
    """Episódios de uma população inteira, em paralelo (vetorizado em numpy)."""

    def __init__(self, cat: Catalogo, p: ParamsSim, pop: Populacao, modo_kt: str = "beta"):
        self.cat, self.p, self.pop = cat, p, pop
        n, K = pop.n, cat.K
        self.n, self.K = n, K
        self.L = pop.L0.copy()
        self.t = 0
        self.ativo = np.ones(n, dtype=bool)
        self.abandonou = np.zeros(n, dtype=bool)
        self.passo_fim = np.full(n, H_PASSOS)
        self.passo_maestria = np.full(n, -1)
        self.n_decisoes = np.zeros(n, dtype=np.int64)
        self.n_diverg = np.zeros(n, dtype=np.int64)
        self.concluida = np.zeros((n, K), dtype=bool)
        self.n_expos = np.zeros((n, K), dtype=np.int64)
        self.ultima_vez = np.full((n, K), -1)
        self.tarefa_atual = np.full(n, -1)
        self.ultimo_acerto = np.zeros(n)
        self.pontos = np.zeros(n)
        self.Fr = np.zeros(n)
        self.Bo = np.zeros(n)
        self.n_frustr = np.zeros(n, dtype=np.int64)
        self.n_tedio = np.zeros(n, dtype=np.int64)
        self.kt = RastreadorBKT(cat, p, pop, modo_kt)
        self.sentimento = self._sentimento(np.arange(n), 0)
        self.historico: List[List[tuple]] = [[] for _ in range(n)]  # (tarefa, acerto) para o planejador LLM

    # -- regras observáveis -------------------------------------------------
    def acao_curriculo(self, linhas: Optional[np.ndarray] = None) -> np.ndarray:
        """Currículo fixo como função do estado observável: primeira tarefa do
        catálogo ainda não concluída; com o catálogo todo concluído, revisão da
        tarefa praticada há mais tempo (empate: ordem do catálogo). Para o próprio
        currículo isso dá 1, 2, …, 11 e depois 1, 2, 3, 4 (revisão em ordem), de
        modo que todas as políticas usam o mesmo orçamento de H = 15 decisões."""
        if linhas is None:
            linhas = np.arange(self.n)
        conc = self.concluida[linhas]
        tem_pendente = ~conc.all(1)
        primeira = np.argmin(conc, axis=1)
        revisao = np.argmin(self.ultima_vez[linhas], axis=1)
        return np.where(tem_pendente, primeira, revisao)

    def prereq_concluidos(self, linhas: np.ndarray) -> np.ndarray:
        """(len(linhas), K) bool: pré-requisitos do catálogo já concluídos (observável)."""
        conc = self.concluida[linhas]
        out = np.ones((len(linhas), self.K), dtype=bool)
        for k, pre in enumerate(self.cat.prereq):
            if pre:
                out[:, k] = conc[:, pre].all(1)
        return out

    def _prereq_verdadeiro(self, linhas: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Maestria média verdadeira dos pré-requisitos da tarefa a (1 se não há)."""
        m = np.ones(len(linhas))
        for k, pre in enumerate(self.cat.prereq):
            if pre:
                sel = a == k
                if sel.any():
                    m[sel] = self.L[linhas[sel]][:, pre].mean(1)
        return m

    def _sentimento(self, linhas: np.ndarray, t: int) -> np.ndarray:
        p = self.p
        s = np.where(self.Fr[linhas] >= 0.75, 2,                       # ansioso
                     np.where(self.Bo[linhas] >= 0.75, 3,               # cansado
                              np.where(self.ultimo_acerto[linhas] > 0, 1, 0)))  # motivado / neutro
        u = self.pop.U_sent[linhas, t]
        ruido = u[:, 0] < p.ruido_sentimento
        return np.where(ruido, np.minimum((u[:, 1] * 4).astype(int), 3), s)

    # -- observação -----------------------------------------------------------
    def observar(self, linhas: Optional[np.ndarray] = None) -> np.ndarray:
        if linhas is None:
            linhas = np.arange(self.n)
        m = len(linhas)
        cat = self.cat
        conc = self.concluida[linhas]
        nm = len(cat.missoes)
        miss_flags = np.stack([conc[:, cat.missao == j].all(1) for j in range(nm)], 1).astype(np.float64)
        cur = self.tarefa_atual[linhas]
        tem_cur = cur >= 0
        cur_m = np.where(tem_cur, cat.missao[np.maximum(cur, 0)], -1)
        prog = np.zeros(m)
        for j in range(nm):
            sel = cur_m == j
            if sel.any():
                prog[sel] = conc[sel][:, cat.missao == j].mean(1)
        oh = lambda idx, size: np.eye(size)[np.maximum(idx, 0)] * (idx >= 0)[:, None]  # noqa: E731
        kt = self.kt.L[linhas]
        obs = np.concatenate([
            np.stack([
                kt.mean(1),                                   # "literacia" ≈ maestria média estimada
                np.minimum(self.pontos[linhas] / 1000.0, 1.0),
                conc.mean(1),                                 # razão de tarefas concluídas
                miss_flags.mean(1),                           # razão de missões
                prog,                                         # progresso na missão atual
                self.ultimo_acerto[linhas],                   # (era cur_completed) acerto na última tarefa
                np.where(tem_cur, cat.terminal[np.maximum(cur, 0)], 0).astype(np.float64),
            ], 1),
            miss_flags,
            oh(self.pop.seg[linhas], 3),
            oh(self.sentimento[linhas], 5),
            oh(cur_m, nm),
            oh(cur, cat.K),
            kt,
            np.full((m, 1), self.t / H_PASSOS),
        ], 1)
        return obs.astype(np.float32)

    # -- dinâmica -------------------------------------------------------------
    def passo(self, acoes: np.ndarray, registro: Optional[Registro] = None) -> np.ndarray:
        """Aplica as ações (vetor de tamanho n; só os ativos contam). Devolve a
        recompensa de treino dos ativos (S8)."""
        p, t = self.p, self.t
        linhas = np.flatnonzero(self.ativo)
        if len(linhas) == 0:
            return np.zeros(0)
        a = np.asarray(acoes)[linhas].astype(np.int64)
        curr = self.acao_curriculo(linhas)
        self.n_decisoes[linhas] += 1
        self.n_diverg[linhas] += (a != curr)
        obs = self.observar(linhas) if registro is not None else None
        kt_antes = self.kt.L[linhas].sum(1)

        # 1) resposta (reflete o estado ANTES do aprendizado, como no BKT)
        L_a = self.L[linhas, a]
        pc = p_correto(L_a, self.pop.S[linhas], self.pop.G[linhas])
        y = self.pop.U_resp[linhas, t] < pc
        # 2) afeto: frustração (pré-req. não dominados) e tédio (habilidade já dominada)
        mpre = self._prereq_verdadeiro(linhas, a)
        frustr = mpre < p.limiar_prereq
        tedio = L_a > p.limiar_tedio
        self.Fr[linhas] = p.decaimento_afeto * self.Fr[linhas] + frustr
        self.Bo[linhas] = p.decaimento_afeto * self.Bo[linhas] + tedio
        self.n_frustr[linhas] += frustr
        self.n_tedio[linhas] += tedio
        # 3) aprendizado e esquecimento
        T_ef = self.pop.T[linhas, a] * np.where(frustr, p.mult_prereq, 1.0)
        Lr = self.L[linhas]
        Lr = Lr - self.pop.F[linhas, None] * (Lr - self.pop.L0[linhas])
        Lr[np.arange(len(linhas)), a] = L_a + (1 - L_a) * T_ef
        self.L[linhas] = Lr
        # 4) registros observáveis
        novo = ~self.concluida[linhas, a]
        self.pontos[linhas] += np.where(novo, self.cat.pontos[a], 0.0)
        self.concluida[linhas, a] = True
        self.n_expos[linhas, a] += 1
        self.ultima_vez[linhas, a] = t
        self.tarefa_atual[linhas] = a
        self.ultimo_acerto[linhas] = y.astype(np.float64)
        for i, ai, yi in zip(linhas, a, y):
            self.historico[i].append((int(ai), bool(yi)))
        revisao, aprendizado = self.kt.atualizar(linhas, a, y)
        # 5) abandono, maestria, horizonte
        p_ab = p.abandono_base + p.alfa_abandono * (self.Fr[linhas] + self.Bo[linhas])
        sai = self.pop.U_aband[linhas, t] < p_ab
        domina = (self.L[linhas] >= p.limiar_maestria).all(1)
        self.passo_maestria[linhas[domina & (self.passo_maestria[linhas] < 0)]] = t + 1
        self.t = t + 1
        fim = sai | domina | (self.t >= H_PASSOS)
        self.abandonou[linhas[sai]] = True
        self.passo_fim[linhas[fim]] = self.t
        self.ativo[linhas[fim]] = False
        self.sentimento[linhas] = self._sentimento(linhas, self.t)
        # 6) recompensa de treino (S8): ganho de maestria ESTIMADO − custo − abandono.
        # Principal: só o termo de aprendizado do BKT. Na execução exploratória, a
        # variante que somava a revisão de crença (ΔΣL̂ completo) foi explorada pelo
        # LinUCB: ele repetia habilidades cuja estimativa caíra por uma resposta errada
        # (regressão à média do ruído de medida), elevando o retorno de treino e
        # reduzindo o Hake. A variante fica disponível como análise de sensibilidade.
        penal = RECOMPENSA_CUSTO_PASSO + RECOMPENSA_PENAL_ABANDONO * sai
        rec = aprendizado - penal
        self.ultima_rec_alt = (self.kt.L[linhas].sum(1) - kt_antes) - penal
        if registro is not None:
            registro.obs.append(obs)
            registro.acao.append(a)
            registro.rec.append(rec)
            registro.obs2.append(self.observar(linhas))
            registro.fim.append(fim.astype(np.float64))
            registro.curr.append(curr)
            registro.aprendiz.append(linhas)
            registro.passo.append(np.full(len(linhas), t))
        return rec

    # -- desfecho ---------------------------------------------------------------
    def desfecho(self) -> Dict[str, np.ndarray]:
        """Pré, pós (relógio H) e retenção (H + 30) por aprendiz; Hake e teste de 10 itens."""
        pop, p = self.pop, self.p
        S, G = pop.S[:, None], pop.G[:, None]
        resto = (H_PASSOS - self.passo_fim)[:, None]
        dec = (1 - pop.F[:, None])
        L_pos = pop.L0 + (self.L - pop.L0) * dec ** resto
        L_ret = pop.L0 + (L_pos - pop.L0) * dec ** PASSOS_RETENCAO
        pre = p_correto(pop.L0, S, G).mean(1)
        pos = p_correto(L_pos, S, G).mean(1)
        ret = p_correto(L_ret, S, G).mean(1)
        hake = (pos - pre) / (1 - pre)
        hake_ret = (ret - pre) / (1 - pre)
        # Teste de 10 itens (instrumento do piloto): erro de medida binomial.
        acertos = [(pop.U_teste[:, j, :] < x[:, None]).sum(1) / ITENS_TESTE_REAL for j, x in enumerate((pre, pos, ret))]
        pre10, pos10 = acertos[0], acertos[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            hake10 = np.where(pre10 < 1, (pos10 - pre10) / (1 - pre10), np.nan)
        return {
            "pre": pre, "pos": pos, "ret": ret, "hake": hake, "hake_ret": hake_ret, "hake10": hake10,
            "abandono": self.abandonou.astype(np.float64), "passos": self.passo_fim.astype(np.float64),
            "passo_maestria": self.passo_maestria.astype(np.float64),
            "diverg": self.n_diverg / np.maximum(self.n_decisoes, 1), "n_decisoes": self.n_decisoes.astype(np.float64),
            "n_diverg": self.n_diverg.astype(np.float64),
            "frustr": self.n_frustr.astype(np.float64), "tedio": self.n_tedio.astype(np.float64),
            "seg": pop.seg,
        }

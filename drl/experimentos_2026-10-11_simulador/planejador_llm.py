"""P5: planejador baseado em modelo de linguagem, local, em CPU.

Escolha RESTRITA POR VEROSSIMILHANÇA: o modelo recebe a descrição textual do
estado e a lista de tarefas rotuladas de A a K; calcula-se, numa única passagem
direta, a log-probabilidade do próximo token para cada letra e escolhe-se a maior.
Não há geração livre, então a temperatura é irrelevante e a política é
determinística dado o texto do estado (o que a torna avaliável por OPE).

Por que letras e não números: o tokenizador do Qwen2.5 separa números em dígitos
("10" = "1" + "0"), e a comparação entre "1" e "10" exigiria verossimilhança de
sequências de comprimentos diferentes. As letras A–K são tokens únicos.

Por que a maestria entra discretizada (baixo/médio/alto): (1) é como um professor
descreveria o aprendiz; (2) estados textuais idênticos repetem-se, e a decisão é
memorizada por texto do prompt, o que reduz muito o custo em CPU.

Modelo: Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0); alternativa 0.5B se a latência
inviabilizar. Sem API paga nem serviço externo.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from politicas_sim import Politica
from simulador import H_PASSOS, ITENS_PRE_OBS, SEGMENTOS, Ambiente

LETRAS = "ABCDEFGHIJK"
ROT_SENT = {0: "neutro", 1: "motivado", 2: "ansioso", 3: "cansado", 4: "outro"}
ROT_SEG = {"iniciante": "iniciante", "intermediario": "intermediário", "avancado": "avançado"}

SISTEMA = (
    "Você é um tutor de educação financeira que planeja a trilha de um aprendiz adulto "
    "num programa por mensagens. A cada passo você escolhe UMA tarefa do catálogo. "
    "Objetivo: maximizar o aprendizado medido num pós-teste sobre todas as tarefas e "
    "evitar que o aprendiz desista. Tarefas cujos pré-requisitos o aprendiz ainda não domina "
    "rendem pouco e frustram; tarefas que ele já domina entediam. Uma tarefa pode ser repetida "
    "para revisão. Responda apenas com a letra da tarefa."
)


def _nivel(x: float) -> str:
    return "baixo" if x < 0.4 else ("médio" if x < 0.7 else "alto")


def descrever_estado(amb: Ambiente, i: int) -> str:
    """Texto do estado observável do aprendiz i (sem a maestria verdadeira)."""
    cat = amb.cat
    acertos_pre = int(amb.kt.itens_pre[i].sum())
    linhas = [
        f"Aprendiz: segmento {ROT_SEG[SEGMENTOS[amb.pop.seg[i]]]}; sentimento declarado: "
        f"{ROT_SENT[int(amb.sentimento[i])]}; passo {amb.t + 1} de {H_PASSOS}.",
        f"Pré-teste: {acertos_pre} acertos em {ITENS_PRE_OBS * cat.K} itens.",
        "Tarefas (letra) nome [missão] | pré-requisito | vezes feita | domínio estimado:",
    ]
    for k in range(cat.K):
        pre = ", ".join(cat.nomes[j] for j in cat.prereq[k]) or "nenhum"
        linhas.append(f"({LETRAS[k]}) {cat.nomes[k]} [{cat.missoes[cat.missao[k]].split('_')[0]}] | {pre} | "
                      f"{int(amb.n_expos[i, k])} | {_nivel(float(amb.kt.L[i, k]))}")
    hist = amb.historico[i][-3:]
    if hist:
        linhas.append("Últimas respostas: " + "; ".join(
            f"{cat.nomes[a]} ({'acertou' if y else 'errou'})" for a, y in hist) + ".")
    else:
        linhas.append("Últimas respostas: nenhuma (início do programa).")
    linhas.append("Qual é a próxima tarefa? Responda só com a letra.")
    return "\n".join(linhas)


def montar_mensagens(estado: str, exemplos: Sequence[tuple] = ()) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": SISTEMA}]
    for txt, letra in exemplos:
        msgs += [{"role": "user", "content": txt}, {"role": "assistant", "content": letra}]
    msgs.append({"role": "user", "content": estado})
    return msgs


class PontuadorHF:
    """Log-probabilidades do próximo token para as letras A–K (transformers, CPU)."""

    def __init__(self, nome_modelo: str, n_threads: int = 16, lote: int = 8, somente_local: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(n_threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(nome_modelo, local_files_only=somente_local)
        self.tok.padding_side = "left"   # o último token de cada linha fica alinhado à direita
        self.modelo = AutoModelForCausalLM.from_pretrained(nome_modelo, torch_dtype=torch.bfloat16,
                                                           local_files_only=somente_local)
        self.modelo.eval()
        self.lote = lote
        self.ids_letras = []
        for L in LETRAS:
            ids = self.tok.encode(L, add_special_tokens=False)
            assert len(ids) == 1, f"letra {L} não é token único"
            self.ids_letras.append(ids[0])

    def __call__(self, conversas: List[List[Dict[str, str]]]) -> np.ndarray:
        textos = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversas]
        out = []
        for j in range(0, len(textos), self.lote):
            enc = self.tok(textos[j:j + self.lote], return_tensors="pt", padding=True)
            with self.torch.no_grad():
                logits = self.modelo(**enc).logits[:, -1, :].float()
            lp = self.torch.log_softmax(logits, -1)[:, self.ids_letras]
            out.append(lp.numpy())
        return np.concatenate(out)


class PlanejadorLLM(Politica):
    """Política que delega a escolha ao pontuador (modelo de linguagem)."""
    deterministica = True

    def __init__(self, pontuador: Callable, exemplos: Sequence[tuple] = (), nome: str = "llm"):
        self.pontuador, self.exemplos, self.nome = pontuador, list(exemplos), nome
        self.cache: Dict[str, int] = {}
        self.n_decisoes = 0
        self.n_chamadas = 0
        self.seg_modelo = 0.0

    def agir(self, amb, linhas, obs):
        textos = [descrever_estado(amb, int(i)) for i in linhas]
        novos = sorted({t for t in textos if t not in self.cache})
        if novos:
            t0 = time.perf_counter()
            lp = self.pontuador([montar_mensagens(t, self.exemplos) for t in novos])
            self.seg_modelo += time.perf_counter() - t0
            self.n_chamadas += len(novos)
            for t, v in zip(novos, lp):
                self.cache[t] = int(np.argmax(v))
        self.n_decisoes += len(textos)
        return np.array([self.cache[t] for t in textos], dtype=np.int64)


def exemplos_few_shot(cat, p, amostrar, n_exemplos: int = 3, passos=(0, 5, 11)) -> List[tuple]:
    """Três exemplos do currículo fixo (S/P5): estados de um aprendiz simulado
    (h = 0,5, semente fixa 4242, fora das populações de avaliação) nos passos
    indicados, com a resposta que o currículo daria."""
    from politicas_sim import Curriculo
    amb = Ambiente(cat, p, amostrar(1, 0.5, 4242))
    ex, cur = [], Curriculo()
    while amb.ativo.any() and len(ex) < n_exemplos:
        a = cur.agir(amb, np.array([0]), None)
        if amb.t in passos:
            ex.append((descrever_estado(amb, 0), LETRAS[int(a[0])]))
        acoes = np.zeros(1, dtype=np.int64)
        acoes[0] = a[0]
        amb.passo(acoes)
    return ex


def carregar_pontuador(nomes=("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"), log=print):
    """Tenta os modelos em ordem. Devolve (pontuador, nome, erro)."""
    erros = []
    for nome in nomes:
        for local in (True, False):  # cache local primeiro; depois o Hugging Face Hub
            try:
                return PontuadorHF(nome, somente_local=local), nome, None
            except Exception as e:  # rede bloqueada, modelo ausente do cache, etc.
                if local:
                    continue
                erros.append(f"{nome}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
                log(f"[llm] falha ao carregar {nome}: {erros[-1]}")
    return None, None, " | ".join(erros)


def pontuador_falso(conversas):
    """Pontuador determinístico para o autoteste da mecânica (NÃO é um modelo):
    prefere a primeira tarefa com domínio estimado 'baixo'."""
    out = np.full((len(conversas), len(LETRAS)), -10.0)
    for j, c in enumerate(conversas):
        linhas = [l for l in c[-1]["content"].split("\n") if l.startswith("(")]
        for k, l in enumerate(linhas):
            if l.endswith("baixo"):
                out[j, k] = -1.0
                break
        else:
            out[j, 0] = -1.0
    return out


if __name__ == "__main__":
    # Autoteste da mecânica (prompt, memorização, escolha restrita) sem baixar modelo.
    import sys
    from dataclasses import replace

    from politicas_sim import simular
    from simulador import ParamsSim, amostrar_populacao, carregar_catalogo
    sys.stdout.reconfigure(encoding="utf-8")
    cat, p = carregar_catalogo(), replace(ParamsSim(), c_T=0.6586, alfa_abandono=0.0108)
    am = lambda n, h, s: amostrar_populacao(n, h, s, p)  # noqa: E731
    ex = exemplos_few_shot(cat, p, am)
    print(montar_mensagens(descrever_estado(Ambiente(cat, p, am(1, 0.5, 1)), 0), ex)[-1]["content"])
    pol = PlanejadorLLM(pontuador_falso, ex, "falso")
    d = simular(Ambiente(cat, p, am(200, 1.0, 3)), pol)
    print("Hake", d["hake"].mean(), "decisões", pol.n_decisoes, "chamadas", pol.n_chamadas)

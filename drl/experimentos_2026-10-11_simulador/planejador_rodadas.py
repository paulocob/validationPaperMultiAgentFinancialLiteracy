"""Harness por rodadas para o planejador LLM executado por um agente externo (P5).

Decisão do autor (11/10/2026): o planejador é um subagente que lê arquivos de
estado e escreve decisões; não há API nem modelo local. Este script guarda o
estado do simulador em disco entre as rodadas, de modo que o agente só vê o que
o sistema real veria (o mesmo texto de `planejador_llm.descrever_estado`).

Comandos:
    python planejador_rodadas.py init --h 1 --n 200 --seed 42 --dir rodadas_h1/
    python planejador_rodadas.py step --dir rodadas_h1/ --decisoes rodadas_h1/decisoes_00.json
    python planejador_rodadas.py finalize --dir rodadas_h1/
    # só para teste do harness (usa o estado interno; NÃO é o planejador):
    python planejador_rodadas.py decidir-teste --dir rodadas_h1/ --modo curriculo|aleatorio

Formato da decisão: {"a000": "C", "a001": "A", ...} com uma letra A–K por
aprendiz ativo na rodada. Letra ausente ou inválida → aplica-se a tarefa do
currículo e a ocorrência é contada como inválida (registrada no resultado).

Pareamento: a população é `amostrar_populacao(n, h, seed)` com os parâmetros
calibrados de resultados.json; `finalize` roda o currículo nos MESMOS aprendizes,
com os mesmos números aleatórios comuns, e calcula a diferença pareada.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import politicas_sim as PS  # noqa: E402
from planejador_llm import LETRAS, SISTEMA, descrever_estado  # noqa: E402
from simulador import H_PASSOS, Ambiente, ParamsSim, amostrar_populacao, carregar_catalogo  # noqa: E402

ESTADO = "estado.pkl"
META = "meta.json"
N_BOOT = 2000

INSTRUCOES = """# Instruções para o planejador

{sistema}

## Como trabalhar

A cada rodada `k`, leia `rodada_<k>.json` nesta pasta. O arquivo contém uma lista
de aprendizes ativos, cada um com `id` e `estado` (descrição textual do que o
sistema sabe sobre o aprendiz). Para cada aprendiz, escolha UMA tarefa pela letra
(A a K) da lista que aparece no estado.

Escreva `decisoes_<k>.json` (mesmo `k`, com dois dígitos) com um objeto JSON que
mapeia cada `id` a uma letra, por exemplo:

```json
{{"a000": "A", "a001": "F", "a002": "C"}}
```

Regras:
- Inclua todos os `id` da rodada; cada valor é uma única letra maiúscula de A a K.
- Decida cada aprendiz apenas com o texto do seu estado.
- Não escreva explicações no arquivo de decisões.

Depois de escrever as decisões, o coordenador roda o passo seguinte, que gera
`rodada_<k+1>.json`. Quando aparecer o arquivo `FIM`, não há mais rodadas.
"""


def _params() -> ParamsSim:
    """Parâmetros calibrados da execução completa (resultados.json)."""
    R = json.loads((AQUI / "resultados.json").read_text(encoding="utf-8"))
    pd = R["params"]
    return replace(ParamsSim(), **{k: (tuple(v) if isinstance(v, list) else v) for k, v in pd.items()})


def _ids(n):
    return [f"a{i:03d}" for i in range(n)]


def _salvar(d: Path, amb, meta):
    with (d / ESTADO).open("wb") as f:
        pickle.dump(amb, f)
    (d / META).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def _carregar(d: Path):
    with (d / ESTADO).open("rb") as f:
        amb = pickle.load(f)
    return amb, json.loads((d / META).read_text(encoding="utf-8"))


def _escrever_rodada(d: Path, amb, k: int):
    ids = _ids(amb.n)
    ativos = np.flatnonzero(amb.ativo)
    if len(ativos) == 0 or amb.t >= H_PASSOS:
        (d / "FIM").write_text(f"fim após {amb.t} passos\n", encoding="utf-8")
        return None
    rod = {"rodada": k, "passo": int(amb.t) + 1, "H": H_PASSOS,
           "aprendizes": [{"id": ids[i], "estado": descrever_estado(amb, int(i))} for i in ativos]}
    p = d / f"rodada_{k:02d}.json"
    p.write_text(json.dumps(rod, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def cmd_init(a):
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("rodada_*.json"):
        f.unlink()
    for f in ("FIM",):
        (d / f).unlink(missing_ok=True)
    p = _params()
    pop = amostrar_populacao(a.n, a.h, a.seed, p)
    amb = Ambiente(carregar_catalogo(), p, pop)
    meta = {"h": a.h, "n": a.n, "seed": a.seed, "k": 0, "invalidas": 0, "decisoes": 0,
            "invalidas_por_rodada": []}
    _salvar(d, amb, meta)
    (d / "INSTRUCOES_PLANEJADOR.md").write_text(INSTRUCOES.format(sistema=SISTEMA), encoding="utf-8")
    print(_escrever_rodada(d, amb, 0))


def cmd_step(a):
    d = Path(a.dir)
    if (d / "FIM").exists():
        sys.exit("FIM já atingido")
    amb, meta = _carregar(d)
    k = meta["k"]
    dec = json.loads(Path(a.decisoes).read_text(encoding="utf-8"))
    ids = _ids(amb.n)
    ativos = np.flatnonzero(amb.ativo)
    curr = amb.acao_curriculo(ativos)
    acoes = np.zeros(amb.n, dtype=np.int64)
    inval = 0
    for i, c in zip(ativos, curr):
        letra = dec.get(ids[i])
        if isinstance(letra, str) and letra.strip().upper() in LETRAS and len(letra.strip()) == 1:
            acoes[i] = LETRAS.index(letra.strip().upper())
        else:
            acoes[i] = c  # fallback: currículo
            inval += 1
    extras = len(set(dec) - {ids[i] for i in ativos})
    amb.passo(acoes)
    meta["k"] = k + 1
    meta["invalidas"] += inval
    meta["decisoes"] += len(ativos)
    meta["invalidas_por_rodada"].append({"rodada": k, "ativos": int(len(ativos)), "invalidas": inval,
                                         "ids_extras_ignorados": extras})
    _salvar(d, amb, meta)
    out = _escrever_rodada(d, amb, k + 1)
    print(f"rodada {k}: {len(ativos)} decisões, {inval} inválidas; ativos agora {int(amb.ativo.sum())}; "
          + (f"próxima {out}" if out else "FIM"))


def _boot(x, rng):
    m = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def cmd_finalize(a):
    d = Path(a.dir)
    if not (d / "FIM").exists():
        sys.exit("ainda há rodadas pendentes (sem FIM)")
    amb, meta = _carregar(d)
    dl = amb.desfecho()
    p = amb.p
    pop = amostrar_populacao(meta["n"], meta["h"], meta["seed"], p)
    dc = PS.simular(Ambiente(amb.cat, p, pop), PS.Curriculo())
    rng = np.random.default_rng(0)
    dif, difr = dl["hake"] - dc["hake"], dl["hake_ret"] - dc["hake_ret"]
    sd = float(np.sqrt((dl["hake"].var(ddof=1) + dc["hake"].var(ddof=1)) / 2))
    res = {
        "h": meta["h"], "n": meta["n"], "seed": meta["seed"], "passos": int(amb.t),
        "decisoes": meta["decisoes"], "invalidas": meta["invalidas"],
        "frac_invalidas": meta["invalidas"] / max(1, meta["decisoes"]),
        "invalidas_por_rodada": meta["invalidas_por_rodada"],
        "llm": {"hake": float(dl["hake"].mean()), "hake_ic": _boot(dl["hake"], rng),
                "hake_ret": float(dl["hake_ret"].mean()), "hake_ret_ic": _boot(dl["hake_ret"], rng),
                "abandono": float(dl["abandono"].mean()), "abandono_ic": _boot(dl["abandono"], rng),
                "diverg": float(dl["n_diverg"].sum() / max(1, dl["n_decisoes"].sum()))},
        "curriculo": {"hake": float(dc["hake"].mean()), "hake_ret": float(dc["hake_ret"].mean()),
                      "abandono": float(dc["abandono"].mean())},
        "vs_curriculo": {"dif_hake": float(dif.mean()), "dif_hake_ic": _boot(dif, rng),
                         "dif_hake_ret": float(difr.mean()), "dif_hake_ret_ic": _boot(difr, rng),
                         "dif_abandono": float((dl["abandono"] - dc["abandono"]).mean()),
                         "d_cohen": float(dif.mean() / sd) if sd > 0 else float("nan")},
        "nota": "Diferença pareada nos mesmos aprendizes e números aleatórios comuns; IC por bootstrap de aprendizes.",
    }
    h = ("%g" % meta["h"]).replace(".", "")
    out = d / f"resultado_llm_h{h}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("llm", "curriculo", "vs_curriculo", "frac_invalidas")}, ensure_ascii=False))


def cmd_decidir_teste(a):
    """Só para testar o harness: escreve decisões do currículo ou aleatórias
    usando o estado interno. Não é o planejador."""
    d = Path(a.dir)
    amb, meta = _carregar(d)
    ids = _ids(amb.n)
    ativos = np.flatnonzero(amb.ativo)
    if a.modo == "curriculo":
        acts = amb.acao_curriculo(ativos)
    else:
        rng = np.random.default_rng(1000 + meta["k"])
        acts = rng.integers(0, len(LETRAS), len(ativos))
    dec = {ids[i]: LETRAS[int(x)] for i, x in zip(ativos, acts)}
    p = d / f"decisoes_{meta['k']:02d}.json"
    p.write_text(json.dumps(dec), encoding="utf-8")
    print(p)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    i = sp.add_parser("init")
    i.add_argument("--h", type=float, required=True)
    i.add_argument("--n", type=int, default=200)
    i.add_argument("--seed", type=int, default=42)
    i.add_argument("--dir", required=True)
    s = sp.add_parser("step")
    s.add_argument("--dir", required=True)
    s.add_argument("--decisoes", required=True)
    f = sp.add_parser("finalize")
    f.add_argument("--dir", required=True)
    t = sp.add_parser("decidir-teste")
    t.add_argument("--dir", required=True)
    t.add_argument("--modo", choices=["curriculo", "aleatorio"], required=True)
    a = ap.parse_args()
    {"init": cmd_init, "step": cmd_step, "finalize": cmd_finalize, "decidir-teste": cmd_decidir_teste}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Converte rodada_<k>.json em rodada_<k>.txt, um aprendiz por bloco.

Por que existe: o JSON de uma rodada tem centenas de KB numa única linha, o que
obriga o planejador (um subagente) a escrever código para lê-lo. O texto gerado
aqui tem exatamente o mesmo conteúdo, sem resumo nem reordenação, para que a
decisão seja tomada lendo o estado, e não um resumo dele.

Uso: python rodada_para_texto.py <pasta> <k>
"""
import json
import sys
from pathlib import Path


def main() -> None:
    pasta, k = Path(sys.argv[1]), int(sys.argv[2])
    origem = pasta / f"rodada_{k:02d}.json"
    destino = pasta / f"rodada_{k:02d}.txt"
    rodada = json.loads(origem.read_text(encoding="utf-8"))
    aprendizes = rodada["aprendizes"]
    cabecalho = f"Rodada {rodada['rodada']} | passo {rodada['passo']} de {rodada['H']} | {len(aprendizes)} aprendizes ativos\n"
    blocos = [f"=== {a['id']} ===\n{a['estado']}\n" for a in aprendizes]
    destino.write_text(cabecalho + "\n" + "\n".join(blocos), encoding="utf-8")
    print(f"{destino}: {len(aprendizes)} aprendizes")


if __name__ == "__main__":
    main()

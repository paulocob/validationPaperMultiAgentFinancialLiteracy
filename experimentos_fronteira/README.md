# experimentos_fronteira/ — Auditoria de equidade e protocolos do classificador de fala

Origem: pasta `Experimentos de Fronteira/` do repositório de escrita da tese. Entraram só os itens ligados a resultados da tese ou ao classificador de variantes.

| Caminho | Situação | Tese |
|---|---|---|
| `trilha5_avaliacao/e5_3_fairness.py` + `resultados/e5_3_fairness.json` | **Executado.** Auditoria de equidade entre países: DPD e razão de impacto disparatado da recuperação (taxa de acerto em top-5 por país) e acurácia do classificador de variantes por grupo geográfico | Cap. 4, `sec:q_equidade` (DPD = 0,003 para OpenCLIP + LoRA+CL) |
| `trilha4_fala/e4_*.py` | **Protocolos de experimentos planejados** (ablação, multissemente, *data augmentation*, robustez a Opus, MMS/Whisper, t-SNE/UMAP). As saídas registradas na origem eram só de *dry-run* e não foram incluídas | Cap. 4, `sec:q_variantes`; plano de trabalho |
| `common/` | Biblioteca compartilhada (carregadores de dados, estatística, *logging*, verificação de *hardware*) | — |

## Como reproduzir a auditoria de equidade

```bash
cd experimentos_fronteira
python trilha5_avaliacao/e5_3_fairness.py
```

A parte da recuperação usa `rag/embeddings/results/rag_retrieval_results.csv` e reproduz os valores de `resultados/e5_3_fairness.json` em segundos. A parte do classificador de variantes precisa das predições de teste do repositório `sbrt2026-lusophone-variants` (ainda não depositado; ver `fala/README.md`); indique a cópia local com a variável de ambiente `VARIANTES_DIR`.

Os scripts de `trilha4_fala/` também dependem desse repositório e de *hardware* com GPU ou Apple MPS (estimativas de 1 h a 18 h por experimento, registradas nas *docstrings*).

## Alterações em relação à origem

- `common/data_loaders.py`: os caminhos passaram a apontar para as pastas deste repositório (`drl/data`, `rag/dataset`, `rag/embeddings/results`), o diretório do classificador passou a aceitar `VARIANTES_DIR`, e as leituras passaram a declarar UTF-8 (na origem, a leitura do corpus falhava no Windows).

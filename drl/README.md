# drl/ — Módulo de decisão (aprendizado por reforço profundo *offline*)

Código, dados sintéticos e resultados dos experimentos de DRL da tese. Origem: pasta `Artigo DRL/treino_DRL/` do repositório de escrita da tese.

| Caminho | Conteúdo | Tese |
|---|---|---|
| `data/` | Conjunto sintético: `interactions.jsonl` (2.000 transições), `users.jsonl` (25 perfis simulados), `tasks.json` (catálogo de tarefas), `splits.json` (partição original por interação, 1.581/206/213) | Cap. 3, seção de abordagem metodológica; Apêndice E de `apendices/`, “Características do conjunto sintético” |
| `experimentos_2026-10-10/` | **Reexecução corrigida** (partição por perfil, validação cruzada de 5 *folds*, seleção de λ_reg na validação, avaliação *off-policy* SNIS/DR, experimento cultural). `run_all.py` reproduz tudo; `RESULTADOS.md` e `resultados.json` são as saídas usadas na tese | Cap. 4, seções `sec:q_drl`, `sec:q_regularizacao`, `sec:q_cultural`; Apêndice E de `apendices/`, “Reprodutibilidade” |
| `experimentos_2026-10-11_simulador/` | Simulador cognitivo (BKT estendido), comparação de políticas, sensibilidade e rodadas do planejador LLM (`rodadas_h05/`, `rodadas_h1/`) | Cap. 4, `sec:q_simulador`; Apêndice E de `apendices/`, “Simulador Cognitivo” |
| `experimentos_2026-08-29/` | Primeira execução (agosto de 2026), preservada **sem alteração** para auditoria. Tem os quatro problemas descritos na “Nota histórica” do Apêndice E de `apendices/`; os números dela **não** são resultados da tese | Apêndice E de `apendices/`, “Nota histórica: a primeira execução” |
| `train_*.py`, `eval_*.py`, `train_utils.py`, `pipeline.py`, `cv_pipeline.py`, `hparam_search.py`, `EXPERIMENTS.md`, `README_pipeline_original.md` | *Pipeline* original (jan.–ago. 2026). `train_utils.py` é importado pela reexecução de 10/10 (codificador de estado) | Histórico |

## Como reproduzir

Ambiente das execuções: Python 3.12.10, CPU, versões em `requirements-experimentos.txt`.

```bash
cd drl
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-experimentos.txt

# Experimentos de DRL (tabelas e 3 figuras do cap. 4)
cd experimentos_2026-10-10
python run_all.py --quick   # fumaça: ~1 min em CPU (1 semente, 2 folds, grade reduzida; não gera figuras)
python run_all.py           # completo: 7,4 min em CPU na execução registrada

# Simulador cognitivo
cd ../experimentos_2026-10-11_simulador
python run_all.py --quick   # fumaça: ~3 min em CPU
python run_all.py           # completo: 17,5 min em CPU na execução registrada
python sensibilidade.py     # análise de sensibilidade (ver NOTAS_SENSIBILIDADE.md)
```

Sementes: 42 a 46 no *split* fixo e 42 na validação cruzada; simulador com 5 sementes de população e semente 42 nas rodadas do planejador LLM. As execuções completas sobrescrevem `resultados.json` e `RESULTADOS.md` da pasta; compare com as versões versionadas. As figuras vão para `./figuras/` de cada pasta (ou para o diretório da variável de ambiente `FIG_DIR`).

## Planejador LLM por rodadas

As pastas `rodadas_h05/` e `rodadas_h1/` registram a execução de 11/10/2026: em cada rodada, um subagente Claude leu `INSTRUCOES_PLANEJADOR.md` e a visão em texto (`rodada_<k>.txt`, gerada de `rodada_<k>.json`) e escreveu `decisoes_<k>.json`. O resultado pareado com o currículo está em `resultado_llm_h*.json`. Para refazer a mecânica com outras decisões:

```bash
python planejador_rodadas.py init --h 1 --n 200 --seed 42 --dir nova_rodada/
python planejador_rodadas.py step --dir nova_rodada/ --decisoes nova_rodada/decisoes_00.json
python planejador_rodadas.py finalize --dir nova_rodada/
```

## Alterações em relação à origem

- `figuras.py` e `figuras_sim.py`: o destino das figuras deixou de ser a pasta do documento da tese e passou a ser `./figuras/` (ou `FIG_DIR`). As docstrings de `run_all.py` e `relatorio_sim.py` foram ajustadas da mesma forma.
- `run_all.log` e `sensibilidade.log`: o caminho absoluto local foi substituído por `<raiz-da-tese>`.
- Nenhum número, semente ou parâmetro foi alterado.

## O que ficou de fora

- `models/` da origem (6,1 MB): *checkpoints* e *logs* do *pipeline* original, obsoletos depois da correção de outubro. `pipeline.py` os regenera.
- `rodadas_h*/estado.pkl` (0,4 MB cada): estado serializado em *pickle* do simulador entre rodadas. O formato *pickle* executa código ao ser carregado e não deve ser distribuído; `planejador_rodadas.py init` com a mesma semente recria o estado inicial.
- `__pycache__/`.
- `SEARCH_STRINGS.md` e `TASKS.md` da origem: notas de trabalho do artigo de DRL, sem relação com os resultados da tese (as *strings* da revisão sistemática estão em `revisao/`).
- O programa gerador do conjunto sintético original **não foi preservado** (registrado na tese, seção `subsec:disponibilidade`).

## Limite de interpretação

Os dados são sintéticos. Os resultados sustentam a viabilidade técnica da formalização e a coerência do protocolo, e não a eficácia pedagógica junto a aprendizes reais. Nenhuma interação dos estudos-piloto foi usada no treinamento ou na avaliação.

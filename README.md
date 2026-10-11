# Artefatos de validação e replicabilidade — Educação financeira transcultural com sistema multiagente, IA generativa e DRL

> **English summary.** This repository holds the validation and replication artifacts of the doctoral research *Multi-Agent System with Generative AI and Deep Reinforcement Learning for Transcultural Financial Education in Lusophone Countries* (Federal University of Campina Grande, Brazil). It contains: the 60 search strings of the systematic review; the full documentation of the six educational missions and of their Angolan and Portuguese cultural variants; the code, synthetic transitions, configurations and seeds of the offline deep reinforcement learning experiments and of the cognitive simulator; the FinEduBench-Luso corpus (999 question–answer pairs for Brazil, Portugal and Angola), the retrieval evaluation code and the LoRA adapter weights of the retriever; the configuration and results of the speech-based variant classifier (no audio); the equity audit; and the aggregated results of the two pilot studies (no individual data). Code is MIT-licensed; data and documents are CC BY 4.0. Each folder has a README (in Portuguese) with the source, the reproduction commands and what was left out and why.

Repositório citado na tese de doutorado *Sistema Multi-Agente com IA Generativa e Aprendizado por Reforço Profundo para Educação Financeira Transcultural em Países Lusófonos* (Paulo César Oliveira Brito, Programa de Pós-Graduação em Ciência da Computação, Universidade Federal de Campina Grande; orientador: José Antão Beltrão Moura), seção “Disponibilidade de Dados e Código” (`subsec:disponibilidade`).

## Conteúdo e ligação com a tese

| Pasta | O que contém | Onde a tese usa |
|---|---|---|
| [`revisao/`](revisao/) | As 60 *strings* de busca (CSV e LaTeX) e o protocolo resumido | Cap. 2, Protocolo Metodológico da Revisão |
| [`missoes/`](missoes/) | Estrutura das seis missões e variantes culturais (Brasil, Angola, Portugal), com vereditos de adaptação e derivação de `f_cult` | Cap. 3; Apêndices C e D |
| [`pseudocodigo/`](pseudocodigo/) | Pseudocódigo do *pipeline* multiagente, do cálculo da recompensa e da progressão das missões | Cap. 3 (agentes de PLN; função de recompensa) |
| [`drl/`](drl/) | Conjunto sintético de 2.000 transições, código de treino e avaliação, reexecução corrigida (10/10/2026), simulador cognitivo e planejador LLM (11/10/2026), primeira execução preservada | Cap. 4, seções de DRL, regularização, componente cultural e simulador; Apêndice E |
| [`rag/`](rag/) | Corpus FinEduBench-Luso (999 pares), código de avaliação do RAG, adaptador LoRA do recuperador e resultados | Cap. 4, RAG multimodal e auditoria de equidade |
| [`fala/`](fala/) | Configuração, composição dos dados e resultados do classificador de 17 variantes; modelo do manifesto dos clipes | Cap. 4, Identificação de Variantes Lusófonas por Fala |
| [`experimentos_fronteira/`](experimentos_fronteira/) | Auditoria de equidade entre países (executada) e protocolos dos experimentos planejados do classificador de fala | Cap. 4, auditoria de equidade |
| [`pilotos/`](pilotos/) | Resultados **agregados** dos pilotos do Brasil e de Angola (CSV) | Cap. 4, Validação Empírica com Usuários |
| [`artigo/`](artigo/) | PDF do artigo CSEDU 2026 e as retificações registradas na tese | Apêndice F |

## Como reproduzir

Requisitos: Python 3.12 (as execuções registradas usaram 3.12.10 em CPU). Cada pasta tem as dependências e os comandos completos no seu README.

| Experimento | Comando | Tempo registrado |
|---|---|---|
| DRL: tabelas e figuras do cap. 4 | `cd drl/experimentos_2026-10-10 && python run_all.py` | 7,4 min (CPU) |
| DRL: fumaça | `python run_all.py --quick` | ~1 min (CPU) |
| Simulador cognitivo | `cd drl/experimentos_2026-10-11_simulador && python run_all.py` | 17,5 min (CPU) |
| Simulador: fumaça | `python run_all.py --quick` | ~3 min (CPU) |
| Recuperação RAG (999 consultas) | `cd rag && python -m embeddings.rag_evaluation` | segundos a ~1 min por modelo, mais o *download* dos modelos |
| Auditoria de equidade (parte RAG) | `cd experimentos_fronteira && python trilha5_avaliacao/e5_3_fairness.py` | segundos (a parte de fala exige o repositório do classificador) |

```bash
git clone https://github.com/paulocob/validationPaperMultiAgentFinancialLiteracy.git
cd validationPaperMultiAgentFinancialLiteracy/drl
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-experimentos.txt
cd experimentos_2026-10-10 && python run_all.py --quick
```

Sementes: 42 a 46 (DRL, *split* fixo e classificador de fala), 42 (validação cruzada, LoRA, partição do conjunto de fala, planejador LLM). Os scripts de geração com LLM do RAG exigem `OPENAI_API_KEY` no ambiente; nenhuma chave está no repositório.

## O que não está incluído, e por quê

| Item | Motivo | Situação |
|---|---|---|
| Código do sistema integrado (*backend* do tutor no WhatsApp) | Em integração com a política DQN regularizada | Depósito previsto na A12 (novembro de 2026) |
| Programa gerador do conjunto sintético original | Não foi preservado | Irrecuperável; o simulador cognitivo (`drl/experimentos_2026-10-11_simulador/`) gera trajetórias novas |
| Respostas individuais do questionário de percepção e escores individuais do teste de conhecimento dos pilotos | Não foram preservados; a análise guardou só médias e percentuais | Irrecuperável; agregados em `pilotos/` |
| Interações, transcrições, áudios e qualquer dado pessoal dos participantes; documentos do comitê de ética e TCLE; excertos pseudonimizados do Apêndice G | Proteção de dados (LGPD) e termos do consentimento | Não serão depositados |
| Áudio do conjunto de fala (Common Voice e telejornais) | Direitos de terceiros; uso só para pesquisa não comercial | Não será depositado; ver o manifesto em `fala/` |
| Código, regras *regex*, predições e manifesto por clipe do classificador de fala | Ficam no repositório `sbrt2026-lusophone-variants`, não localizado nesta montagem | Pendente (A12) |
| Pesos do classificador de variantes (~380 MB) | Acima do limite de 50 MB por arquivo | Disponíveis mediante solicitação; a depositar em *release* ou no Zenodo |
| Pesos da ablação do LoRA (65 MB, 13 configurações) | Volume; resultados já tabulados | Disponíveis mediante solicitação; regeneráveis por `rag/embeddings/ablation_study.py` |
| *Checkpoints* do *pipeline* original de DRL (`models/`, 6,1 MB) | Obsoletos depois da correção de outubro de 2026 | Regeneráveis por `drl/pipeline.py` |
| Estado serializado do simulador (`estado.pkl`) | Formato *pickle* executa código ao ser carregado | Regenerável por `planejador_rodadas.py init` |

## Licenças

- Código: [MIT](LICENSE).
- Dados, documentos, figuras e pesos do adaptador LoRA: [CC BY 4.0](LICENSE-DATA.md).
- Exceção: o PDF em `artigo/` segue a política da editora (SCITEPRESS).

## Como citar

Tese (qualificação de doutorado):

```bibtex
@phdthesis{brito2026tese,
  author = {Brito, Paulo César Oliveira},
  title  = {Sistema Multi-Agente com IA Generativa e Aprendizado por Reforço Profundo para Educação Financeira Transcultural em Países Lusófonos},
  school = {Universidade Federal de Campina Grande},
  type   = {Qualificação de Doutorado em Ciência da Computação},
  address = {Campina Grande, PB},
  year   = {2026},
  note   = {Orientador: José Antão Beltrão Moura. Artefatos: https://github.com/paulocob/validationPaperMultiAgentFinancialLiteracy}
}
```

Artigo:

```bibtex
@inproceedings{brito2026,
  author    = {Brito, Paulo Cesar Oliveira and Hon{\'o}rio, Joaquim and Moura, Jos{\'e} Ant{\~a}o Beltr{\~a}o and Jones, Christian and Terton, Uwe},
  title     = {Multi-Agent System with Generative {AI} and Deep Reinforcement Learning for Adaptive Financial Education},
  booktitle = {Proceedings of the 18th International Conference on Computer Supported Education (CSEDU 2026)},
  volume    = {2},
  pages     = {1313--1324},
  publisher = {SCITEPRESS},
  address   = {Benidorm, Spain},
  year      = {2026},
  doi       = {10.5220/0015042800004021}
}
```

O arquivo [`CITATION.cff`](CITATION.cff) permite citar o repositório pelo botão “Cite this repository” do GitHub.

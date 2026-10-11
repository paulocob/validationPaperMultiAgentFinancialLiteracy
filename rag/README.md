# rag/ — FinEduBench-Luso e recuperação multimodal

Corpus de avaliação transcultural, código de avaliação do RAG e pesos do adaptador LoRA do recuperador. Origem: pasta `Pesquisa Fronteira RAG Multimodal/finedu_bench_luso/` do repositório de escrita da tese.

Seções da tese: cap. 4, `sec:q_rag` (RAG multimodal) e `sec:q_equidade` (auditoria de equidade); cap. 3, seção de avaliação da geração.

## Corpus: 999 pares pergunta–resposta

| Arquivo | Pares |
|---|---|
| `dataset/BR/br_seed.jsonl` | 333 (Brasil) |
| `dataset/PT/pt_seed.jsonl` | 333 (Portugal) |
| `dataset/AO/ao_seed.jsonl` | 333 (Angola) |

Cada linha é um objeto JSON com os campos `id`, `question`, `answer`, `country`, `topic` (budget, savings, investment, credit, debt, cultural), `difficulty` (basic, intermediate, advanced), `modalities`, `image_description`, `audio_transcript`, `cultural_notes` e `sources`. O esquema está em `dataset/schema.py`.

- **Não há imagens nem áudio.** A modalidade de imagem é representada pela descrição textual (`image_description`) e a de áudio pela transcrição simulada em registro oral (`audio_transcript`). Nenhum dado provém de pessoas reais.
- `dataset/generator.py` e `dataset/expansions/*.py` são as fontes a partir das quais os arquivos `.jsonl` foram montados; `dataset/stats.py` produz as estatísticas do corpus.

## Adaptador LoRA do recuperador

| Pasta | Configuração | Uso |
|---|---|---|
| `embeddings/lora_output_cl/` | LoRA + alinhamento interlíngue (*cross-lingual*, peso 0,3) — **configuração reportada na tese** | Todos os experimentos de RAG |
| `embeddings/lora_output/` | LoRA sem alinhamento interlíngue | Comparação em `clip_compare.py` |

Base: OpenCLIP ViT-B-32, pesos `laion2b_s34b_b79k`; posto 4, α = 8, 20 épocas com parada antecipada (15 executadas), lote 32, taxa 1e-4, semente 42 (`config.json`). Os arquivos `best_lora.pt` (2,2 MB) contêm só os parâmetros treináveis (LoRA e projeção de texto), salvos com `torch.save` de um dicionário de tensores. Carregue com `torch.load(caminho, weights_only=True)`.

## Como reproduzir

```bash
cd rag
pip install -r requirements.txt
python -m embeddings.rag_evaluation          # recuperação: P@k, R@k, MRR por modelo e por país
python -m embeddings.rag_cultural_experiment # efeito do filtro por país
python -m embeddings.rag_bias_mitigation     # auditoria de equidade (usa a API da OpenAI)
bash run_experiments.sh train                # retreina os dois adaptadores LoRA
bash run_experiments.sh compare              # Baseline vs LoRA vs LoRA+CL
```

- Os modelos de base (OpenCLIP, XLM-R CLIP, `multilingual-e5-large`) são baixados na primeira execução.
- `rag_generation.py`, `rag_generation_reranker.py`, `rag_bias_mitigation.py` e `expand_audio_transcripts.py` chamam `gpt-4o-mini` e exigem a variável de ambiente `OPENAI_API_KEY`. Nenhuma chave está no repositório.
- Tempos: a recuperação sobre os 999 pares leva de segundos a cerca de 1 minuto por modelo (coluna `time_s` de `embeddings/results/rag_retrieval_summary.csv`). O treino LoRA não foi cronometrado; use GPU ou Apple MPS.
- Resultados de referência: `embeddings/results/` (CSV e figuras geradas pela equipe). Exemplo: MRR de 0,829 para OpenCLIP + LoRA+CL em `rag_retrieval_summary.csv`.

## Alterações em relação à origem

- `dqn_retriever_agent.py`, `rag_generation_reranker.py` e `rag_reranker_experiment.py`: a constante `FIGURES_DIR` apontava para uma pasta local do autor e passou a apontar para `embeddings/results/figures/`.

## O que ficou de fora

- `embeddings/ablation/` (65 MB): pesos dos 13 adaptadores do estudo de ablação (posto, peso interlíngue, número de âncoras). Os resultados estão em `embeddings/results/ablation_results.csv`; `embeddings/ablation_study.py` regenera os pesos. Disponíveis mediante solicitação.
- `__pycache__/`.

# fala/ — Classificador de variantes lusófonas por fala

Configuração, composição dos dados e resultados do classificador de 17 variantes do português (wav2vec2-base ajustado). Tese: cap. 4, seção `sec:q_variantes`; cap. 3, seção de ética (“Voz e inferência de origem”).

| Arquivo | Conteúdo |
|---|---|
| `configuracao_classificador.yaml` | Taxonomia, fontes, balanceamento, partição, arquitetura, hiperparâmetros e sementes |
| `composicao_por_fonte.csv` | Clipes por país e emissora (telejornais) e amostras do Common Voice 25.0 |
| `resultados_agregados.csv` | Acurácia, F1 macro e ponderado, McNemar, intervalos de Wilson, estudo com 5 sementes e análise de erros |
| `resultados_por_classe_wav2vec2_base.csv` | Precisão, revocação, F1 e suporte por classe (737 amostras de teste) |
| `manifesto_clipes_MODELO.csv` | **Modelo** do manifesto dos clipes (só o cabeçalho; ver pendência abaixo) |

A acurácia por grupo geográfico usada na auditoria de equidade está em `experimentos_fronteira/resultados/e5_3_fairness.json`.

## Pendências (registradas, não resolvidas neste depósito)

1. **Código de treino e avaliação.** Fica no repositório `sbrt2026-lusophone-variants`, que não está no repositório de escrita da tese e não foi localizado nesta montagem. Depósito previsto na A12 (novembro de 2026).
2. **Pesos do classificador.** O wav2vec2-base ajustado tem 94,6 M parâmetros (cerca de 380 MB em float32), acima do limite de 50 MB por arquivo deste repositório. Os pesos ficam disponíveis mediante solicitação e serão depositados em *release* do GitHub ou no Zenodo, com o identificador registrado aqui.
3. **Manifesto dos clipes.** A tese promete um manifesto (URL, intervalo e *script* de extração) no lugar do áudio. Os metadados por clipe (identificador do vídeo, início e fim do segmento, rótulo, partição) não existem no repositório de escrita da tese; eles devem ser exportados do repositório do classificador. Até lá, `manifesto_clipes_MODELO.csv` fixa as colunas, e `composicao_por_fonte.csv` dá a composição por emissora. O procedimento de extração está em `configuracao_classificador.yaml` (`dados.telejornais.extracao`).
4. **As 12 regras *regex*** de padronização dos rótulos do Common Voice também ficam no repositório do classificador e entram com ele.

## O que não é depositado, e por quê

- **Nenhum áudio.** Os clipes de telejornais são material de terceiros, usado só para pesquisa não comercial; o Common Voice é distribuído pela Mozilla (CC0) e deve ser obtido na fonte. Nenhuma gravação de participante dos estudos-piloto integrou o conjunto, e as gravações do sistema em operação são descartadas após o processamento.
- **Nenhuma identificação de falante.**

## Divergência a conferir

A versão extensa da tese (cap. 7) registra F1 ponderado de 0,915 para o wav2vec2-base e, para o XLSR-53, F1 macro de 0,664 e ponderado de 0,757; a versão enxuta da qualificação registra 0,914, 0,739 e 0,764. Os arquivos desta pasta seguem a versão enxuta. A conferência depende das predições do repositório do classificador.

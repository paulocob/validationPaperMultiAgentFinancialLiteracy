# revisao/ — Strings de busca da revisão sistemática

Tese: cap. 2, seção `sec:metodologia_revisao` (“Protocolo Metodológico da Revisão”), que remete a este repositório para as 60 *strings*.

| Arquivo | Conteúdo |
|---|---|
| `strings_busca.csv` | As 60 *strings* em forma canônica (colunas `id`, `eixo`, `string`): 53 distribuídas pelos dez eixos temáticos (5 ou 6 por eixo) e 7 interseccionais |
| `strings_busca_integrais.tex` | Fonte LaTeX literal (apêndice “Strings de Busca da Revisão da Literatura” da versão extensa). Os rótulos remetem à versão extensa e não compilam isoladamente |

O CSV foi gerado do `.tex` por conversão automática (aspas e caracteres especiais do LaTeX convertidos para texto simples), sem alterar o conteúdo das consultas.

## Protocolo resumido

- **Período da revisão:** março a setembro de 2026. Corte do *corpus* sistemático em 21/07/2026; varreduras dirigidas até 27/09/2026.
- **Bases:** Google Scholar, IEEE Xplore, ACM Digital Library, Scopus, ScienceDirect e arXiv. As consultas foram adaptadas à sintaxe de cada base; o CSV traz a forma canônica, em notação booleana comum.
- **Filtros:** 2019–2026 (ênfase em 2022–2026); inglês e português; artigos de periódico, de conferência e revisões sistemáticas.
- **Fase 1, coleta:** 60 *strings*, 520 registros.
- **Fase 2, seleção:** título, resumo e texto completo numa única etapa, com critérios explícitos de inclusão e exclusão; 4 duplicatas removidas; 265 registros (51,0 %).
- **Fase 3, *snowballing* bidirecional:** retrospectivo sobre 29 revisões, prospectivo sobre 8 artigos seminais e bidirecional sobre 15 artigos interseccionais; +70 registros.
- **Fase 4, análise:** leitura integral e extração de metadados dos 335 registros do *corpus* final.

Os critérios completos de inclusão e exclusão e o funil por eixo estão no cap. 2 da tese.

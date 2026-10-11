# apendices/ — Apêndices da qualificação

A partir de outubro de 2026, o documento de qualificação mantém apenas dois apêndices: a publicação decorrente do trabalho e os exemplos de interações. Os demais apêndices estão nesta pasta, e a tese remete a eles como “repositório de artefatos (`apendices/`)”.

| Arquivo | Conteúdo |
|---|---|
| `apendices_qualificacao.pdf` | Apêndices A a E da qualificação, compilados a partir do texto da tese |
| `fonte/apendices_repositorio.tex` | Documento LaTeX que gera o PDF |
| `fonte/apendices_repositorio_corpo.tex` | Texto dos cinco apêndices |

## Apêndices do PDF

| Apêndice | Título | Onde a tese usa |
|---|---|---|
| A | Questionário Pós-Sessão (25 itens Likert, quatro dimensões) | Cap. 3, instrumentos; cap. 4, pilotos |
| B | Teste de Conhecimento Financeiro (10 itens, pré e pós-sessão), com a matriz de especificação | Cap. 3, desfecho primário do ensaio; cap. 4, pilotos |
| C | Estrutura das Missões Educacionais | Cap. 3, missões |
| D | Variantes Culturais das Missões Educacionais, com o veredito por missão e a derivação de `f_cult` | Cap. 3, recompensa cultural; cap. 4, componente cultural |
| E | Hiperparâmetros e Configurações dos Experimentos de DRL, com o codificador, as partições, a avaliação *off-policy*, o simulador cognitivo, o planejador LLM e a nota histórica da primeira execução | Cap. 3, agente de DRL; cap. 4, DRL e simulador |

Os apêndices C e D são a versão enxuta da documentação completa que está em [`missoes/`](../missoes/).

## Como ler as remissões

As remissões a capítulos, seções, tabelas e equações que não pertencem a estes apêndices referem-se ao documento de qualificação e usam a numeração dele. Os caminhos citados no Apêndice E são caminhos deste repositório, por exemplo `drl/experimentos_2026-10-10/`.

## Como compilar

O PDF resolve as remissões externas com o pacote `xr`, a partir do arquivo `.aux` do documento de qualificação. Por isso, a fonte compila apenas na pasta da tese, depois do documento principal:

```bash
latexmk -pdf main_qualificacao.tex
latexmk -pdf apendices_repositorio.tex
```

## O que não está aqui

Os exemplos de interações pseudonimizadas continuam como apêndice da tese e não são depositados, pelo mesmo motivo dos demais dados de participantes (ver o README da raiz).

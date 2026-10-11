# Instruções para o planejador

Você é um tutor de educação financeira que planeja a trilha de um aprendiz adulto num programa por mensagens. A cada passo você escolhe UMA tarefa do catálogo. Objetivo: maximizar o aprendizado medido num pós-teste sobre todas as tarefas e evitar que o aprendiz desista. Tarefas cujos pré-requisitos o aprendiz ainda não domina rendem pouco e frustram; tarefas que ele já domina entediam. Uma tarefa pode ser repetida para revisão. Responda apenas com a letra da tarefa.

## Como trabalhar

A cada rodada `k`, leia `rodada_<k>.json` nesta pasta. O arquivo contém uma lista
de aprendizes ativos, cada um com `id` e `estado` (descrição textual do que o
sistema sabe sobre o aprendiz). Para cada aprendiz, escolha UMA tarefa pela letra
(A a K) da lista que aparece no estado.

Escreva `decisoes_<k>.json` (mesmo `k`, com dois dígitos) com um objeto JSON que
mapeia cada `id` a uma letra, por exemplo:

```json
{"a000": "A", "a001": "F", "a002": "C"}
```

Regras:
- Inclua todos os `id` da rodada; cada valor é uma única letra maiúscula de A a K.
- Decida cada aprendiz apenas com o texto do seu estado.
- Não escreva explicações no arquivo de decisões.

Depois de escrever as decisões, o coordenador roda o passo seguinte, que gera
`rodada_<k+1>.json`. Quando aparecer o arquivo `FIM`, não há mais rodadas.

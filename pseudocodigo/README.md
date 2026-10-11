# Pseudocódigo do sistema

Três listagens transcritas literalmente da versão extensa do capítulo de proposta da tese (`material_suplementar/cap_proposta.tex`, setembro de 2026). A versão enxuta da qualificação remete a este repositório para elas.

**Atenção.** As listagens descrevem a especificação do sistema: vetor de estado de 37 dimensões e recompensa composta com pesos 0,35/0,30/0,20/0,15. Os experimentos reexecutados em `drl/experimentos_2026-10-10/` usam a representação de 34 dimensões (37 no experimento cultural), a mistura convexa `0,85·R + 0,15·f_cult` e fator de desconto γ = 0,9 (DQN) ou 0,95 (PPO e CQL). Em caso de divergência, valem o texto da tese e o código.

## 1. Pipeline de orquestração multiagente

A tese remete a esta listagem na seção do agente de análise de sentimento (“O pseudocódigo do fluxo consta do repositório de artefatos”).

```text
Algoritmo: PipelineMultiAgente
Entrada: mensagem m do usuário via WhatsApp
Saída: resposta resp culturalmente adaptada

1. Pré-processamento
   Se m é áudio: texto ← Whisper(m)
   Senão: texto ← m

2. Análise de PLN (paralelo)
   intenção ← AgenteIntenção(texto)
   sentimento ← AgenteSentimento(texto)
   f ← [OneHot(intenção), sentimento, embeddings_contextuais]  // Eq. vetor formatado

3. Atualização do vetor de estado (37 dimensões)
   proficiência ← AtualizarProficiência(resultado_anterior)     // 2 dims
   progresso ← [razão_tarefas, razão_missões, avanço_na_missão] // 3 dims
   contexto  ← [OneHot(segmento, 3), sentimento]                // 4 dims
   atividade ← OneHot(tarefa_atual, 17)                         // 17 dims
   conclusão ← [flag_tarefa, tarefa_terminal, flags_missões(6)] // 8 dims
   origem    ← OneHot(país_do_aprendiz, 3)  // BR, AO, PT       // 3 dims
   s_t ← Concatenar(proficiência, progresso, contexto,
                    atividade, conclusão, origem)               // 37 dims

4. Seleção da próxima tarefa
   a_t ← AgenteDRL(s_t)

5. Geração de resposta
   contexto_rag ← RAG(a_t, BaseConhecimento)
   resp ← LLM(texto, a_t, contexto_rag, sentimento)

6. Atualização da recompensa
   r_t ← CalcularRecompensa(s_t, a_t, C^cult_u, f)  // a intenção entra aqui, via f

Retorne resp
```

## 2. Cálculo da função de recompensa multidimensional

A tese remete a esta listagem na seção da função de recompensa (“O pseudocódigo do cálculo da recompensa consta do repositório de artefatos”).

```text
Algoritmo: CalcularRecompensa
Entrada: estado s_t, ação a_t, perfil cultural C^cult_u,
         vetor de características f (Eq. vetor formatado)
Saída: recompensa total r_tot

1. Recompensa de conclusão (Eq. r_concl)
   concluída  → r_concl = +1.0
   ignorada   → r_concl = -0.1
   abandonada → r_concl = -1.0

2. Recompensa de engajamento (Eq. r_eng)
   r_eng ← min(t_ativo / t_esperado, 1.0)

3. Recompensa de aprendizagem (Eq. r_aprend)
   r_aprend ← (quiz / 100) × 1.5

4. Recompensa cultural (Eq. f_cult)
   missão ← MissãoDe(a_t)
   Se país(C^cult_u) = BR ou missão tem variante profunda para o país:
      r_cult ← 1.0      // estrutura profunda ou contexto de referência
   Senão: r_cult ← 0.5  // adaptação de superfície, ou país não registrado

5. Composição ponderada (Eq. recompensa)
   r_base ← 0.35×r_concl + 0.30×r_eng + 0.20×r_aprend + 0.15×r_cult

6. Refinamento (Eq. recompensa_refinada)
   r_novidade  ← bônus se a_t não visitada
   r_progresso ← pontos_t - pontos_{t-1}
   r_conclusao ← bônus se missão completada
   R ← r_base + r_novidade + r_progresso + r_conclusao

7. Boost de missão, igual nos 4 algoritmos (Eq. recompensa_refinada)
   Se a_t pertence a M2: R ← R + 0.05

8. Alinhamento contextual (Eq. recompensa_contextual)
   r_tot ← R + lambda_ctx × CosSim(Embed(a_t), f)

Retorne r_tot
```

## 3. Progressão e ativação condicional das missões

Complementa a pasta `missoes/`.

```text
Algoritmo: FluxoMissões
Entrada: perfil do aprendiz P
Saída: sequência de missões executadas

1. Missões obrigatórias
   Executar M1: Raio-X das Finanças (275 pts)
   Executar M2: Compra Inteligente (50 pts)
   Executar M3: Plano de Ação (275 pts)

2. Avaliação condicional
   Se P.despesas > P.receitas:
     Executar M4: Plano de Emergência (125 pts)
     Executar M5: Renegociação de Dívidas (275 pts)

3. Missão complementar
   Executar M6: Pesquisa de Preços (275 pts)

4. Conclusão
   pontos_total ← soma dos pontos das missões executadas

Cada missão é composta por fases do tipo:
  - Ação: coleta de dados financeiros reais do aprendiz
  - Reflexão: apresentação de conteúdo educacional
  A cada fase, avaliar resultado e atualizar progresso
```

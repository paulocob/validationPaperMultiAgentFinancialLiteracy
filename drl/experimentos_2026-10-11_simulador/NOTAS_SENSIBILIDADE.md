**Interpretação** (escrita após a execução de 11/10/2026; os números vêm da tabela acima).

- **O limiar de pré-requisito é o fator dominante.** Com limiar 0,5, nenhuma política implementável supera o currículo em h = 1 (Δg de −0,025 a 0,000). O oráculo ainda ganha de +0,038 a +0,087. Com limiar 0,6 ou 0,7, o DQN online ganha de +0,008 a +0,060, e o IC exclui zero nas 18 combinações com limiar ≥ 0,6. O multiplicador afeta sobretudo o teto: com multiplicador 0,5, o Δg do oráculo cai para 0,019–0,052.
- **"Efeito pequeno" se mantém nas 27 combinações.** O maior Δg implementável é +0,064 (guloso) e +0,060 (DQN online), com d ≤ 0,31. Isso fica abaixo do MDE de 0,15 em todas as combinações. O maior Δg do oráculo é +0,135 (d = 0,66), e também fica abaixo de 0,15.
- **"Cerca de 75 % do teto perdido" NÃO é robusto.** O número depende da escala de dispersão:
  - escala 1 ou 1,25 com limiar ≥ 0,6: a melhor política implementável perde 53–80 % do teto;
  - escala 0,75 com limiar ≥ 0,6: perde 40–53 %, e numa combinação (0,5/0,7/0,75) iguala o oráculo, cujo teto ali é só +0,019;
  - limiar 0,5: perde ≥ 100 %, porque nenhuma política implementável ganha algo.
  - Formulação sustentável: a maior parte do teto se perde por observabilidade parcial (≥ 50 % em 23 das 27 combinações). A proporção exata depende de parâmetros não calibrados.
- **Mais dispersão reduz o ganho implementável.** Com limiar ≥ 0,6, passar a escala de 0,75 para 1,25 aumenta o teto do oráculo e reduz o Δg do DQN online. Com mais dispersão, o rastreador, que usa parâmetros médios do segmento, erra mais sobre cada aprendiz.
- **Implicação para A3.** Mesmo no cenário mais favorável (Δg = 0,06, dp do teste de 10 itens ≈ 0,44), seriam necessários cerca de 850 aprendizes por braço para poder 0,80. O diagnóstico de subdimensionamento do plano (n ≈ 90) não depende dos três parâmetros testados.

"""
Gerador do dataset semente (seed) do FinEduBench-Luso.

Cria pares de pergunta-resposta (QA) artesanais e culturalmente
precisos para educacao financeira em tres variantes do portugues:
pt-BR (Brasil), pt-PT (Portugal) e pt-AO (Angola).

Minimo de 30 QA pairs por pais (90 total), cobrindo:
budget, savings, debt, investment, credit, cultural.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Adiciona o diretorio raiz ao path para imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import QAPair, save_jsonl
from dataset.expansions import AO_EXPANSION_ALL, BR_EXPANSION_ALL, PT_EXPANSION_ALL


# ---------------------------------------------------------------------------
# BRASIL (BR) — 30 QA pairs
# ---------------------------------------------------------------------------

BR_PAIRS: list[QAPair] = [
    # --- budget ---
    QAPair(
        id="BR-BUD-001",
        question="O que e a regra 50-30-20 e como eu aplico no meu salario?",
        answer=(
            "A regra 50-30-20 divide sua renda liquida em tres partes: "
            "50% para necessidades essenciais (moradia, alimentacao, transporte), "
            "30% para desejos pessoais (lazer, streaming, restaurantes) e "
            "20% para objetivos financeiros (poupanca, investimentos, quitar dividas). "
            "Exemplo: com salario liquido de R$ 3.000, voce destinaria R$ 1.500 para necessidades, "
            "R$ 900 para desejos e R$ 600 para poupanca/investimento."
        ),
        country="BR",
        topic="budget",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Grafico de pizza dividido em tres fatias coloridas: "
            "50% azul (Necessidades), 30% verde (Desejos), 20% laranja (Poupanca). "
            "Valores em reais (R$) com exemplo para salario de R$ 3.000."
        ),
        cultural_notes="Regra amplamente divulgada por influenciadores financeiros brasileiros como Nath Finan\u00e7as e Me Poupe!.",
        sources=["Banco Central do Brasil - Cidadania Financeira", "CVM Educacional"],
    ),
    QAPair(
        id="BR-BUD-002",
        question="Quais apps gratuitos eu posso usar pra controlar meus gastos?",
        answer=(
            "Existem varios apps gratuitos populares no Brasil: "
            "1) Mobills - permite categorizar gastos e tem integracao com bancos; "
            "2) Organizze - interface simples com graficos de despesas; "
            "3) Guiabolso - conecta direto com sua conta bancaria e categoriza automaticamente; "
            "4) Os proprios apps dos bancos digitais (Nubank, Inter, C6) ja oferecem ferramentas de controle. "
            "O mais importante e registrar TODOS os gastos, mesmo os pequenos como cafezinho e delivery."
        ),
        country="BR",
        topic="budget",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="Apps brasileiros como Guiabolso se conectam ao sistema bancario nacional via Open Banking Brasil.",
        sources=["Banco Central do Brasil - Open Finance"],
    ),
    QAPair(
        id="BR-BUD-003",
        question="Como faco um orcamento mensal se minha renda e variavel? Sou autonomo.",
        answer=(
            "Para renda variavel, use a estrategia do 'piso financeiro': "
            "1) Calcule a media dos ultimos 6 meses de receita; "
            "2) Use o menor valor como base para seu orcamento fixo; "
            "3) Nos meses bons, direcione o excedente para uma reserva de emergencia; "
            "4) Mantenha uma reserva de pelo menos 6 a 12 meses de despesas (maior que para CLT); "
            "5) Como autonomo/MEI, separe no minimo 6% para o DAS mensal e provisione o IRPF. "
            "Uma conta separada so para despesas fixas ajuda a nao misturar dinheiro pessoal e do negocio."
        ),
        country="BR",
        topic="budget",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Fluxograma mostrando: Renda variavel mensal -> Separar despesas fixas (conta 1) "
            "-> Separar impostos MEI/IRPF (conta 2) -> Reserva de emergencia (conta 3) "
            "-> Restante para gastos variaveis. Com setas e icones."
        ),
        cultural_notes="No Brasil, autonomos podem se formalizar como MEI (Microempreendedor Individual) com CNPJ e pagamento simplificado via DAS.",
        sources=["Portal do Empreendedor - gov.br", "Sebrae"],
    ),
    QAPair(
        id="BR-BUD-004",
        question="Vale a pena parcelar no cartao sem juros? Como encaixar no orcamento?",
        answer=(
            "Parcelamento sem juros nao tem custo adicional, mas compromete renda futura. "
            "Regras para usar com seguranca: "
            "1) Some TODAS as parcelas que voce ja esta pagando antes de assumir nova; "
            "2) O total de parcelas nao deve ultrapassar 30% da sua renda liquida; "
            "3) Anote no orcamento o valor total, nao so a parcela do mes; "
            "4) Evite parcelar itens de consumo rapido (comida, combustivel); "
            "5) Prefira parcelas curtas (ate 3x) para nao perder controle. "
            "Lembre: parcela 'sem juros' geralmente ja tem o preco ajustado - tente pedir desconto a vista via PIX."
        ),
        country="BR",
        topic="budget",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="O parcelamento sem juros no cartao e uma pratica tipicamente brasileira, rara em outros paises. O comercio precifica considerando isso.",
        sources=["Procon-SP", "Banco Central do Brasil"],
    ),
    QAPair(
        id="BR-BUD-005",
        question="Como funciona o desconto em folha e o que sobra do meu salario bruto?",
        answer=(
            "Do salario bruto sao descontados obrigatoriamente: "
            "1) INSS - de 7,5% a 14% conforme faixa salarial (tabela progressiva); "
            "2) IRRF - de 0% a 27,5% conforme faixa, ja descontando o INSS; "
            "3) Outros possiveis: vale-transporte (ate 6% do salario), plano de saude, previdencia privada. "
            "Exemplo: salario bruto de R$ 5.000 -> INSS ~R$ 477 -> IRRF ~R$ 142 -> liquido ~R$ 4.381 "
            "(sem outros descontos). Sempre consulte seu holerite detalhado."
        ),
        country="BR",
        topic="budget",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Infografico tipo cascata (waterfall chart) mostrando: Salario Bruto R$ 5.000 "
            "-> desconto INSS -> desconto IRRF -> desconto VT -> Salario Liquido. "
            "Cada barra mostra o valor do desconto e a cor indica se e obrigatorio ou opcional."
        ),
        cultural_notes="A CLT brasileira tem estrutura de descontos propria. O 13o salario e as ferias remuneradas sao direitos trabalhistas brasileiros unicos.",
        sources=["Receita Federal do Brasil - Tabela IRRF", "INSS - Tabela de contribuicao"],
    ),
    # --- savings ---
    QAPair(
        id="BR-SAV-001",
        question="Poupanca ainda vale a pena em 2025? Quais alternativas rendem mais?",
        answer=(
            "A poupanca rende 0,5% ao mes + TR quando a Selic esta acima de 8,5% ao ano. "
            "Alternativas que geralmente rendem mais com seguranca similar: "
            "1) Tesouro Selic - rende proximo a taxa Selic, liquidez D+1, garantido pelo governo; "
            "2) CDB de liquidez diaria - rende 100% a 110% do CDI, protegido pelo FGC ate R$ 250 mil; "
            "3) LCI/LCA - isenta de IR para pessoa fisica, rende entre 90-100% do CDI. "
            "A poupanca so faz sentido para valores muito pequenos (pela simplicidade) "
            "ou se voce precisa de isencao total de IR com liquidez imediata."
        ),
        country="BR",
        topic="savings",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Tabela comparativa com colunas: Produto | Rentabilidade | Liquidez | Risco | IR. "
            "Linhas: Poupanca, Tesouro Selic, CDB 100% CDI, LCI/LCA. "
            "Destaques visuais indicando o melhor em cada categoria."
        ),
        cultural_notes="A poupanca tem enorme penetracao cultural no Brasil - muitos brasileiros so conhecem ela como opcao de investimento.",
        sources=["Banco Central do Brasil", "Tesouro Nacional"],
    ),
    QAPair(
        id="BR-SAV-002",
        question="Como funciona o Tesouro Direto? E seguro? Quanto preciso pra comecar?",
        answer=(
            "O Tesouro Direto e o programa do governo federal que permite investir em titulos publicos. "
            "E o investimento mais seguro do Brasil pois e garantido pelo Tesouro Nacional. "
            "Voce pode comecar com cerca de R$ 30 (valor minimo de compra de uma fracao de titulo). "
            "Principais titulos: "
            "1) Tesouro Selic - rende a taxa Selic, ideal para reserva de emergencia; "
            "2) Tesouro Prefixado - taxa fixa definida na compra; "
            "3) Tesouro IPCA+ - rende inflacao (IPCA) + juros fixos, protege o poder de compra. "
            "Para investir: abra conta em uma corretora (muitas sao gratuitas), transfira o dinheiro e compre pelo site ou app."
        ),
        country="BR",
        topic="savings",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Diagrama explicativo do fluxo do Tesouro Direto: Investidor -> Corretora -> Tesouro Nacional. "
            "Abaixo, tres icones representando os tres tipos de titulo com suas caracteristicas resumidas."
        ),
        cultural_notes="O Tesouro Direto democratizou o acesso a titulos publicos no Brasil a partir de 2002. Antes, so grandes investidores tinham acesso.",
        sources=["Tesouro Nacional - tesourodireto.com.br"],
    ),
    QAPair(
        id="BR-SAV-003",
        question="Qual a diferenca entre CDB, LCI e LCA? Qual escolher?",
        answer=(
            "Todos sao titulos de renda fixa emitidos por bancos, mas diferem em tributacao e lastro: "
            "CDB (Certificado de Deposito Bancario): paga IR regressivo (de 22,5% a 15% conforme prazo). "
            "LCI (Letra de Credito Imobiliario): isenta de IR, lastro em credito imobiliario. "
            "LCA (Letra de Credito do Agronegocio): isenta de IR, lastro no agronegocio. "
            "Todos sao protegidos pelo FGC ate R$ 250 mil por CPF por instituicao. "
            "Regra pratica: compare a rentabilidade liquida (apos IR). "
            "Um CDB a 110% do CDI pode render menos que uma LCI a 95% do CDI, porque a LCI e isenta de IR."
        ),
        country="BR",
        topic="savings",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Tabela comparativa lado a lado: CDB vs LCI vs LCA. "
            "Colunas: Imposto de Renda, Garantia FGC, Liquidez tipica, Rentabilidade tipica. "
            "Com calculo exemplo mostrando rendimento liquido de R$ 10.000 em cada um apos 1 ano."
        ),
        cultural_notes="O FGC (Fundo Garantidor de Creditos) e uma instituicao privada brasileira que protege depositantes, similar ao FDIC americano.",
        sources=["FGC - fgc.org.br", "CVM Educacional"],
    ),
    QAPair(
        id="BR-SAV-004",
        question="Como montar uma reserva de emergencia? Quanto preciso guardar?",
        answer=(
            "A reserva de emergencia deve cobrir de 3 a 12 meses de despesas fixas mensais. "
            "Para CLT: 3 a 6 meses; para autonomos/MEI: 6 a 12 meses. "
            "Passos: 1) Calcule suas despesas fixas mensais (aluguel, contas, alimentacao); "
            "2) Multiplique pelo numero de meses desejado; "
            "3) Aplique em investimento com liquidez diaria e baixo risco: Tesouro Selic ou CDB de liquidez diaria; "
            "4) Contribua todo mes ate atingir a meta; "
            "5) NUNCA use a reserva para oportunidades - ela e so para emergencias reais "
            "(perda de emprego, problema de saude, conserto urgente)."
        ),
        country="BR",
        topic="savings",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Termometro de meta financeira mostrando niveis: "
            "1 mes (vermelho/critico), 3 meses (amarelo/minimo CLT), "
            "6 meses (verde/ideal CLT e minimo autonomo), 12 meses (azul/ideal autonomo). "
            "Com valor exemplo em reais para despesas de R$ 2.500/mes."
        ),
        cultural_notes="Pesquisa do Banco Central (2023) mostra que 58% dos brasileiros nao possuem reserva de emergencia.",
        sources=["Banco Central do Brasil - Relatorio de Cidadania Financeira"],
    ),
    QAPair(
        id="BR-SAV-005",
        question="Previdencia privada (PGBL e VGBL): quando vale a pena?",
        answer=(
            "PGBL: indicado para quem faz declaracao completa do IR - voce pode deduzir ate 12% da renda bruta. "
            "O IR incide sobre o valor total no resgate. "
            "VGBL: indicado para quem faz declaracao simplificada - nao tem deducao, "
            "mas o IR incide so sobre os rendimentos. "
            "Ambos podem usar tabela regressiva (de 35% a 10% conforme prazo) ou progressiva. "
            "Vale a pena quando: voce ja tem reserva de emergencia, ja investe em Tesouro/renda fixa, "
            "e quer o beneficio fiscal (PGBL) ou sucessorio (ambos - nao entra em inventario). "
            "Atencao: taxas de administracao acima de 1% ao ano corroem o rendimento."
        ),
        country="BR",
        topic="savings",
        difficulty="advanced",
        modalities=["text", "image"],
        image_description=(
            "Arvore de decisao: 'Faz declaracao completa do IR?' -> Sim: PGBL / Nao: VGBL. "
            "Abaixo, comparacao visual das tabelas regressiva e progressiva de tributacao."
        ),
        cultural_notes="No Brasil, previdencia privada tem papel importante no planejamento sucessorio pois nao entra em inventario.",
        sources=["Susep", "Receita Federal do Brasil"],
    ),
    # --- debt ---
    QAPair(
        id="BR-DEB-001",
        question="To endividado no cartao de credito, o que faco primeiro?",
        answer=(
            "Passo a passo para sair da divida de cartao: "
            "1) PARE de usar o cartao imediatamente - pague tudo no debito ou dinheiro; "
            "2) Levante o valor total da divida (fatura + rotativo + parcelamentos); "
            "3) Negocie: ligue para o banco e peca reducao de juros ou parcelamento; "
            "4) Use o programa Desenrola Brasil (gov.br) se elegivel - oferece descontos de ate 96%; "
            "5) Considere um emprestimo pessoal com juros menores para quitar o cartao (portabilidade de divida); "
            "6) Os juros do rotativo do cartao podem passar de 400% ao ano - priorize quitar isso. "
            "Lembre: o minimo do cartao e uma armadilha - voce paga juros sobre juros."
        ),
        country="BR",
        topic="debt",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="O rotativo do cartao de credito no Brasil tem uma das maiores taxas de juros do mundo, frequentemente acima de 400% a.a.",
        sources=["Banco Central do Brasil - Taxas de juros", "Desenrola Brasil - gov.br"],
    ),
    QAPair(
        id="BR-DEB-002",
        question="O que e o cheque especial e por que devo evitar?",
        answer=(
            "O cheque especial e um credito pre-aprovado que o banco disponibiliza na sua conta corrente. "
            "Quando seu saldo fica negativo, voce automaticamente usa o cheque especial. "
            "O problema: os juros sao altissimos - em media 8% ao mes (mais de 150% ao ano). "
            "Desde 2020, os bancos cobram ate 8% ao mes no cheque especial. "
            "Alternativas melhores: emprestimo pessoal (juros menores), antecipacao do FGTS, "
            "ou ate emprestimo consignado (se disponivel). "
            "Dica: peca ao banco para REDUZIR ou CANCELAR o limite do cheque especial "
            "para evitar usa-lo sem querer."
        ),
        country="BR",
        topic="debt",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="O cheque especial e uma fonte cronica de endividamento no Brasil. Bancos o oferecem automaticamente, induzindo uso involuntario.",
        sources=["Banco Central do Brasil - Resolucao 4.765/2019"],
    ),
    QAPair(
        id="BR-DEB-003",
        question="Como funciona o Desenrola Brasil? Quem pode participar?",
        answer=(
            "O Desenrola Brasil e um programa do governo federal para renegociacao de dividas. "
            "Faixa 1: para quem ganha ate 2 salarios minimos e tem dividas de ate R$ 5.000 - "
            "oferece descontos de ate 96% e parcelamento em ate 60x. "
            "Faixa 2: para quem ganha ate R$ 20.000 e tem dividas com bancos - "
            "os bancos oferecem condicoes especiais de renegociacao. "
            "Como acessar: pelo app ou site dos bancos credores, pelo portal Desenrola (gov.br), "
            "ou pelo Serasa Limpa Nome (parceiro oficial). "
            "Documentos: CPF e dados da conta bancaria. A adesao e gratuita."
        ),
        country="BR",
        topic="debt",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="O Desenrola Brasil (2023-2024) foi o maior programa de renegociacao de dividas da historia do pais, atingindo mais de 14 milhoes de pessoas.",
        sources=["gov.br/desenrola", "Ministerio da Fazenda"],
    ),
    QAPair(
        id="BR-DEB-004",
        question="O que e a portabilidade de divida e como posso usar pra pagar menos juros?",
        answer=(
            "Portabilidade de credito e o direito de transferir sua divida de um banco para outro "
            "que ofereca condicoes melhores (juros menores, prazo maior). "
            "E regulamentada pelo Banco Central e os bancos sao obrigados a aceitar. "
            "Passos: 1) Consulte o CET (Custo Efetivo Total) da sua divida atual; "
            "2) Pesquise ofertas em outros bancos; "
            "3) Solicite a portabilidade no novo banco - ele cuida da burocracia; "
            "4) O banco antigo tem 5 dias uteis para fazer contraproposta. "
            "Funciona para: credito consignado, financiamento imobiliario, financiamento de veiculos e emprestimo pessoal."
        ),
        country="BR",
        topic="debt",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="A portabilidade de credito e um direito do consumidor brasileiro regulado pelo Banco Central desde 2013.",
        sources=["Banco Central do Brasil - Portabilidade de Credito"],
    ),
    QAPair(
        id="BR-DEB-005",
        question="Devo usar meu FGTS pra quitar dividas?",
        answer=(
            "Depende da situacao. O FGTS rende apenas TR + 3% ao ano, bem abaixo da inflacao. "
            "Vale a pena usar o FGTS se: a divida tem juros altos (cartao, cheque especial), "
            "voce tem acesso via saque-aniversario ou demissao. "
            "NAO vale se: voce pode precisar do FGTS para dar entrada em imovel (uso mais vantajoso), "
            "ou se a divida tem juros baixos (menores que o rendimento do FGTS). "
            "Opcoes de acesso: saque-aniversario (parcela anual do saldo), "
            "antecipacao do saque-aniversario (bancos emprestam com juros menores usando FGTS como garantia), "
            "ou saque-rescisao (ao ser demitido). "
            "Atencao: ao optar pelo saque-aniversario, voce perde o direito ao saque total em caso de demissao."
        ),
        country="BR",
        topic="debt",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="O FGTS e uma peculiaridade trabalhista brasileira - um fundo compulsorio que funciona como poupanca forcada do trabalhador CLT.",
        sources=["Caixa Economica Federal - FGTS"],
    ),
    # --- investment ---
    QAPair(
        id="BR-INV-001",
        question="O que sao fundos imobiliarios (FIIs) e como recebo dividendos?",
        answer=(
            "Fundos imobiliarios sao fundos que investem em imoveis (shopping centers, lajes corporativas, "
            "galpoes logisticos) ou em titulos ligados ao setor imobiliario (CRI). "
            "Voce compra cotas na bolsa (B3) como se fossem acoes. "
            "Vantagem principal: os rendimentos (dividendos) sao isentos de IR para pessoa fisica, "
            "desde que o fundo tenha mais de 50 cotistas e as cotas sejam negociadas em bolsa. "
            "Os dividendos geralmente sao pagos mensalmente na sua conta da corretora. "
            "Riscos: vacancia (imoveis desocupados), desvalorizacao das cotas, risco de credito (FIIs de papel). "
            "Comece estudando FIIs de tijolo diversificados e com bom historico de dividendos."
        ),
        country="BR",
        topic="investment",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Diagrama mostrando o fluxo: Investidores compram cotas -> Fundo adquire imoveis -> "
            "Imoveis geram aluguel -> Aluguel distribuido como dividendos (isento de IR). "
            "Com icones de predios, cifras e setas."
        ),
        cultural_notes="FIIs se tornaram muito populares no Brasil pos-2019, especialmente pela isencao de IR nos dividendos.",
        sources=["B3 - bolsa de valores", "CVM"],
    ),
    QAPair(
        id="BR-INV-002",
        question="Qual a diferenca entre renda fixa e renda variavel? O que e melhor pra iniciante?",
        answer=(
            "Renda fixa: voce empresta dinheiro (ao governo ou banco) e recebe juros. "
            "Previsivel, menor risco. Ex: Tesouro Direto, CDB, LCI/LCA. "
            "Renda variavel: voce compra participacao em ativos cujo preco varia. "
            "Potencial de ganho maior, mas com risco de perda. Ex: acoes, FIIs, ETFs. "
            "Para iniciante, a recomendacao e: "
            "1) Primeiro: monte reserva de emergencia em renda fixa (Tesouro Selic); "
            "2) Segundo: aprenda sobre renda fixa e diversifique (CDB, LCI/LCA, Tesouro IPCA+); "
            "3) Terceiro: so entao comece a explorar renda variavel, com no maximo 10-20% do patrimonio. "
            "Nunca invista em algo que voce nao entende."
        ),
        country="BR",
        topic="investment",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Piramide de investimentos com base larga (renda fixa/baixo risco) e topo estreito "
            "(renda variavel/alto risco). Cada nivel mostra exemplos de produtos. "
            "Seta indicando: 'Comece por aqui' na base."
        ),
        cultural_notes="Com a Selic historicamente alta no Brasil, a renda fixa frequentemente oferece retornos atraentes mesmo comparada a renda variavel.",
        sources=["CVM Educacional", "Anbima"],
    ),
    QAPair(
        id="BR-INV-003",
        question="Como funciona o Tesouro IPCA+ e por que protege da inflacao?",
        answer=(
            "O Tesouro IPCA+ paga a variacao do IPCA (inflacao oficial) mais uma taxa de juros prefixada. "
            "Exemplo: IPCA + 6% ao ano significa que voce recebe a inflacao do periodo mais 6% reais. "
            "Isso garante que seu dinheiro SEMPRE renda acima da inflacao, protegendo o poder de compra. "
            "Prazos disponiveis: 2029, 2035, 2045 (variam). Tem opcao com juros semestrais. "
            "Atencao: se vender antes do vencimento, o preco varia conforme o mercado (marcacao a mercado) - "
            "voce pode ter ganho ou perda. Se levar ate o vencimento, recebe exatamente o contratado. "
            "Ideal para objetivos de longo prazo: aposentadoria, faculdade dos filhos, compra de imovel."
        ),
        country="BR",
        topic="investment",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Grafico de linhas ao longo de 10 anos comparando: "
            "rendimento do Tesouro IPCA+ (linha verde acima), inflacao IPCA acumulada (linha vermelha abaixo), "
            "e poupanca (linha cinza no meio). Mostrando como o IPCA+ sempre supera a inflacao."
        ),
        cultural_notes="O Tesouro IPCA+ e particularmente relevante no Brasil devido ao historico de inflacao alta do pais.",
        sources=["Tesouro Nacional - tesourodireto.com.br"],
    ),
    QAPair(
        id="BR-INV-004",
        question="ETFs valem a pena no Brasil? Como investir?",
        answer=(
            "ETFs (Exchange Traded Funds) sao fundos de indice negociados na bolsa. "
            "No Brasil, os principais sao: BOVA11 (replica o Ibovespa), IVVB11 (replica o S&P 500 em reais), "
            "SMAL11 (small caps brasileiras), HASH11 (criptomoedas). "
            "Vantagens: diversificacao instantanea, taxas de administracao baixas (0,10% a 0,50% a.a.), "
            "facilidade de compra (como uma acao). "
            "Desvantagens no Brasil: nao distribuem dividendos (reinvestem automaticamente), "
            "IR de 15% sobre ganho de capital sem isencao para vendas ate R$ 20 mil/mes "
            "(diferente de acoes que tem essa isencao). "
            "Para investir: abra conta em corretora e compre pelo home broker usando o codigo do ETF."
        ),
        country="BR",
        topic="investment",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="ETFs ainda sao menos populares no Brasil que nos EUA, mas vem crescendo. A tributacao brasileira difere da americana.",
        sources=["B3", "CVM"],
    ),
    # --- credit ---
    QAPair(
        id="BR-CRE-001",
        question="Como consultar e melhorar meu score no Serasa?",
        answer=(
            "Voce consulta gratuitamente pelo app Serasa ou pelo site serasa.com.br. "
            "O score vai de 0 a 1000 - quanto maior, melhor. "
            "Para melhorar: "
            "1) Pague suas contas em dia (fator mais importante); "
            "2) Negocie e quite dividas em aberto pelo Serasa Limpa Nome; "
            "3) Mantenha seu cadastro atualizado (endereco, renda); "
            "4) Ative o Cadastro Positivo - ele registra seu historico de bom pagador; "
            "5) Evite pedir muito credito em pouco tempo (cada consulta ao seu CPF pode afetar); "
            "6) Use credito de forma consciente e pague a fatura total do cartao. "
            "O score nao melhora da noite pro dia - e uma construcao de meses."
        ),
        country="BR",
        topic="credit",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Escala visual tipo velocimetro do score Serasa: "
            "0-300 (vermelho/muito baixo), 301-500 (laranja/baixo), "
            "501-700 (amarelo/bom), 701-1000 (verde/muito bom). "
            "Com dicas rapidas ao lado de cada faixa."
        ),
        cultural_notes="O Serasa Score e a principal referencia de credito no Brasil. O Cadastro Positivo se tornou obrigatorio em 2019.",
        sources=["Serasa Experian", "Banco Central do Brasil"],
    ),
    QAPair(
        id="BR-CRE-002",
        question="Meu CPF ta irregular, o que acontece e como regularizar?",
        answer=(
            "CPF irregular pode impedir: abrir conta bancaria, fazer compras parceladas, "
            "acessar programas do governo, tirar passaporte, e ate receber salario. "
            "Motivos comuns: pendencia com a Receita Federal (declaracao de IR nao entregue), "
            "inscricao de oficio sem regularizacao, ou informacao cadastral incorreta. "
            "Para regularizar: "
            "1) Consulte a situacao no site da Receita Federal (cpf.receita.fazenda.gov.br); "
            "2) Se for pendencia de IR: entregue as declaracoes pendentes; "
            "3) Se for cadastral: va a uma agencia dos Correios, Banco do Brasil ou Caixa com documentos; "
            "4) O custo para regularizar e de R$ 7,00 (taxa dos Correios) se for via atendimento presencial. "
            "Pode levar ate 48 horas para atualizar no sistema apos a regularizacao."
        ),
        country="BR",
        topic="credit",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="O CPF e o documento fundamental para a vida financeira no Brasil. Qualquer irregularidade bloqueia praticamente todas as operacoes.",
        sources=["Receita Federal do Brasil"],
    ),
    # --- cultural ---
    QAPair(
        id="BR-CUL-001",
        question="PIX e seguro? Quais os golpes mais comuns e como me proteger?",
        answer=(
            "O PIX em si e seguro - usa criptografia e e regulado pelo Banco Central. "
            "Golpes mais comuns: "
            "1) Falso funcionario de banco pedindo transferencia 'de teste'; "
            "2) QR Code adulterado em boletos ou maquininhas; "
            "3) Clonagem de WhatsApp pedindo PIX para contatos; "
            "4) Sequestro relampago (coacao para fazer PIX); "
            "5) Golpe do PIX errado (pedem devolucao de valor que nunca foi enviado). "
            "Protecao: configure limite de PIX noturno (reduzido), ative verificacao em 2 etapas no WhatsApp, "
            "nunca faca PIX sob pressao, confirme dados do destinatario antes de enviar. "
            "Em caso de golpe: registre BO e acione o MED (Mecanismo Especial de Devolucao) pelo seu banco."
        ),
        country="BR",
        topic="cultural",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Infografico com 5 icones de golpes comuns do PIX, cada um com titulo curto e dica de prevencao. "
            "Visual tipo alerta de seguranca com cores vermelho/amarelo/verde."
        ),
        cultural_notes="O PIX revolucionou pagamentos no Brasil desde 2020, com mais de 150 milhoes de usuarios. Golpes evoluiram junto.",
        sources=["Banco Central do Brasil - PIX", "Febraban"],
    ),
    QAPair(
        id="BR-CUL-002",
        question="Como funciona o consorcio? E melhor que financiamento?",
        answer=(
            "Consorcio e um grupo de pessoas que contribui mensalmente para um fundo comum. "
            "Todo mes, um ou mais participantes sao contemplados (por sorteio ou lance) e recebem "
            "a carta de credito para adquirir o bem (carro, imovel, etc.). "
            "Vantagens: nao tem juros (apenas taxa de administracao, geralmente de 10-20% do total), "
            "disciplina de poupanca, poder de compra a vista ao ser contemplado. "
            "Desvantagens: voce pode demorar meses ou anos para ser contemplado, "
            "nao tem liquidez (dificil sair do grupo), taxa de administracao pode ser alta. "
            "Comparacao: financiamento = paga juros mas tem o bem imediato; "
            "consorcio = sem juros mas sem garantia de quando recebe o bem. "
            "Nao e investimento - e uma forma de compra planejada."
        ),
        country="BR",
        topic="cultural",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Timeline visual comparando consorcio vs financiamento ao longo de 5 anos: "
            "mostrando quando voce recebe o bem e quanto paga no total em cada modalidade. "
            "Destaque para o custo total de cada opcao."
        ),
        cultural_notes="O consorcio e uma invencao brasileira dos anos 1960, muito popular ate hoje. Nao existe equivalente direto em muitos paises.",
        sources=["ABAC - Associacao Brasileira de Administradoras de Consorcios", "Banco Central do Brasil"],
    ),
    QAPair(
        id="BR-CUL-003",
        question="O que e o Bolsa Familia e como posso me inscrever?",
        answer=(
            "O Bolsa Familia e o principal programa de transferencia de renda do governo federal. "
            "Beneficio minimo de R$ 600 por familia, com adicionais para criancas, gestantes e jovens. "
            "Para se inscrever: "
            "1) A familia deve estar no CadUnico (Cadastro Unico) - procure o CRAS da sua cidade; "
            "2) Renda per capita familiar de ate R$ 218/mes; "
            "3) Leve documentos de todos os membros da familia (RG, CPF, comprovante de residencia); "
            "4) O beneficio e pago pela Caixa Economica Federal, geralmente pelo app Caixa Tem. "
            "Condicionalidades: criancas devem ter frequencia escolar minima e vacinacao em dia. "
            "O programa e um direito - nao exige intermediarios nem pagamento para inscricao."
        ),
        country="BR",
        topic="cultural",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="O Bolsa Familia e o maior programa de transferencia de renda da America Latina, atingindo mais de 21 milhoes de familias.",
        sources=["Ministerio do Desenvolvimento Social - gov.br", "Caixa Economica Federal"],
    ),
    QAPair(
        id="BR-CUL-004",
        question="Sou MEI, como funciona a parte financeira?",
        answer=(
            "MEI (Microempreendedor Individual) e a forma mais simples de formalizar um negocio no Brasil. "
            "Financeiramente: "
            "1) Faturamento maximo: R$ 81.000/ano (R$ 6.750/mes em media); "
            "2) Imposto unico (DAS): R$ 71,60 a R$ 76,60/mes (valores de 2024), inclui INSS; "
            "3) Voce tem CNPJ e pode emitir nota fiscal; "
            "4) Direito a aposentadoria por idade e auxilio-doenca pelo INSS; "
            "5) Pode abrir conta PJ e ter acesso a credito especifico para MEI. "
            "Dicas financeiras: separe conta PJ e PF, faca um pro-labore fixo (seu 'salario'), "
            "guarde para impostos trimestrais se necessario, e faca a declaracao anual (DASN-SIMEI) ate 31 de maio."
        ),
        country="BR",
        topic="cultural",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Infografico resumindo os numeros do MEI: faturamento maximo, valor do DAS, "
            "beneficios inclusos (INSS, nota fiscal, CNPJ). "
            "Com calendario indicando datas importantes: pagamento DAS (dia 20) e declaracao anual (maio)."
        ),
        cultural_notes="O MEI formalizou mais de 15 milhoes de brasileiros desde 2008, sendo essencial para a economia informal do pais.",
        sources=["Portal do Empreendedor - gov.br", "Sebrae"],
    ),
    QAPair(
        id="BR-CUL-005",
        question="Como declarar PIX e transferencias no Imposto de Renda?",
        answer=(
            "O PIX em si nao e declarado - voce declara os BENS e RENDIMENTOS. "
            "O que muda: desde 2025, instituicoes financeiras e de pagamento (Nubank, Mercado Pago, PicPay) "
            "devem reportar a Receita Federal movimentacoes acima de R$ 5.000/mes (PF) via e-Financeira. "
            "Isso nao e imposto novo - e apenas mais informacao para a Receita cruzar dados. "
            "O que declarar: seus rendimentos (salario, freelance, aluguel), saldos em contas, "
            "investimentos e bens. Recebimentos via PIX que sao rendimentos devem ser declarados normalmente. "
            "PIX entre contas proprias ou presentes de pequeno valor nao geram imposto."
        ),
        country="BR",
        topic="cultural",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="A ampliacao do monitoramento de PIX pela Receita Federal gerou grande polemica no Brasil em 2025, com muita desinformacao.",
        sources=["Receita Federal do Brasil", "Instrucao Normativa RFB 2.219/2024"],
    ),
    # --- investment (extra) ---
    QAPair(
        id="BR-INV-005",
        question="O que sao acoes e como comeco a investir na B3?",
        answer=(
            "Acoes sao pequenas fracoes do capital de uma empresa. Ao comprar uma acao, voce se torna socio daquela empresa. "
            "Para investir na B3 (bolsa de valores brasileira): "
            "1) Abra conta em uma corretora de valores (Nubank, XP, Clear, Rico - muitas com taxa zero); "
            "2) Transfira dinheiro para a conta da corretora; "
            "3) Acesse o home broker e pesquise o codigo da acao (ex: PETR4 = Petrobras, VALE3 = Vale); "
            "4) Comece com pouco - o lote padrao e de 100 acoes, mas no mercado fracionario voce compra a partir de 1 acao; "
            "5) Diversifique entre setores diferentes. "
            "Custos: IR de 15% sobre lucro na venda (isento ate R$ 20 mil em vendas/mes). "
            "Riscos: preco das acoes oscila diariamente - invista apenas dinheiro que nao vai precisar no curto prazo."
        ),
        country="BR",
        topic="investment",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Tela simplificada de home broker mostrando: lista de acoes com codigo, preco, variacao. "
            "Destaque para os botoes de compra e venda e o campo de quantidade."
        ),
        cultural_notes="A B3 e a unica bolsa de valores do Brasil. O numero de investidores pessoa fisica cresceu de 800 mil (2018) para mais de 5 milhoes (2024).",
        sources=["B3", "CVM Educacional"],
    ),
    QAPair(
        id="BR-CRE-003",
        question="O que e o Cadastro Positivo e como ele me ajuda?",
        answer=(
            "O Cadastro Positivo e um banco de dados que registra seu historico de BOM pagador - "
            "contas pagas em dia, creditos quitados corretamente. "
            "Desde 2019, a inscricao e automatica para todos os brasileiros com CPF. "
            "Beneficios: melhora seu score de credito, pode resultar em juros menores em emprestimos, "
            "e facilita aprovacao de credito. "
            "Diferenca: antes, so existia o cadastro negativo (quem devia). Agora, quem paga em dia tambem e registrado. "
            "Voce pode consultar gratuitamente nos birôs de credito: Serasa, Boa Vista, SPC Brasil. "
            "Pode tambem solicitar a exclusao do Cadastro Positivo, mas isso pode prejudicar seu acesso a credito."
        ),
        country="BR",
        topic="credit",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="O Cadastro Positivo obrigatorio foi uma mudanca significativa no mercado de credito brasileiro em 2019.",
        sources=["Lei Complementar 166/2019", "Banco Central do Brasil"],
    ),
    QAPair(
        id="BR-DEB-006",
        question="Como funciona o emprestimo consignado? Quem pode pegar?",
        answer=(
            "O emprestimo consignado e um credito em que as parcelas sao descontadas diretamente do salario ou beneficio. "
            "Quem pode: aposentados e pensionistas do INSS, servidores publicos, trabalhadores CLT (se a empresa tiver convenio). "
            "Vantagens: juros bem mais baixos que outras modalidades (a partir de 1,5% ao mes para INSS). "
            "Limite: as parcelas nao podem ultrapassar 35% da renda (30% para emprestimo + 5% para cartao consignado). "
            "Cuidados: nao aceite ofertas por telefone sem verificar se sao do seu banco; "
            "golpistas se passam por funcionarios de banco para aplicar fraudes; "
            "avalie se realmente precisa antes de contratar, pois compromete renda futura."
        ),
        country="BR",
        topic="debt",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="O consignado e amplamente usado por aposentados no Brasil, mas fraudes e assedio comercial sao problemas recorrentes.",
        sources=["Banco Central do Brasil", "INSS"],
    ),
    QAPair(
        id="BR-CUL-006",
        question="O que e o Open Finance no Brasil e como me afeta?",
        answer=(
            "Open Finance (antes chamado Open Banking) e um sistema regulado pelo Banco Central que permite "
            "compartilhar seus dados financeiros entre instituicoes, com sua autorizacao. "
            "Na pratica: voce pode autorizar um banco a ver seu historico em outro banco, "
            "facilitando ofertas personalizadas de credito, investimento e seguros. "
            "Beneficios: mais concorrencia entre bancos, propostas melhores de credito e investimento, "
            "visao consolidada das suas financas em um unico app. "
            "Seguranca: o compartilhamento so acontece com seu consentimento explicito, "
            "os dados sao protegidos pela LGPD, e voce pode revogar a qualquer momento. "
            "O Brasil e referencia mundial nessa iniciativa, a frente de muitos paises europeus."
        ),
        country="BR",
        topic="cultural",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Diagrama mostrando o fluxo do Open Finance: Usuario autoriza -> Banco A envia dados para Banco B "
            "-> Banco B oferece produto personalizado. Com icone de cadeado indicando seguranca e consentimento."
        ),
        cultural_notes="O Open Finance brasileiro e considerado um dos mais ambiciosos do mundo, incluindo dados bancarios, de investimentos, seguros e cambio.",
        sources=["Banco Central do Brasil - Open Finance"],
    ),
]


# ---------------------------------------------------------------------------
# PORTUGAL (PT) — 30 QA pairs
# ---------------------------------------------------------------------------

PT_PAIRS: list[QAPair] = [
    # --- budget ---
    QAPair(
        id="PT-BUD-001",
        question="Como organizo o meu orcamento com o subsidio de ferias e de Natal?",
        answer=(
            "Em Portugal, os trabalhadores por conta de outrem recebem 14 meses de salario: "
            "12 meses normais + subsidio de ferias + subsidio de Natal. "
            "Existem duas formas de receber: em duodecimos (diluido nos 12 meses) ou em dois montantes extra. "
            "Se recebe em duodecimos: o orcamento mensal ja inclui tudo, mais facil de gerir. "
            "Se recebe de uma vez: nao trate como 'dinheiro extra' - reserve pelo menos 50% para poupanca "
            "ou amortizacao de creditos. "
            "Dica: faca o orcamento anual dividido por 12, nao por 14, para evitar meses de aperto."
        ),
        country="PT",
        topic="budget",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Grafico de barras comparando dois cenarios: recebimento em duodecimos (barras iguais) "
            "vs recebimento com subsidios extra (barras altas em junho e novembro). "
            "Linha horizontal mostrando a media mensal ideal."
        ),
        cultural_notes="Os subsidios de ferias e Natal sao direitos consagrados no Codigo do Trabalho portugues, equivalendo a 14 salarios anuais.",
        sources=["Codigo do Trabalho - Artigo 263 e 264", "DECO Proteste"],
    ),
    QAPair(
        id="PT-BUD-002",
        question="Quais despesas posso deduzir no IRS atraves do e-fatura?",
        answer=(
            "No e-fatura, as suas despesas sao automaticamente categorizadas para deducao no IRS: "
            "1) Despesas gerais familiares: 35% das despesas, ate 250 euros por sujeito passivo; "
            "2) Saude: 15% das despesas, ate 1.000 euros; "
            "3) Educacao: 30% das despesas, ate 800 euros; "
            "4) Habitacao: 15% de rendas, ate 502 euros (ou juros de credito habitacao); "
            "5) Lares: 25% das despesas, ate 403,75 euros; "
            "6) Reparacao automovel e alojamento/restauracao: 15%, ate 250 euros. "
            "Para maximizar: peca sempre fatura com NIF, verifique mensalmente o e-fatura, "
            "e classifique despesas pendentes antes de 25 de fevereiro."
        ),
        country="PT",
        topic="budget",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Tabela resumo com categorias do e-fatura: icone da categoria, percentagem dedutivel, "
            "teto maximo em euros. Com destaque visual para as categorias mais vantajosas."
        ),
        cultural_notes="O sistema e-fatura portugues e inovador na Europa - liga faturas eletronicas diretamente a deducoes fiscais, incentivando a exigencia de fatura.",
        sources=["Portal das Financas - efatura.gov.pt", "Autoridade Tributaria"],
    ),
    QAPair(
        id="PT-BUD-003",
        question="Como faco um planeamento financeiro anual em Portugal?",
        answer=(
            "Um planeamento anual em Portugal deve considerar datas especificas: "
            "Janeiro: renovacao de seguros, pagamento de IMI (1a prestacao se parcelado); "
            "Marco-abril: entrega da declaracao de IRS; "
            "Abril-maio: pagamento de IUC (imposto sobre veiculos); "
            "Junho: subsidio de ferias (se nao em duodecimos); "
            "Setembro: despesas com regresso as aulas; "
            "Novembro: subsidio de Natal; "
            "Dezembro: despesas natalcias. "
            "Dica: crie um fundo para despesas anuais previsveis (impostos, seguros, inspecao do carro) "
            "e va alimentando mensalmente."
        ),
        country="PT",
        topic="budget",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Calendario anual com 12 meses, cada mes com icones das despesas tipicas portuguesas. "
            "Meses com mais despesas destacados em vermelho, meses com entradas extra em verde."
        ),
        cultural_notes="O calendario fiscal portugues tem datas especificas (IRS, IMI, IUC) que devem ser consideradas no planeamento.",
        sources=["Autoridade Tributaria e Aduaneira", "DECO Proteste"],
    ),
    QAPair(
        id="PT-BUD-004",
        question="Quanto custa o passe metropolitano e como poupar nos transportes?",
        answer=(
            "O passe unico metropolitano custa 40 euros/mes em Lisboa e no Porto, "
            "dando acesso ilimitado a todos os transportes publicos da area metropolitana. "
            "Para jovens ate 23 anos com Cartao Jovem: passe gratuito (sub23@educacao). "
            "Para maiores de 65 anos ou reformados: tambem existem tarifarios reduzidos. "
            "Dicas de poupanca: se nao usa diariamente, compare com bilhetes avulsos (Zapping); "
            "Zapping oferece desconto de cerca de 20% sobre o bilhete normal. "
            "Para deslocacoes interurbanas: CP (comboios) oferece desconto de 25% com antecedencia."
        ),
        country="PT",
        topic="budget",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="O passe metropolitano a 40 euros implementado em 2019 foi uma das maiores reformas de transportes em Portugal.",
        sources=["Transportes Metropolitanos de Lisboa", "Metro do Porto"],
    ),
    QAPair(
        id="PT-BUD-005",
        question="O comparador da ERSE ajuda mesmo a poupar na eletricidade?",
        answer=(
            "Sim, o simulador da ERSE (Entidade Reguladora dos Servicos Energeticos) permite comparar "
            "todas as ofertas de eletricidade e gas natural disponiveis no mercado liberalizado. "
            "Como usar: aceda a simulador.erse.pt, insira os dados da sua fatura atual "
            "(potencia contratada, consumo medio mensal), e o simulador mostra as ofertas ordenadas por preco. "
            "A poupanca pode chegar a 10-20% anuais ao mudar de comercializador. "
            "A mudanca e gratuita e sem interrupcao de fornecimento. "
            "Dica: faca a comparacao pelo menos uma vez por ano, pois as tarifas mudam frequentemente. "
            "Considere tambem tarifas bi-horarias ou tri-horarias se concentrar consumos a noite."
        ),
        country="PT",
        topic="budget",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="A ERSE e o regulador energetico portugues. O mercado liberalizado permite ao consumidor escolher o fornecedor de energia.",
        sources=["ERSE - simulador.erse.pt"],
    ),
    # --- savings ---
    QAPair(
        id="PT-SAV-001",
        question="O que sao Certificados de Aforro e como subscrevo?",
        answer=(
            "Os Certificados de Aforro sao instrumentos de poupanca emitidos pelo Estado portugues, "
            "exclusivos para particulares. Caracteristicas: "
            "Capital garantido pelo Estado; remuneracao indexada a Euribor a 3 meses + premio de permanencia; "
            "prazo maximo de 15 anos; subscricao minima de 100 euros (multiplos de 100); "
            "maximo de 250.000 euros por titular; resgate apos 3 meses sem penalizacao. "
            "Tributacao: 28% sobre juros (pode optar por englobamento no IRS se vantajoso). "
            "Para subscrever: nos balcoes dos CTT (Correios) com CC ou BI e NIF, "
            "ou online atraves do AforroNet (apos registo presencial nos CTT)."
        ),
        country="PT",
        topic="savings",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Ficha resumo dos Certificados de Aforro: montante minimo, montante maximo, "
            "taxa de juro atual, prazo, tributacao, e local de subscricao. "
            "Com selo do Estado portugues para transmitir seguranca."
        ),
        cultural_notes="Os Certificados de Aforro sao um dos produtos de poupanca mais tradicionais e populares em Portugal, com garantia do Estado.",
        sources=["IGCP - Agencia de Gestao da Tesouraria e da Divida Publica"],
    ),
    QAPair(
        id="PT-SAV-002",
        question="O que e um PPR e quais os beneficios fiscais?",
        answer=(
            "PPR (Plano Poupanca Reforma) e um produto de poupanca destinado a reforma/aposentadoria. "
            "Beneficios fiscais na entrada (deducao no IRS): "
            "Ate 35 anos: 20% do investido, maximo 400 euros de deducao; "
            "35 a 50 anos: 20% do investido, maximo 350 euros de deducao; "
            "Mais de 50 anos: 20% do investido, maximo 300 euros de deducao. "
            "Beneficio fiscal na saida: taxa reduzida de 8% (em vez de 28%) se resgatado nas condicoes legais "
            "(reforma, desemprego longa duracao, doenca grave, credito habitacao). "
            "Existem PPR em forma de seguro (capital garantido) e em forma de fundo (sem garantia, potencial maior). "
            "Atencao: PPR com comissoes altas podem anular o beneficio fiscal."
        ),
        country="PT",
        topic="savings",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Tabela comparativa por faixa etaria: idade, deducao maxima, investimento necessario para deducao maxima. "
            "Abaixo, comparacao PPR-seguro vs PPR-fundo com vantagens e desvantagens de cada um."
        ),
        cultural_notes="Os PPR sao incentivados fiscalmente pelo Estado portugues como complemento a Seguranca Social.",
        sources=["Autoridade de Supervisao de Seguros e Fundos de Pensoes (ASF)", "CMVM"],
    ),
    QAPair(
        id="PT-SAV-003",
        question="Qual a melhor forma de poupar para a entrada de uma casa em Portugal?",
        answer=(
            "Para comprar casa em Portugal, precisa tipicamente de 10-20% do valor para entrada, "
            "mais cerca de 5-8% para impostos e despesas (IMT, Imposto de Selo, escritura, registos). "
            "Estrategia de poupanca: "
            "1) Defina o valor-alvo (ex: casa de 200.000 euros -> entrada de 20.000 + despesas de 12.000 = 32.000 euros); "
            "2) Para prazo de 2-3 anos: Certificados de Aforro ou deposito a prazo; "
            "3) Para prazo de 5+ anos: PPR (com beneficio fiscal adicional que pode ser usado para credito habitacao); "
            "4) Use o subsidio de ferias e Natal para reforcar a poupanca; "
            "5) O programa Porta 65 pode apoiar jovens ate 35 anos com comparticipacao na renda enquanto poupam."
        ),
        country="PT",
        topic="savings",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Barra de progresso tipo meta financeira mostrando: preco do imovel -> entrada necessaria "
            "-> impostos e custos -> total a poupar. Com timeline indicando marcos mensais de poupanca."
        ),
        cultural_notes="O mercado imobiliario portugues sofreu forte valorizacao pos-2015, tornando a entrada para primeira casa um desafio crescente.",
        sources=["Banco de Portugal", "Portal da Habitacao - gov.pt"],
    ),
    QAPair(
        id="PT-SAV-004",
        question="Depositos a prazo ainda compensam com as taxas atuais?",
        answer=(
            "Os depositos a prazo em Portugal oferecem taxas baixas comparadas com alternativas: "
            "a taxa media ronda 1-2% para prazos de 12 meses (muito abaixo da media europeia). "
            "Alternativas a considerar: "
            "1) Certificados de Aforro: indexados a Euribor, geralmente rendem mais; "
            "2) Certificados do Tesouro Poupanca Crescimento (CTPC): taxas crescentes ao longo de 7 anos; "
            "3) Fundos de obrigacoes de curto prazo; "
            "4) ETFs de obrigacoes europeias. "
            "O deposito a prazo pode fazer sentido se: o banco oferece taxa promocional superior, "
            "ou se valoriza a simplicidade e garantia do Fundo de Garantia de Depositos (ate 100.000 euros)."
        ),
        country="PT",
        topic="savings",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="Os bancos portugueses oferecem taxas de deposito consistentemente abaixo da media da zona euro.",
        sources=["Banco de Portugal - Taxas de juro", "IGCP"],
    ),
    QAPair(
        id="PT-SAV-005",
        question="Como funciona o Fundo de Garantia de Depositos em Portugal?",
        answer=(
            "O Fundo de Garantia de Depositos (FGD) protege os depositos dos clientes bancarios em Portugal. "
            "Cobertura: ate 100.000 euros por depositante, por instituicao. "
            "Abrange: depositos a ordem, depositos a prazo, certificados de deposito. "
            "NAO abrange: produtos de investimento (acoes, fundos, obrigacoes), depositos de instituicoes financeiras. "
            "Em caso de insolvencia do banco, o FGD paga no prazo de 7 dias uteis. "
            "Dica: se tem mais de 100.000 euros, distribua por diferentes bancos para maximizar a protecao. "
            "O FGD portugues esta integrado no sistema europeu de garantia de depositos."
        ),
        country="PT",
        topic="savings",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="Apos a crise do BES/Novo Banco em 2014, a consciencia sobre a protecao de depositos aumentou significativamente em Portugal.",
        sources=["Fundo de Garantia de Depositos - fgd.pt", "Banco de Portugal"],
    ),
    # --- debt ---
    QAPair(
        id="PT-DEB-001",
        question="O que e a TAEG e por que e importante quando peco credito?",
        answer=(
            "TAEG (Taxa Anual de Encargos Efetiva Global) e o indicador que mostra o custo TOTAL "
            "de um credito, incluindo juros, comissoes, seguros obrigatorios e impostos. "
            "E mais fiavel que a TAN (Taxa Anual Nominal) porque esta so mostra os juros. "
            "Exemplo: um credito pessoal pode ter TAN de 7% mas TAEG de 12% apos incluir todos os custos. "
            "Por lei, todas as instituicoes financeiras sao obrigadas a apresentar a TAEG. "
            "Regra fundamental: compare SEMPRE pela TAEG, nunca pela TAN. "
            "A TAEG maxima permitida pelo Banco de Portugal varia trimestralmente conforme tipo de credito."
        ),
        country="PT",
        topic="debt",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Comparacao visual entre TAN e TAEG: dois circulos concentricos mostrando "
            "TAN (circulo interno = so juros) vs TAEG (circulo externo = juros + comissoes + seguros + impostos). "
            "Com exemplo numerico de credito pessoal."
        ),
        cultural_notes="A TAEG e obrigatoria por regulamento do Banco de Portugal e diretiva europeia de credito ao consumo.",
        sources=["Banco de Portugal - Portal do Cliente Bancario"],
    ),
    QAPair(
        id="PT-DEB-002",
        question="Como funciona o credito habitacao e o que e o spread?",
        answer=(
            "O credito habitacao em Portugal financia ate 90% do valor do imovel (ou da avaliacao bancaria). "
            "A prestacao tem dois componentes: "
            "1) Euribor: taxa de referencia europeia (ex: Euribor a 6 ou 12 meses) - varia com o mercado; "
            "2) Spread: margem fixa do banco (tipicamente 0,5% a 2%) - negociavel conforme seu perfil. "
            "Taxa total = Euribor + Spread. Pode ser taxa variavel, fixa, ou mista. "
            "Custos adicionais: comissao de avaliacao, comissao de dossier, seguro de vida, seguro multiriscos, IMT, Imposto de Selo. "
            "Dica: use o simulador do Banco de Portugal e compare pelo menos 3 bancos. "
            "A taxa de esforco (prestacao/rendimento) nao deve ultrapassar 35%."
        ),
        country="PT",
        topic="debt",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Diagrama mostrando a composicao da prestacao mensal: Euribor (parte variavel, com seta indicando que flutua) "
            "+ Spread (parte fixa do banco). Abaixo, grafico de evolucao da Euribor nos ultimos 5 anos."
        ),
        cultural_notes="A Euribor impacta diretamente milhoes de familias portuguesas com credito habitacao a taxa variavel.",
        sources=["Banco de Portugal - Portal do Cliente Bancario", "Euribor-rates.eu"],
    ),
    QAPair(
        id="PT-DEB-003",
        question="Posso transferir o meu credito habitacao para outro banco?",
        answer=(
            "Sim, pode fazer transferencia de credito habitacao (portabilidade) para outro banco. "
            "Vantagens: negociar spread mais baixo, melhores condicoes de seguro, bonificacoes. "
            "Processo: 1) Peca simulacao a outros bancos com os dados do seu credito atual; "
            "2) O novo banco avalia o imovel e o seu perfil; "
            "3) Se aprovar, o novo banco paga ao antigo e fica com o seu credito; "
            "4) Os custos de transferencia incluem: nova avaliacao, escritura de hipoteca, registo predial. "
            "Regra pratica: a poupanca nos juros deve compensar os custos de transferencia em ate 2-3 anos. "
            "O banco atual pode fazer contraproposta - use isso como alavanca de negociacao."
        ),
        country="PT",
        topic="debt",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="Com as subidas da Euribor em 2022-2024, muitas familias portuguesas recorreram a transferencia de credito para reduzir prestacoes.",
        sources=["Banco de Portugal", "DECO Proteste"],
    ),
    QAPair(
        id="PT-DEB-004",
        question="O que faco se nao consigo pagar as prestacoes do credito?",
        answer=(
            "Se esta com dificuldades, atue ANTES de entrar em incumprimento: "
            "1) Contacte o banco imediatamente - explique a situacao; "
            "2) Solicite o PERSI (Procedimento Extrajudicial de Regularizacao de Situacoes de Incumprimento) "
            "- o banco e obrigado a avaliar solucoes; "
            "3) Opcoes possiveis: alargamento do prazo, periodo de carencia, reducao de spread, consolidacao de creditos; "
            "4) Recorra ao DECO (associacao de defesa do consumidor) para mediacao gratuita; "
            "5) Em ultimo caso, a insolvencia pessoal permite reestruturar dividas com plano de pagamento. "
            "Atencao: nao ignore as cartas do banco - o incumprimento agrava os juros e pode levar a execucao da hipoteca."
        ),
        country="PT",
        topic="debt",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="O PERSI e um mecanismo especifico do sistema bancario portugues, obrigando os bancos a negociar antes de executar dividas.",
        sources=["Banco de Portugal", "DECO - deco.proteste.pt"],
    ),
    QAPair(
        id="PT-DEB-005",
        question="O credito ao consumo em Portugal tem limites? Quanto posso pedir?",
        answer=(
            "Sim, o Banco de Portugal impoe limites prudenciais: "
            "Taxa de esforco maxima recomendada: 50% do rendimento liquido (somando TODAS as prestacoes); "
            "Prazo maximo: 10 anos para credito pessoal, 10 anos para credito automovel; "
            "TAEG maxima: definida trimestralmente pelo BdP (varia por montante e prazo); "
            "Os bancos devem avaliar a sua capacidade financeira antes de aprovar. "
            "Dica: antes de pedir credito, calcule se a prestacao cabe no seu orcamento sem comprometer "
            "despesas essenciais. Utilize o simulador do Banco de Portugal."
        ),
        country="PT",
        topic="debt",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="O Banco de Portugal implementou medidas macroprudenciais rigorosas apos a crise financeira de 2011-2014.",
        sources=["Banco de Portugal - Recomendacao macroprudencial"],
    ),
    # --- investment ---
    QAPair(
        id="PT-INV-001",
        question="Como investir em ETFs a partir de Portugal?",
        answer=(
            "Em Portugal, pode investir em ETFs europeus (UCITS) e, em alguns casos, americanos. "
            "Passos: 1) Abra conta numa corretora: Degiro, Interactive Brokers, XTB ou ActivoBank; "
            "2) Escolha ETFs acumulativos (reinvestem dividendos) para eficiencia fiscal; "
            "3) ETFs populares: Vanguard FTSE All-World (VWCE), iShares Core MSCI World (IWDA); "
            "4) Tributacao: mais-valias tributadas a 28% (ou englobamento se vantajoso). "
            "Atencao: ETFs americanos (SPY, VOO) estao restringidos para investidores europeus "
            "pela regulacao PRIIPs - opte pelas versoes UCITS (domiciliadas na Irlanda geralmente). "
            "Estrategia simples e eficaz: investimento periodico (DCA) num ETF mundial diversificado."
        ),
        country="PT",
        topic="investment",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Diagrama de fluxo: Escolher corretora -> Selecionar ETF UCITS -> Definir montante mensal -> "
            "Comprar periodicamente. Com exemplos de 3 ETFs populares e seus codigos ISIN."
        ),
        cultural_notes="A regulacao europeia PRIIPs limita o acesso de investidores europeus a ETFs americanos, diferenciando a experiencia de investimento.",
        sources=["CMVM", "Regulamento PRIIPs (UE)"],
    ),
    QAPair(
        id="PT-INV-002",
        question="PPR como investimento para a reforma: quanto preciso investir por mes?",
        answer=(
            "Depende da reforma desejada e do tempo ate la. Exemplo pratico: "
            "Objetivo: complemento de 500 euros/mes na reforma (em valores atuais). "
            "Se tem 30 anos e quer reformar-se aos 66: 36 anos de investimento. "
            "Com rentabilidade media de 5% anual (PPR de maior risco): precisa investir cerca de 175 euros/mes. "
            "Com rentabilidade de 3% (PPR conservador): precisa de cerca de 265 euros/mes. "
            "Beneficio fiscal extra: ao investir 2.000 euros/ano num PPR (ate 35 anos), "
            "deduz 400 euros no IRS - retorno imediato de 20%. "
            "Combine: PPR para o beneficio fiscal + ETFs para diversificacao = estrategia robusta para a reforma."
        ),
        country="PT",
        topic="investment",
        difficulty="advanced",
        modalities=["text", "image"],
        image_description=(
            "Grafico de projecao mostrando crescimento do investimento ao longo de 36 anos "
            "com contribuicoes mensais de 175 euros e rentabilidade de 5%. "
            "Destaque para o efeito dos juros compostos na segunda metade do periodo."
        ),
        cultural_notes="A taxa de substituicao da Seguranca Social portuguesa tem vindo a diminuir, tornando a poupanca complementar cada vez mais importante.",
        sources=["Seguranca Social - seg-social.pt", "ASF"],
    ),
    # --- credit ---
    QAPair(
        id="PT-CRE-001",
        question="O que e a Central de Responsabilidades de Credito do Banco de Portugal?",
        answer=(
            "A Central de Responsabilidades de Credito (CRC) e uma base de dados gerida pelo Banco de Portugal "
            "que regista todos os creditos contratados em Portugal: emprestimos, cartoes de credito, "
            "descobertos, garantias, leasings, etc. "
            "Qualquer credito acima de 50 euros e registado. "
            "Para que serve: os bancos consultam a CRC antes de conceder credito, "
            "para avaliar o seu nivel de endividamento. "
            "Pode consultar o seu mapa gratuitamente: online no site do BdP (com Chave Movel Digital), "
            "presencialmente nas agencias do BdP, ou por correio. "
            "O mapa mostra: creditos ativos, montantes em divida, situacao de cumprimento. "
            "Nao e um 'score' - e um registo factual."
        ),
        country="PT",
        topic="credit",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="Ao contrario do sistema de score brasileiro (Serasa), Portugal usa um registo factual centralizado no Banco de Portugal.",
        sources=["Banco de Portugal - Central de Responsabilidades de Credito"],
    ),
    QAPair(
        id="PT-CRE-002",
        question="Como funciona a Chave Movel Digital para servicos financeiros?",
        answer=(
            "A Chave Movel Digital (CMD) e um sistema de autenticacao digital do Estado portugues. "
            "Permite: consultar a CRC no Banco de Portugal, aceder ao Portal das Financas (IRS, e-fatura), "
            "assinar documentos digitalmente, aceder a servicos da Seguranca Social. "
            "Para ativar: presencialmente num balcao de atendimento (Loja do Cidadao, Espacos Cidadao, CTT) "
            "com CC e telemovel; ou online se ja tiver leitor de cartao de cidadao. "
            "Funciona com numero de telemovel + PIN + codigo SMS. "
            "E essencial para gerir a vida financeira digital em Portugal."
        ),
        country="PT",
        topic="credit",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="A CMD e um dos pilares da administracao digital portuguesa, cada vez mais necessaria para servicos publicos e financeiros.",
        sources=["AMA - Agencia para a Modernizacao Administrativa", "autenticacao.gov.pt"],
    ),
    # --- cultural ---
    QAPair(
        id="PT-CUL-001",
        question="Como funciona o IRS em Portugal? Devo fazer a declaracao automatica?",
        answer=(
            "O IRS (Imposto sobre o Rendimento das Pessoas Singulares) e o imposto sobre rendimentos em Portugal. "
            "A declaracao e obrigatoria para quem teve rendimentos tributaveis. "
            "Declaracao automatica: disponivel para situacoes simples (rendimentos da categoria A/H, "
            "sem rendimentos no estrangeiro, sem deducoes especiais). A AT pre-preenche os dados. "
            "Recomendacao: SEMPRE verifique os dados antes de aceitar. Compare o valor da declaracao automatica "
            "com uma simulacao manual - por vezes ha deducoes nao incluidas. "
            "Prazo: 1 de abril a 30 de junho. "
            "Taxas: progressivas de 14,5% a 48% conforme escalao de rendimento. "
            "Dica: considere englobamento de rendimentos de capitais se a sua taxa marginal for inferior a 28%."
        ),
        country="PT",
        topic="cultural",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Tabela de escaloes do IRS com taxas progressivas: 1o escalao ate 7.703 euros (14,5%), "
            "ate ao ultimo escalao (48%). Com seta mostrando onde se aplica cada taxa."
        ),
        cultural_notes="O sistema fiscal portugues tem complexidades proprias, incluindo tributacao autonoma e englobamento opcional de rendimentos de capitais.",
        sources=["Autoridade Tributaria e Aduaneira - portaldasfinancas.gov.pt"],
    ),
    QAPair(
        id="PT-CUL-002",
        question="Que apoios sociais existem em Portugal para familias com dificuldades?",
        answer=(
            "Principais apoios sociais em Portugal: "
            "1) RSI (Rendimento Social de Insercao): para quem esta em situacao de pobreza extrema; "
            "2) Abono de familia: prestacao mensal por filho, escaloes conforme rendimento; "
            "3) CSI (Complemento Solidario para Idosos): complementa pensoes muito baixas; "
            "4) Subsidio de desemprego: para quem contribuiu para a Seguranca Social e ficou desempregado; "
            "5) Tarifas sociais de energia e agua: descontos para familias carenciadas; "
            "6) ADSE: subsistema de saude para funcionarios publicos (desconto em saude); "
            "7) Programa de apoio a renda: prestacoes reduzidas para familias com taxa de esforco elevada. "
            "Inscricao: geralmente atraves da Seguranca Social (online ou presencial)."
        ),
        country="PT",
        topic="cultural",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="O Estado social portugues oferece uma rede de protecao que, embora menos generosa que os paises nordicos, cobre as principais situacoes de vulnerabilidade.",
        sources=["Seguranca Social - seg-social.pt", "DGSS"],
    ),
    # --- investment (extra) ---
    QAPair(
        id="PT-INV-003",
        question="O que sao Certificados do Tesouro Poupanca Crescimento (CTPC)?",
        answer=(
            "Os CTPC sao titulos de divida publica portuguesa com taxa crescente ao longo de 7 anos. "
            "Caracteristicas: capital garantido pelo Estado, taxa de juro fixa crescente por ano, "
            "acrescida de um premio ligado ao crescimento do PIB a partir do 2o ano. "
            "Subscricao: minimo de 1.000 euros, maximo de 1.000.000 euros por titular. "
            "Resgate: apos 1 ano, sem penalizacao (perde apenas os juros do ano corrente). "
            "Tributacao: 28% sobre juros (ou englobamento). "
            "Subscreve-se online no AforroNet ou presencialmente nos CTT (Correios). "
            "E uma opcao interessante para prazos mais longos que os Certificados de Aforro."
        ),
        country="PT",
        topic="investment",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="Os CTPC substituiram os antigos CTPM e oferecem rendimento crescente como incentivo a poupanca de longo prazo.",
        sources=["IGCP - igcp.pt"],
    ),
    QAPair(
        id="PT-INV-004",
        question="Como funciona a tributacao de investimentos em Portugal?",
        answer=(
            "Em Portugal, os rendimentos de investimentos sao tributados da seguinte forma: "
            "Juros (depositos, obrigacoes): 28% de retencao na fonte (pode englobar no IRS). "
            "Dividendos: 28% de retencao (pode englobar). "
            "Mais-valias de acoes/ETFs: 28% sobre o ganho (diferenca entre venda e compra). "
            "Mais-valias detidas mais de 365 dias: englobamento parcial (50% da mais-valia). "
            "PPR: taxa reduzida de 8% se resgatado nas condicoes legais. "
            "Englobamento: pode optar por somar rendimentos de capitais ao IRS - vantajoso se a taxa marginal for inferior a 28%. "
            "Dica: simule sempre os dois cenarios (retencao autonoma vs englobamento) antes de entregar o IRS."
        ),
        country="PT",
        topic="investment",
        difficulty="advanced",
        modalities=["text", "image"],
        image_description=(
            "Tabela resumo de tributacao: tipo de rendimento | taxa autonoma | opcao de englobamento | notas. "
            "Com destaque para a regra dos 365 dias em mais-valias."
        ),
        cultural_notes="O sistema fiscal portugues permite opcao entre retencao autonoma e englobamento, criando oportunidades de otimizacao fiscal.",
        sources=["Autoridade Tributaria e Aduaneira", "Codigo do IRS"],
    ),
    # --- credit (extra) ---
    QAPair(
        id="PT-CRE-003",
        question="O que e a taxa de esforco e como a calculo?",
        answer=(
            "A taxa de esforco e a percentagem do rendimento liquido mensal que vai para prestacoes de credito. "
            "Calculo: soma de todas as prestacoes mensais / rendimento liquido mensal x 100. "
            "Exemplo: rendimento de 1.500 euros, prestacao habitacao 450 euros + prestacao auto 150 euros = "
            "taxa de esforco de 40%. "
            "Limites recomendados pelo Banco de Portugal: idealmente ate 35%, maximo absoluto de 50%. "
            "Acima de 50%, os bancos geralmente nao concedem mais credito. "
            "Para reduzir: amortize creditos mais caros, alargue prazos, renegocie spreads, "
            "ou consolide creditos. "
            "Avalie a sua taxa de esforco ANTES de pedir qualquer novo credito."
        ),
        country="PT",
        topic="credit",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Grafico tipo gauge/velocimetro: 0-35% (verde/saudavel), 35-50% (amarelo/atencao), "
            "50%+ (vermelho/risco). Com formula de calculo e exemplo numerico."
        ),
        cultural_notes="As medidas macroprudenciais do Banco de Portugal desde 2018 limitam formalmente a taxa de esforco na concessao de novo credito.",
        sources=["Banco de Portugal - Medidas macroprudenciais"],
    ),
    QAPair(
        id="PT-CRE-004",
        question="O que e a consolidacao de creditos e quando faz sentido?",
        answer=(
            "Consolidacao de creditos e juntar varios creditos (pessoal, automovel, cartoes) num unico emprestimo. "
            "Vantagens: prestacao mensal unica (geralmente mais baixa), taxa de juro potencialmente menor, "
            "gestao mais simples. "
            "Desvantagens: o prazo total aumenta (paga mais juros no total), pode incluir custos de processo, "
            "e se incluir credito habitacao, a casa fica como garantia de TODOS os creditos. "
            "Faz sentido quando: tem 3 ou mais creditos, a taxa de esforco esta acima de 40%, "
            "ou os juros dos creditos individuais sao muito altos. "
            "Compare propostas de varios bancos e intermediarios de credito. "
            "Cuidado com empresas que prometem 'juntar tudo' - verifique se sao licenciadas pelo Banco de Portugal."
        ),
        country="PT",
        topic="credit",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="A consolidacao de creditos cresceu em Portugal apos a subida da Euribor, com muitas familias a procurar reduzir prestacoes mensais.",
        sources=["Banco de Portugal", "DECO Proteste"],
    ),
    # --- cultural (extra) ---
    QAPair(
        id="PT-CUL-003",
        question="Como funciona o MB WAY e quais cuidados devo ter?",
        answer=(
            "O MB WAY e um servico de pagamentos e transferencias por telemovel, ligado ao seu cartao bancario. "
            "Funcionalidades: transferencias instantaneas (gratis ate 750 euros/dia), "
            "pagamentos em lojas por QR code ou NFC, compras online, levantamentos sem cartao. "
            "Cuidados de seguranca: "
            "1) NUNCA partilhe o seu codigo MB WAY com ninguem - golpistas pedem o codigo para 'transferir' dinheiro; "
            "2) O golpe mais comum: alguem no OLX/Marketplace pede o seu numero de telefone e codigo MB WAY "
            "'para fazer a transferencia' - na verdade esta a associar a SUA conta ao telemovel DELE; "
            "3) Defina limites diarios de transferencia; "
            "4) Ative notificacoes para cada operacao; "
            "5) Nunca insira o seu PIN a pedido de terceiros. "
            "O MB WAY e seguro - o problema sao os golpes de engenharia social."
        ),
        country="PT",
        topic="cultural",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Infografico de alerta com 3 exemplos de golpes MB WAY: OLX scam, falso suporte tecnico, "
            "e pedido de codigo por SMS. Cada um com descricao curta e como evitar."
        ),
        cultural_notes="O MB WAY e um dos sistemas de pagamento movel mais usados na Europa, com penetracao massiva em Portugal. Os golpes sao uma preocupacao crescente.",
        sources=["SIBS", "Policia Judiciaria"],
    ),
    QAPair(
        id="PT-CUL-004",
        question="O que devo saber sobre o sistema de reformas em Portugal?",
        answer=(
            "A reforma (aposentadoria) em Portugal funciona atraves da Seguranca Social. "
            "Idade legal: 66 anos e 7 meses (2024), ajustada anualmente conforme esperanca de vida. "
            "Reforma antecipada: possivel a partir dos 60 anos, com penalizacao de 0,5% por cada mes de antecipacao. "
            "Calculo: baseado nos melhores 40 anos de contribuicoes (desde 2002), com taxa de formacao de 2% por ano. "
            "Taxa de substituicao: 50-70% do salario medio de referencia (quanto mais ganha, menor a taxa). "
            "Complementos essenciais: PPR com beneficio fiscal, ETFs para longo prazo. "
            "Simule a sua reforma no site da Seguranca Social (Seguranca Social Direta). "
            "Conselho: comece a poupar para a reforma o mais cedo possivel - os juros compostos fazem a diferenca."
        ),
        country="PT",
        topic="cultural",
        difficulty="advanced",
        modalities=["text", "image"],
        image_description=(
            "Grafico mostrando: salario medio vs pensao estimada (gap previdenciario). "
            "Linha de tempo desde a idade atual ate a reforma, com contribuicoes necessarias para complementar."
        ),
        cultural_notes="O sistema de reformas portugues enfrenta desafios demograficos serios com o envelhecimento da populacao.",
        sources=["Seguranca Social - seg-social.pt"],
    ),
    QAPair(
        id="PT-CUL-005",
        question="Como funcionam os imoveis e o IMI em Portugal?",
        answer=(
            "O IMI (Imposto Municipal sobre Imoveis) e pago anualmente por todos os proprietarios de imoveis em Portugal. "
            "Taxas: entre 0,3% e 0,45% do VPT (Valor Patrimonial Tributario) para predios urbanos. "
            "Isencoes: imoveis adquiridos para habitacao propria e permanente podem ter isencao de 3 anos "
            "se o VPT nao exceder 125.000 euros e o rendimento do agregado nao exceder 153.300 euros. "
            "Pagamento: ate 100 euros paga em abril; 100-500 euros em duas prestacoes (abril e novembro); "
            "mais de 500 euros em tres prestacoes (abril, julho, novembro). "
            "O VPT pode ser atualizado a seu pedido se o imovel desvalorizou. "
            "Familias com dependentes podem ter desconto adicional no IMI."
        ),
        country="PT",
        topic="cultural",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="O IMI e uma das receitas mais importantes para os municipios portugueses, com taxas que variam de concelho para concelho.",
        sources=["Autoridade Tributaria e Aduaneira", "Codigo do IMI"],
    ),
    QAPair(
        id="PT-BUD-006",
        question="Como gerir despesas de saude em Portugal alem do SNS?",
        answer=(
            "O SNS (Servico Nacional de Saude) oferece cobertura universal, mas com tempos de espera. "
            "Opcoes complementares: "
            "1) Seguro de saude: cobre consultas privadas, exames e cirurgias; precos desde 15-30 euros/mes; "
            "2) ADSE: subsistema para funcionarios publicos - coparticipa despesas no privado; "
            "3) Seguros de grupo: muitas empresas oferecem seguro de saude como beneficio. "
            "Deducoes fiscais: 15% das despesas de saude, ate 1.000 euros no IRS. "
            "Para poupar: use o SNS para consultas de rotina, o seguro para urgencias e especialidades; "
            "peca sempre fatura com NIF; compare precos entre clinicas privadas; "
            "analise se o seguro compensa face as suas despesas tipicas."
        ),
        country="PT",
        topic="budget",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="O SNS portugues e universal mas enfrenta problemas de falta de medicos de familia e listas de espera, levando muitas familias a recorrer ao privado.",
        sources=["SNS", "ASF - Autoridade de Supervisao de Seguros"],
    ),
    QAPair(
        id="PT-DEB-006",
        question="O que e a amortizacao antecipada do credito habitacao e compensa?",
        answer=(
            "Amortizacao antecipada e pagar parte ou totalidade do credito habitacao antes do prazo. "
            "Custos: a comissao de reembolso antecipado e no maximo 0,5% do capital amortizado (taxa variavel) "
            "ou 2% (taxa fixa). Em 2023-2024, o governo suspendeu temporariamente esta comissao. "
            "Quando compensa: se tem poupanca a render menos que a taxa do credito (Euribor + spread); "
            "se quer reduzir a prestacao mensal ou o prazo do emprestimo. "
            "Quando NAO compensa: se a poupanca rende mais que a taxa do credito (raro com Euribor alta); "
            "se ficar sem reserva de emergencia ao amortizar; "
            "se pode investir o dinheiro com retorno superior (considere o risco). "
            "Pode amortizar parcialmente (reduz prestacao ou prazo) ou totalmente."
        ),
        country="PT",
        topic="debt",
        difficulty="advanced",
        modalities=["text", "image"],
        image_description=(
            "Comparacao em duas colunas: cenario A (amortizar 10.000 euros) vs cenario B (investir 10.000 euros). "
            "Mostrando poupanca em juros no cenario A vs retorno potencial no cenario B ao longo de 5 anos."
        ),
        cultural_notes="Com a subida da Euribor, muitas familias portuguesas aceleraram a amortizacao do credito habitacao como estrategia de poupanca.",
        sources=["Banco de Portugal", "Decreto-Lei 74-A/2017"],
    ),
]


# ---------------------------------------------------------------------------
# ANGOLA (AO) — 30 QA pairs
# ---------------------------------------------------------------------------

AO_PAIRS: list[QAPair] = [
    # --- budget ---
    QAPair(
        id="AO-BUD-001",
        question="Como faco orcamento se o meu rendimento vem de negocio informal?",
        answer=(
            "Mesmo com rendimento informal, o orcamento e essencial. "
            "Passo a passo adaptado: "
            "1) Anote TUDO o que ganha durante 1 mes (mesmo valores pequenos); "
            "2) Use o mes de menor ganho como base para as despesas fixas; "
            "3) Separe fisicamente o dinheiro em envelopes: aluguer, comida, transporte, poupanca; "
            "4) O metodo dos envelopes funciona bem quando nao se tem conta bancaria; "
            "5) Guarde pelo menos 10% do que ganha, mesmo que pouco - consistencia e mais importante que valor; "
            "6) Registre num caderno simples: entrada, saida, saldo. "
            "Se tiver telemovel: apps como o Multicaixa Express permitem guardar dinheiro digital "
            "com seguranca, sem precisar de conta bancaria tradicional."
        ),
        country="AO",
        topic="budget",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Ilustracao do metodo dos envelopes: 4-5 envelopes coloridos etiquetados "
            "(Aluguer, Alimentacao, Transporte, Poupanca, Emergencia) com valores exemplo em Kwanzas. "
            "Visual simples e acessivel."
        ),
        cultural_notes="Grande parte da populacao angolana opera na economia informal. Metodos analogicos de gestao financeira sao mais realistas que apps digitais para muitos.",
        sources=["Banco Nacional de Angola - Educacao Financeira"],
    ),
    QAPair(
        id="AO-BUD-002",
        question="Sou zungueira, como posso organizar melhor o dinheiro do meu negocio?",
        answer=(
            "Como zungueira, e importante separar o dinheiro do negocio do dinheiro pessoal: "
            "1) No inicio do dia, anote quanto tem de 'capital' (dinheiro para comprar mercadoria); "
            "2) No final do dia, conte tudo: capital + lucro; "
            "3) Retire o capital inicial e guarde separado - esse dinheiro NAO e para gastar; "
            "4) Do lucro, separe: despesas pessoais, poupanca (mesmo que pequena), fundo para dias ruins; "
            "5) Tente comprar em maior quantidade para ter desconto no fornecedor; "
            "6) Se possivel, registre-se no Multicaixa Express para guardar dinheiro em seguranca. "
            "O mais importante: nunca gaste o capital de giro (dinheiro para repor a mercadoria) "
            "em despesas pessoais - isso mata o negocio."
        ),
        country="AO",
        topic="budget",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Diagrama simples tipo fluxo: Vendas do dia -> Separar capital de giro (caixa 1) -> "
            "Lucro -> Dividir em: Despesas pessoais + Poupanca + Fundo de emergencia. "
            "Visual com ilustracao de uma zungueira com produtos na cabeca."
        ),
        cultural_notes="As zungueiras sao vendedoras ambulantes angolanas, geralmente mulheres, que representam uma forca economica enorme no pais. O termo vem do kimbundu 'kuzunga' (vender ambulando).",
        sources=["INE Angola", "ADRA Angola"],
    ),
    QAPair(
        id="AO-BUD-003",
        question="Como lidar com o aumento constante dos precos em Angola?",
        answer=(
            "A inflacao em Angola tem sido persistentemente alta, afetando especialmente alimentos e transportes. "
            "Estrategias praticas: "
            "1) Compre alimentos nao pereciveis em quantidade quando o preco estiver bom; "
            "2) Prefira mercados informais e produtores locais aos supermercados para produtos frescos; "
            "3) Reduza dependencia de produtos importados (que sofrem com a desvalorizacao do Kwanza); "
            "4) Se possivel, converta parte da poupanca para moeda mais estavel (USD) como reserva de valor; "
            "5) Invista em bens que acompanham a inflacao (terrenos, por exemplo); "
            "6) Ajuste o orcamento trimestralmente, nao anualmente - os precos mudam rapido."
        ),
        country="AO",
        topic="budget",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Grafico de barras mostrando a evolucao da inflacao em Angola nos ultimos 5 anos, "
            "com linha indicando o impacto no poder de compra de 10.000 Kz ao longo do tempo."
        ),
        cultural_notes="Angola enfrenta inflacao elevada ha anos, agravada pela dependencia do petroleo e desvalorizacao do Kwanza. Isso afeta desproporcionalmente as familias de baixa renda.",
        sources=["Banco Nacional de Angola", "INE Angola"],
    ),
    QAPair(
        id="AO-BUD-004",
        question="Quanto custa viver em Luanda atualmente? Como planear o meu orcamento?",
        answer=(
            "Luanda e uma das cidades mais caras de Africa para se viver. "
            "Custos medios aproximados (2025): "
            "Aluguer apartamento T1 periferia: 80.000-150.000 Kz/mes; "
            "Alimentacao basica familia 4 pessoas: 120.000-200.000 Kz/mes; "
            "Transporte (candongueiro): 200-500 Kz por viagem; "
            "Eletricidade (ENDE): variavel, frequentes cortes - muitos recorrem a gerador (diesel caro); "
            "Agua (EPAL): fornecimento irregular, agua por cisterna e alternativa cara. "
            "Dica: o orcamento em Luanda deve incluir margens maiores para imprevistos "
            "(cortes de energia, subidas subitas de precos, reparacoes). "
            "Se trabalha na economia formal, negocie beneficios como subsidio de transporte e alimentacao."
        ),
        country="AO",
        topic="budget",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="Luanda ja foi considerada a cidade mais cara do mundo para expatriados, refletindo distorcoes entre custos e rendimentos da populacao local.",
        sources=["INE Angola", "Mercer Cost of Living Survey"],
    ),
    QAPair(
        id="AO-BUD-005",
        question="Como gerir as despesas com educacao dos filhos em Angola?",
        answer=(
            "A educacao em Angola representa uma fatia significativa do orcamento familiar. "
            "Ensino publico: gratuito em teoria, mas com custos indiretos (material, fardamento, propinas informais); "
            "Ensino privado: qualidade variavel, custos de 30.000 a 300.000+ Kz/mes conforme a escola. "
            "Estrategias: "
            "1) Comece a poupar para educacao assim que o filho nascer - mesmo valores pequenos; "
            "2) Considere kixikila com outros pais para juntar o valor da matricula; "
            "3) Pesquise bolsas de estudo: universidades angolanas e programas como INAGBE oferecem bolsas; "
            "4) O material escolar e mais barato fora da epoca de regresso as aulas; "
            "5) Livros e material usados sao opcao valida. "
            "Investir na educacao dos filhos e o melhor investimento de longo prazo."
        ),
        country="AO",
        topic="budget",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="Em Angola, a educacao privada e vista como necessidade por muitas familias devido a limitacoes do ensino publico, criando pressao financeira significativa.",
        sources=["Ministerio da Educacao de Angola", "INAGBE"],
    ),
    # --- savings ---
    QAPair(
        id="AO-SAV-001",
        question="O que e a kixikila e como funciona?",
        answer=(
            "A kixikila e um sistema tradicional angolano de poupanca rotativa em grupo. "
            "Funcionamento: um grupo de pessoas (geralmente 10-20) contribui um valor fixo periodicamente. "
            "A cada rodada, um membro recebe o montante total. "
            "Exemplo: 10 pessoas contribuem 10.000 Kz por mes. Cada mes, uma pessoa recebe 100.000 Kz. "
            "Vantagens: nao precisa de banco, cria disciplina de poupanca, fortalece lacos comunitarios, "
            "permite acesso a montantes maiores que a poupanca individual. "
            "Cuidados: escolha membros de confianca, defina regras claras por escrito, "
            "combine o que acontece se alguem nao pagar, defina a ordem de recebimento antecipadamente. "
            "A kixikila e legal e pode complementar a poupanca formal."
        ),
        country="AO",
        topic="savings",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Diagrama circular mostrando 10 participantes em roda. "
            "Setas indicam contribuicoes de cada um para o centro. "
            "Uma pessoa destacada (a contemplada do mes) recebe o montante total. "
            "Calendario ao lado mostrando a rotacao ao longo dos meses."
        ),
        cultural_notes="A kixikila e similar as tontines africanas e ao tandas latino-americano. E uma instituicao financeira informal enraizada na cultura angolana, especialmente entre mulheres.",
        sources=["Banco Nacional de Angola - Inclusao Financeira", "PNUD Angola"],
    ),
    QAPair(
        id="AO-SAV-002",
        question="Poupanca no banco ou guardar dinheiro em casa? O que e mais seguro em Angola?",
        answer=(
            "Guardar em casa parece mais acessivel, mas tem riscos serios: "
            "roubo, incendio, tentacao de gastar, e perda de valor pela inflacao. "
            "Vantagens do banco: seguranca, possibilidade de rendimento (mesmo pequeno), "
            "acesso a outros servicos (transferencias, credito). "
            "Contas bancarias acessiveis em Angola: "
            "1) Conta a ordem basica (isenta de comissoes para saldos baixos em varios bancos); "
            "2) Deposito a prazo: juros de 7-15% ao ano (conforme prazo e banco); "
            "3) Multicaixa Express: conta digital acessivel com o telemovel. "
            "Dica: se nao confia totalmente no banco, faca uma estrategia mista - "
            "parte no banco, parte em kixikila, e uma pequena reserva em casa para emergencias imediatas."
        ),
        country="AO",
        topic="savings",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="A bancarizacao em Angola ronda 35-40% da populacao adulta. A desconfianca nos bancos tem raizes historicas em experiencias de desvalorizacao abrupta.",
        sources=["Banco Nacional de Angola", "Finscope Angola"],
    ),
    QAPair(
        id="AO-SAV-003",
        question="Como abrir uma conta bancaria em Angola? Que documentos preciso?",
        answer=(
            "Para abrir conta bancaria em Angola, geralmente precisa de: "
            "1) Bilhete de Identidade (BI) ou Passaporte angolano; "
            "2) NIF (Numero de Identificacao Fiscal) - obtido na AGT; "
            "3) Comprovativo de residencia (carta de um bairro, fatura de servico, ou declaracao); "
            "4) Comprovativo de rendimento (para algumas contas; para conta basica, pode nao ser necessario); "
            "5) Foto tipo passe (alguns bancos). "
            "Bancos com maior rede em Angola: BFA, BAI, BIC, BPC, Standard Bank, Atlantico. "
            "Alguns bancos oferecem abertura simplificada com apenas BI e NIF. "
            "Para Multicaixa Express: basta o BI e um telemovel com numero angolano - pode registar-se num agente."
        ),
        country="AO",
        topic="savings",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="A obtencao de documentos como o NIF pode ser um obstaculo em zonas rurais de Angola, dificultando o acesso a servicos bancarios.",
        sources=["Banco Nacional de Angola", "AGT - Administracao Geral Tributaria"],
    ),
    QAPair(
        id="AO-SAV-004",
        question="Os juros dos depositos a prazo em Angola compensam a inflacao?",
        answer=(
            "Os depositos a prazo em Angola oferecem taxas nominais relativamente altas "
            "(7-15% ao ano conforme o banco e o prazo). "
            "Porem, com a inflacao anual acima de 20% em varios anos recentes, "
            "o rendimento REAL (descontando a inflacao) e frequentemente negativo. "
            "Isto significa que o dinheiro no banco perde poder de compra, embora mais lentamente que parado em casa. "
            "Alternativas para proteger o poder de compra: "
            "1) Depositos em moeda estrangeira (USD, EUR) - se disponivel; "
            "2) Investir em bens que acompanham a inflacao (terrenos, materiais de construcao); "
            "3) Reforcar o negocio proprio (o retorno pode superar a inflacao); "
            "4) Obrigacoes do Tesouro angolano (OT) - taxas geralmente superiores aos depositos."
        ),
        country="AO",
        topic="savings",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Grafico comparativo: taxa de deposito a prazo (barra verde) vs taxa de inflacao (barra vermelha) "
            "ao longo de 5 anos em Angola. Destacando o rendimento real (diferenca entre as barras, frequentemente negativo)."
        ),
        cultural_notes="A relacao entre inflacao e rendimento de poupanca e uma preocupacao central em Angola, onde a desvalorizacao do Kwanza amplifica o problema.",
        sources=["Banco Nacional de Angola - Estatisticas Monetarias"],
    ),
    QAPair(
        id="AO-SAV-005",
        question="O que sao Obrigacoes do Tesouro (OT) em Angola e posso investir?",
        answer=(
            "As Obrigacoes do Tesouro sao titulos de divida publica emitidos pelo Estado angolano. "
            "Caracteristicas: prazo tipico de 1 a 10 anos, taxas de juro geralmente superiores aos depositos bancarios, "
            "disponiveis em Kwanzas e em Dolares (OT-USD). "
            "Para investir: "
            "1) Abra conta num banco angolano que opere no mercado de titulos; "
            "2) O investimento minimo varia (geralmente a partir de 100.000 Kz); "
            "3) O banco actua como intermediario na compra. "
            "Vantagens: rentabilidade superior aos depositos, garantia do Estado, diversificacao. "
            "Riscos: risco soberano (depende da saude financeira do Estado), liquidez limitada "
            "(pode ser dificil vender antes do vencimento). "
            "As OT em USD oferecem protecao contra a desvalorizacao do Kwanza."
        ),
        country="AO",
        topic="savings",
        difficulty="advanced",
        modalities=["text"],
        cultural_notes="O mercado de titulos publicos em Angola e relativamente jovem, com a BODIVA (Bolsa de Divida e Valores de Angola) a funcionar desde 2014.",
        sources=["BODIVA", "Ministerio das Financas de Angola"],
    ),
    # --- debt ---
    QAPair(
        id="AO-DEB-001",
        question="Quero pedir microcredito para o meu negocio. Como funciona em Angola?",
        answer=(
            "O microcredito em Angola destina-se a pequenos empreendedores, especialmente informais. "
            "Opcoes: "
            "1) Bancos comerciais com programas de microcredito: BPC, KNB, BDA (Banco de Desenvolvimento de Angola); "
            "2) Instituicoes de microfinancas: algumas operam em mercados locais; "
            "3) Programas governamentais: PRODESI, PROAPEN (apoio a producao nacional). "
            "Cuidados: "
            "- Compare SEMPRE as taxas de juro antes de aceitar; "
            "- Leia todo o contrato (peca ajuda se necessario); "
            "- Verifique se a instituicao e licenciada pelo Banco Nacional de Angola; "
            "- NUNCA recorra a agiotas (taxas abusivas, coacao para pagamento); "
            "- Faca um plano simples de como o credito vai gerar retorno no negocio."
        ),
        country="AO",
        topic="debt",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="O microcredito em Angola e essencial para o empreendedorismo informal, mas o acesso ainda e limitado fora de Luanda.",
        sources=["Banco Nacional de Angola", "BDA"],
    ),
    QAPair(
        id="AO-DEB-002",
        question="Quais os perigos de pedir dinheiro a agiotas?",
        answer=(
            "Os agiotas (emprestadores informais ilegais) sao extremamente perigosos: "
            "1) Taxas de juro absurdas: podem cobrar 30-100% ao MES (nao ao ano!); "
            "2) Coacao e violencia: usam ameacas e intimidacao para cobrar; "
            "3) Sem contrato legal: voce nao tem protecao nenhuma; "
            "4) Divida perpetua: os juros podem fazer a divida crescer sem fim; "
            "5) Podem exigir bens pessoais (telemovel, electrodomesticos) como garantia. "
            "Alternativas: kixikila com pessoas de confianca, microcredito formal, "
            "emprestimo de familiares com acordo claro, programa de microfinancas. "
            "Se ja esta numa situacao com agiota: procure ajuda legal na Ordem dos Advogados de Angola "
            "ou nas organizacoes de apoio ao consumidor."
        ),
        country="AO",
        topic="debt",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="A agiotagem prospera em contextos de baixa bancarizacao. Em Angola, e um problema serio nas zonas periurbanas.",
        sources=["Banco Nacional de Angola", "Ordem dos Advogados de Angola"],
    ),
    QAPair(
        id="AO-DEB-003",
        question="Como funciona o credito a habitacao em Angola? Posso financiar uma casa?",
        answer=(
            "O credito habitacao existe em Angola mas e pouco acessivel para a maioria da populacao. "
            "Condicoes tipicas: "
            "- Entrada exigida: 20-40% do valor do imovel; "
            "- Taxas de juro: geralmente acima de 15-20% ao ano; "
            "- Prazo maximo: ate 20 anos (mas poucos bancos oferecem prazos longos); "
            "- Rendimento minimo exigido: elevado (a prestacao nao pode ultrapassar ~40% do rendimento). "
            "Alternativas: autoconstrucao progressiva (construir a casa aos poucos conforme junta dinheiro), "
            "programas habitacionais do governo (Centralidades como Kilamba), "
            "cooperativas de habitacao. "
            "Dica: se optar por autoconstrucao, faca um plano e orcamento por fases - e a forma mais comum de adquirir casa em Angola."
        ),
        country="AO",
        topic="debt",
        difficulty="advanced",
        modalities=["text", "image"],
        image_description=(
            "Diagrama de fases de autoconstrucao: Fase 1 (terreno e fundacao) -> Fase 2 (paredes e cobertura) "
            "-> Fase 3 (acabamentos e instalacoes). Cada fase com custo estimado e tempo. "
            "Comparacao com prestacao de credito habitacao ao lado."
        ),
        cultural_notes="A autoconstrucao progressiva e a forma predominante de aquisicao de habitacao em Angola, refletindo a dificuldade de acesso ao credito formal.",
        sources=["Banco Nacional de Angola", "Ministerio das Obras Publicas de Angola"],
    ),
    QAPair(
        id="AO-DEB-004",
        question="Como evitar dividas quando o Kwanza desvaloriza?",
        answer=(
            "A desvalorizacao do Kwanza encarece produtos importados e pode aumentar dividas indexadas ao dolar. "
            "Estrategias de protecao: "
            "1) Evite dividas em moeda estrangeira (USD) se o seu rendimento e em Kwanzas; "
            "2) Se ja tem divida em USD, tente renegociar para Kwanzas; "
            "3) Reduza a dependencia de produtos importados - prefira producao nacional; "
            "4) Mantenha uma reserva em USD se possivel (limite legal: pode ter conta em USD nos bancos angolanos); "
            "5) Nao faca compras a credito de bens importados cujo preco sobe com o cambio; "
            "6) Acompanhe a taxa de cambio oficial no BNA e nos bancos comerciais. "
            "Em periodos de desvalorizacao, o mais importante e reduzir despesas e evitar novo endividamento."
        ),
        country="AO",
        topic="debt",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Grafico de linha dupla: evolucao da taxa de cambio Kz/USD ao longo de 5 anos (linha azul) "
            "e indice de precos ao consumidor (linha vermelha). "
            "Destacando os momentos de desvalorizacao abrupta."
        ),
        cultural_notes="Angola sofreu desvalorizacoes significativas do Kwanza, especialmente apos a queda do preco do petroleo. O impacto no custo de vida e direto.",
        sources=["Banco Nacional de Angola - Taxas de Cambio"],
    ),
    # --- investment ---
    QAPair(
        id="AO-INV-001",
        question="O que e o Multicaixa Express e como pode ajudar-me financeiramente?",
        answer=(
            "O Multicaixa Express e um servico de pagamentos e transferencias moveis em Angola. "
            "Funcionalidades: transferencias entre pessoas (P2P), pagamentos em comercios, "
            "recebimento de pagamentos, carregamento de telemovel, pagamento de servicos (agua, eletricidade). "
            "Vantagens financeiras: "
            "1) Nao precisa de conta bancaria tradicional - basta BI e telemovel; "
            "2) O dinheiro fica mais seguro que em casa; "
            "3) Facilita poupanca (dinheiro que nao esta fisicamente acessivel e menos tentador); "
            "4) Permite receber pagamentos de clientes de forma rapida e segura; "
            "5) Reduz necessidade de andar com dinheiro fisico (menos risco de roubo). "
            "Para aderir: va a um agente Multicaixa Express com o seu BI."
        ),
        country="AO",
        topic="investment",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Infografico mostrando as funcionalidades do Multicaixa Express: "
            "telemovel no centro, com setas para: transferir dinheiro, pagar contas, "
            "receber pagamentos, carregar telemovel. Com icones simples e passos para aderir."
        ),
        cultural_notes="O Multicaixa Express e o principal instrumento de inclusao financeira digital em Angola, permitindo acesso a servicos financeiros sem conta bancaria.",
        sources=["EMIS - Empresa Interbancaria de Servicos"],
    ),
    QAPair(
        id="AO-INV-002",
        question="Como funciona a BODIVA e posso investir na bolsa angolana?",
        answer=(
            "A BODIVA (Bolsa de Divida e Valores de Angola) e o mercado de capitais angolano. "
            "Atualmente, negoceia sobretudo titulos de divida publica (Obrigacoes do Tesouro e Bilhetes do Tesouro). "
            "O mercado acionista esta em desenvolvimento, com poucas empresas cotadas. "
            "Para investir: "
            "1) Abra conta num banco ou corretora autorizada pela CMC (Comissao do Mercado de Capitais); "
            "2) O investimento minimo em OT e geralmente a partir de 100.000 Kz; "
            "3) O banco executa as ordens de compra e venda. "
            "Vantagens: rentabilidades potencialmente superiores aos depositos; "
            "investir no desenvolvimento do pais. "
            "Limitacoes: mercado ainda pouco liquido, informacao limitada, poucas opcoes alem de divida publica."
        ),
        country="AO",
        topic="investment",
        difficulty="advanced",
        modalities=["text"],
        cultural_notes="A BODIVA foi criada em 2014 e esta em fase de crescimento. O governo angolano tem promovido privatizacoes para dinamizar o mercado de capitais.",
        sources=["BODIVA - bodiva.ao", "CMC - Comissao do Mercado de Capitais de Angola"],
    ),
    QAPair(
        id="AO-INV-003",
        question="Investir em terrenos e boa opcao em Angola?",
        answer=(
            "Terrenos sao um dos investimentos mais tradicionais e populares em Angola. "
            "Vantagens: protecao contra inflacao e desvalorizacao do Kwanza, "
            "possibilidade de autoconstrucao futura, valorizacao com urbanizacao. "
            "Cuidados essenciais: "
            "1) Verifique a titularidade legal do terreno no IPGC (Instituto de Planeamento e Gestao Urbanistica); "
            "2) Confirme que nao e terreno do Estado ou de reserva; "
            "3) Obtenha o DUAT (Direito de Uso e Aproveitamento do Terreno) devidamente registado; "
            "4) Desconfie de precos muito abaixo do mercado; "
            "5) Nao pague sem documentacao legal completa; "
            "6) Use um advogado para verificar a documentacao. "
            "Muitas fraudes imobiliarias acontecem com terrenos sem documentacao regular."
        ),
        country="AO",
        topic="investment",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="A propriedade fundiaria em Angola e complexa - a terra pertence ao Estado, e os particulares obtem direitos de uso (DUAT). Fraudes sao comuns.",
        sources=["Lei de Terras de Angola (Lei 9/04)", "IPGC"],
    ),
    QAPair(
        id="AO-INV-004",
        question="O que significa bancarizacao e por que e importante para mim?",
        answer=(
            "Bancarizacao significa ter acesso e usar servicos financeiros formais: "
            "conta bancaria, cartao, transferencias, poupanca, credito. "
            "Por que e importante: "
            "1) Seguranca: dinheiro no banco esta protegido contra roubo e perda; "
            "2) Acesso a credito: so com historico bancario consegue emprestimo formal; "
            "3) Poupanca: conta bancaria facilita guardar dinheiro de forma disciplinada; "
            "4) Pagamentos: pode receber salario, pagar contas, fazer compras sem dinheiro fisico; "
            "5) Formalidade: necessario para emitir facturas, pagar impostos, participar na economia formal; "
            "6) Protecao: em caso de disputa comercial, ha registo das transacoes. "
            "Angola tem meta de aumentar a bancarizacao para 50% da populacao. "
            "O Multicaixa Express e um caminho simples para comecar."
        ),
        country="AO",
        topic="investment",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="Angola tem uma das taxas de bancarizacao mais baixas da regiao SADC, com grande disparidade entre Luanda e o resto do pais.",
        sources=["Banco Nacional de Angola - Estrategia de Inclusao Financeira"],
    ),
    # --- credit ---
    QAPair(
        id="AO-CRE-001",
        question="Como funciona o acesso a credito formal em Angola?",
        answer=(
            "O acesso a credito em Angola e ainda limitado para a maioria da populacao. "
            "Requisitos tipicos dos bancos: "
            "1) Conta bancaria ativa com historico de movimentacao; "
            "2) Comprovativo de rendimento (dificil para trabalhadores informais); "
            "3) BI e NIF; "
            "4) Garantias (imovel, veiculo, ou avalista); "
            "5) Bom historico de credito (sem incumprimentos registados no BNA). "
            "O Banco Nacional de Angola gere a Central de Informacao de Risco de Credito (CIRC) "
            "onde ficam registados os seus creditos e eventuais incumprimentos. "
            "Para melhorar o acesso: mantenha conta bancaria ativa, formalize rendimentos quando possivel, "
            "comece com creditos pequenos para construir historico."
        ),
        country="AO",
        topic="credit",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="A CIRC do BNA e o equivalente angolano ao Serasa brasileiro ou a CRC portuguesa, mas com menor cobertura da populacao.",
        sources=["Banco Nacional de Angola - CIRC"],
    ),
    QAPair(
        id="AO-CRE-002",
        question="O Banco Nacional de Angola protege o consumidor bancario?",
        answer=(
            "Sim, o BNA tem funcoes de protecao do consumidor financeiro. "
            "Pode reclamar ao BNA se: "
            "1) O banco cobrou taxas ou comissoes indevidas; "
            "2) O banco recusou abertura de conta basica sem justificacao; "
            "3) Houve falha nos servicos (ATM, transferencias, Multicaixa); "
            "4) O contrato de credito tem clausulas abusivas. "
            "Como reclamar: presencialmente nas agencias do BNA, por carta, ou pelo portal online. "
            "Prazos: o BNA tem 30 dias uteis para responder. "
            "Dica: antes de reclamar ao BNA, tente resolver diretamente com o banco "
            "(livro de reclamacoes, provedor do cliente). "
            "Guarde sempre comprovativos de todas as operacoes bancarias."
        ),
        country="AO",
        topic="credit",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="A regulacao de protecao ao consumidor bancario em Angola esta em evolucao, com o BNA a fortalecer progressivamente o seu papel.",
        sources=["Banco Nacional de Angola - Departamento de Protecao ao Consumidor"],
    ),
    # --- cultural ---
    QAPair(
        id="AO-CUL-001",
        question="O que e o alembamento e qual o impacto financeiro?",
        answer=(
            "O alembamento e a cerimonia tradicional angolana que formaliza a uniao entre duas familias, "
            "equivalente cultural do dote ou casamento tradicional. "
            "O noivo e sua familia devem entregar bens e dinheiro a familia da noiva. "
            "Os pedidos podem incluir: bebidas, tecidos, dinheiro (valores variam muito - de 100.000 Kz a milhoes), "
            "gado (nas zonas rurais), eletrodomesticos. "
            "Impacto financeiro: "
            "1) Pode representar meses ou anos de poupanca; "
            "2) Muitos jovens endividam-se para o alembamento; "
            "3) Pressao social para fazer alembamento 'a altura'. "
            "Dica financeira: negociem um alembamento dentro das possibilidades reais, "
            "poupar especificamente para isso com antecedencia, "
            "e nunca se endividem com agiotas para o alembamento."
        ),
        country="AO",
        topic="cultural",
        difficulty="basic",
        modalities=["text", "image"],
        image_description=(
            "Infografico mostrando itens tipicos de um alembamento: lista de itens pedidos "
            "com custo estimado de cada um. Total geral e sugestao de plano de poupanca mensal."
        ),
        cultural_notes="O alembamento e uma das mais importantes tradicoes culturais angolanas, com profundas implicacoes financeiras para os jovens casais.",
        sources=["Estudos antropologicos sobre casamento em Angola"],
    ),
    QAPair(
        id="AO-CUL-002",
        question="Qual o papel da economia informal em Angola e como me posicionar?",
        answer=(
            "A economia informal representa mais de 50% do PIB angolano e emprega a maioria da forca de trabalho. "
            "Actividades comuns: venda ambulante (zungueiras), oficinas mecanicas, cabeleireiras, "
            "transporte (candongueiros e kupapatas), agricultura de subsistencia, comercio no mercado. "
            "Estrategias para prosperar: "
            "1) Registe o seu negocio se possivel (AGT tem regimes simplificados); "
            "2) Mantenha registos financeiros mesmo informais; "
            "3) Diversifique fontes de rendimento; "
            "4) Use kixikila e poupanca formal em paralelo; "
            "5) Invista na qualificacao (cursos do INAPEM, Sebrae Angola); "
            "6) Considere cooperativas para ter mais forca de negociacao. "
            "A formalizacao gradual abre portas para credito, contratos e protecao legal."
        ),
        country="AO",
        topic="cultural",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="A economia informal e a espinha dorsal da sobrevivencia de milhoes de angolanos. A formalizacao e um processo gradual que deve respeitar esta realidade.",
        sources=["INE Angola", "OIT - Organizacao Internacional do Trabalho"],
    ),
    QAPair(
        id="AO-CUL-003",
        question="Como a desvalorizacao do Kwanza afeta o meu dia a dia?",
        answer=(
            "A desvalorizacao do Kwanza tem impacto direto na vida quotidiana: "
            "1) Alimentos: Angola importa mais de 50% dos alimentos - precos sobem quando o Kz desvaloriza; "
            "2) Combustiveis: embora Angola produza petroleo, derivados sao parcialmente importados; "
            "3) Medicamentos: maioria importada, ficam mais caros; "
            "4) Tecnologia: telemoveis, computadores, electrodomesticos importados encarecem; "
            "5) Educacao: escolas internacionais e material didatico importado sobem de preco. "
            "Como proteger-se: prefira produtos nacionais quando possivel, "
            "mantenha alguma poupanca em USD (conta bancaria em moeda estrangeira), "
            "acompanhe a taxa de cambio e antecipe compras essenciais antes de desvalorizacoes anunciadas, "
            "e invista em activos que preservem valor (terrenos, negocio proprio)."
        ),
        country="AO",
        topic="cultural",
        difficulty="intermediate",
        modalities=["text", "image"],
        image_description=(
            "Infografico tipo cesta basica mostrando itens do dia a dia "
            "e como o preco em Kwanzas mudou ao longo de 3 anos. "
            "Destaque para a proporcao de itens importados vs nacionais."
        ),
        cultural_notes="Angola e altamente dependente de importacoes apesar de ser rica em recursos naturais. A diversificacao economica e um desafio central do pais.",
        sources=["Banco Nacional de Angola", "INE Angola"],
    ),
    QAPair(
        id="AO-CUL-004",
        question="Que oportunidades de negocio existem para jovens em Angola?",
        answer=(
            "Angola tem oportunidades crescentes para jovens empreendedores: "
            "1) Agricultura: o governo apoia producao nacional (PRODESI) - frutas, horticolas, avicultura; "
            "2) Tecnologia: startups em fintech, e-commerce, delivery estao em crescimento; "
            "3) Servicos: reparacao de telemoveis, cabeleireiro, costura, formacao profissional; "
            "4) Comercio: revenda de produtos no Multicaixa Express e redes sociais; "
            "5) Energia: solucoes de energia solar para zonas sem eletricidade. "
            "Apoios disponiveis: INAPEM (formacao), BDA (credito), PRODESI (producao nacional), "
            "incubadoras como o Polo Tecnologico de Luanda. "
            "Chave: comece pequeno, valide o negocio, reinvista os lucros, e formalize-se gradualmente."
        ),
        country="AO",
        topic="cultural",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="Angola tem uma populacao muito jovem (mais de 60% com menos de 25 anos), criando pressao e oportunidade empreendedora.",
        sources=["INAPEM", "Angola Startup Summit"],
    ),
    # --- debt (extra) ---
    QAPair(
        id="AO-DEB-005",
        question="O que devo saber antes de assinar um contrato de credito no banco?",
        answer=(
            "Antes de assinar qualquer contrato de credito: "
            "1) Leia TUDO, especialmente a taxa de juro (TAN e TAEG), comissoes e penalizacoes; "
            "2) Verifique a prestacao mensal e confirme que cabe no seu orcamento (maximo 30-40% do rendimento); "
            "3) Pergunte sobre: seguros obrigatorios, comissao de abertura, comissao de amortizacao antecipada; "
            "4) Peca simulacao por escrito antes de assinar; "
            "5) Compare com pelo menos 2 outros bancos; "
            "6) Nao assine sob pressao - peca tempo para analisar em casa; "
            "7) Se nao entender alguma clausula, peca explicacao ou leve a um advogado. "
            "Guarde SEMPRE uma copia do contrato assinado."
        ),
        country="AO",
        topic="debt",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="A literacia financeira sobre contratos bancarios e baixa em Angola, tornando muitos mutuarios vulneraveis a clausulas desfavoraveis.",
        sources=["Banco Nacional de Angola - Protecao ao Consumidor"],
    ),
    # --- investment (extra) ---
    QAPair(
        id="AO-INV-005",
        question="O que sao cooperativas de poupanca e credito em Angola?",
        answer=(
            "Cooperativas de poupanca e credito sao organizacoes comunitarias onde membros poupam juntos "
            "e podem aceder a emprestimos com condicoes mais favoraveis que os bancos comerciais. "
            "Funcionamento: os membros fazem contribuicoes regulares; os fundos sao emprestados a membros necessitados; "
            "os juros cobrados sao redistribuidos como dividendos aos membros. "
            "Vantagens: taxas de juro geralmente mais baixas, decisoes locais, apoio comunitario. "
            "Cuidados: verifique se a cooperativa e registada e supervisionada pelo BNA; "
            "desconfie de cooperativas que prometem retornos muito altos; "
            "exija transparencia nas contas e relatorios financeiros. "
            "E uma alternativa entre a kixikila informal e o banco comercial."
        ),
        country="AO",
        topic="investment",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="As cooperativas financeiras em Angola estao em crescimento como alternativa de inclusao financeira, especialmente fora de Luanda.",
        sources=["Banco Nacional de Angola", "Lei das Cooperativas de Angola"],
    ),
    # --- cultural (extra) ---
    QAPair(
        id="AO-CUL-005",
        question="Como proteger o meu dinheiro em tempos de crise economica em Angola?",
        answer=(
            "Em periodos de crise economica em Angola (queda do preco do petroleo, inflacao alta): "
            "1) Reduza despesas nao essenciais imediatamente; "
            "2) Evite dividas novas, especialmente em moeda estrangeira; "
            "3) Diversifique fontes de rendimento (se possivel, mais de uma actividade); "
            "4) Mantenha alguma poupanca em USD como reserva de valor; "
            "5) Invista em bens tangiveis (terrenos, materiais de construcao) antes que os precos subam; "
            "6) Stocke alimentos nao pereciveis quando encontrar bons precos; "
            "7) Fortaleca a rede de apoio comunitario (familia, kixikila); "
            "8) Evite o panico - tome decisoes racionais, nao emocionais. "
            "As crises sao ciclicas - quem se prepara atravessa melhor."
        ),
        country="AO",
        topic="cultural",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="Angola sofreu crises economicas severas ligadas a flutuacao do preco do petroleo, que representa mais de 90% das exportacoes do pais.",
        sources=["Banco Nacional de Angola", "FMI - Relatorio sobre Angola"],
    ),
    QAPair(
        id="AO-CUL-006",
        question="O que e o PRODESI e como pode ajudar o meu negocio?",
        answer=(
            "O PRODESI (Programa de Apoio a Producao, Diversificacao das Exportacoes e Substituicao das Importacoes) "
            "e um programa do governo angolano para fomentar a producao nacional. "
            "Apoios disponiveis: credito bonificado para producao agricola e industrial, "
            "facilitacao de acesso a terrenos, apoio tecnico e formacao, protecao de mercado para produtos nacionais. "
            "Sectores prioritarios: agricultura, pecuaria, pescas, industria transformadora, turismo. "
            "Como aceder: atraves dos bancos comerciais parceiros (BDA, BPC, BFA) "
            "ou directamente nos balcoes de atendimento PRODESI em cada provincia. "
            "Requisitos: ter empresa registada (ou dispor-se a registar), "
            "projecto viavel de producao, contribuicao propria minima. "
            "O PRODESI e uma das melhores oportunidades para formalizar e expandir negocios em Angola."
        ),
        country="AO",
        topic="cultural",
        difficulty="intermediate",
        modalities=["text"],
        cultural_notes="O PRODESI faz parte da estrategia de diversificacao economica de Angola para reduzir a dependencia do petroleo.",
        sources=["PRODESI - Ministerio da Economia e Planeamento de Angola"],
    ),
    QAPair(
        id="AO-SAV-006",
        question="Como poupar para a educacao dos filhos no estrangeiro?",
        answer=(
            "Muitas familias angolanas sonham em enviar os filhos para estudar no estrangeiro "
            "(Portugal, Brasil, Africa do Sul, etc.). O planeamento financeiro e essencial: "
            "1) Comece a poupar 10+ anos antes - o valor necessario e elevado; "
            "2) Estime custos: propinas (2.000-15.000 USD/ano conforme pais), alojamento, alimentacao, seguro; "
            "3) Poupe em USD ou EUR para proteger contra desvalorizacao do Kwanza; "
            "4) Use depositos a prazo em moeda estrangeira nos bancos angolanos; "
            "5) Pesquise bolsas: INAGBE oferece bolsas governamentais, universidades portuguesas e brasileiras "
            "tem programas para paises lusofonos (CPLP); "
            "6) Considere alternativas: universidades angolanas de qualidade estao a crescer "
            "(Universidade Agostinho Neto, Universidade Catolica). "
            "Planeie com realismo e explore todas as opcoes de bolsas antes de assumir custos totais."
        ),
        country="AO",
        topic="savings",
        difficulty="advanced",
        modalities=["text", "image"],
        image_description=(
            "Tabela comparativa: custo anual estimado de estudar em Portugal, Brasil, Africa do Sul e Angola. "
            "Com conversao em Kwanzas e plano de poupanca mensal necessario para cada cenario."
        ),
        cultural_notes="A educacao no estrangeiro e altamente valorizada em Angola, especialmente em Portugal e Brasil por razoes linguisticas.",
        sources=["INAGBE", "Embaixada de Portugal em Luanda"],
    ),
    QAPair(
        id="AO-CRE-003",
        question="O que e o NIF angolano e por que e tao importante?",
        answer=(
            "O NIF (Numero de Identificacao Fiscal) e o seu numero unico perante a AGT "
            "(Administracao Geral Tributaria). E indispensavel para: "
            "abrir conta bancaria, registar empresa, celebrar contratos, importar e exportar, "
            "comprar imoveis, obter credito, emitir facturas. "
            "Para obter: dirija-se a uma reparticao fiscal da AGT com o seu BI; "
            "em algumas provincias, pode obtê-lo nos balcoes unicos de atendimento. "
            "O processo demora tipicamente 1-5 dias uteis. "
            "Dica: obtenha o NIF mesmo que trabalhe na economia informal - "
            "tera acesso a mais servicos e podera formalizar o seu negocio gradualmente. "
            "O NIF e gratuito e valido permanentemente."
        ),
        country="AO",
        topic="credit",
        difficulty="basic",
        modalities=["text"],
        cultural_notes="A baixa penetracao do NIF na populacao angolana e um dos principais obstaculos a bancarizacao e formalizacao economica.",
        sources=["AGT - Administracao Geral Tributaria de Angola"],
    ),
]


def _check_unique_ids(pairs: list[QAPair], country_code: str) -> None:
    """Valida que nao existem IDs duplicados dentro da lista do pais."""
    seen: dict[str, int] = {}
    for qa in pairs:
        seen[qa.id] = seen.get(qa.id, 0) + 1
    duplicates = {qa_id: count for qa_id, count in seen.items() if count > 1}
    if duplicates:
        raise ValueError(f"IDs duplicados em {country_code}: {duplicates}")


def _distribution(pairs: list[QAPair], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for qa in pairs:
        key = getattr(qa, attr)
        if isinstance(key, list):
            key = "+".join(sorted(key))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def generate_dataset() -> None:
    """Gera o dataset completo (seed + expansao) e salva em JSONL por pais."""
    base_dir = Path(__file__).resolve().parent

    country_data: dict[str, list[QAPair]] = {
        "BR": BR_PAIRS + BR_EXPANSION_ALL,
        "PT": PT_PAIRS + PT_EXPANSION_ALL,
        "AO": AO_PAIRS + AO_EXPANSION_ALL,
    }

    total = 0
    for country_code, pairs in country_data.items():
        _check_unique_ids(pairs, country_code)
        output_dir = base_dir / country_code
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{country_code.lower()}_seed.jsonl"
        save_jsonl(pairs, output_path)
        total += len(pairs)
        seed_n = {"BR": len(BR_PAIRS), "PT": len(PT_PAIRS), "AO": len(AO_PAIRS)}[country_code]
        exp_n = len(pairs) - seed_n
        print(f"[{country_code}] {len(pairs)} QA pairs ({seed_n} seed + {exp_n} expansao) -> {output_path}")

    print(f"\nTotal: {total} QA pairs gerados.")
    print("\nDistribuicao por topico:")
    for country_code, pairs in country_data.items():
        print(f"  {country_code}: {_distribution(pairs, 'topic')}")
    print("\nDistribuicao por dificuldade:")
    for country_code, pairs in country_data.items():
        print(f"  {country_code}: {_distribution(pairs, 'difficulty')}")
    print("\nDistribuicao por modalidades:")
    for country_code, pairs in country_data.items():
        print(f"  {country_code}: {_distribution(pairs, 'modalities')}")


if __name__ == "__main__":
    generate_dataset()

"""Expansao 2A do dataset FinEduBench-Luso — Angola (budget + savings)."""

AO_EXPANSION_2A: list[dict] = [
    # =========================================================================
    # BUDGET — 39 pairs (AO-BUD-018 .. AO-BUD-056)
    # =========================================================================
    {
        "id": "AO-BUD-018",
        "question": "Como é que posso fazer um orçamento familiar se ganho o salário mínimo angolano?",
        "answer": (
            "Com o salário mínimo em Angola, a primeira coisa é anotar todas as despesas fixas: renda da casa, água, "
            "luz e transporte. Depois, separe pelo menos 10% para uma reserva, mesmo que pareça pouco. "
            "Priorize a alimentação comprando no mercado informal, onde os preços costumam ser mais baixos. "
            "Evite compras por impulso e tente negociar sempre que possível nas praças. "
            "Se sobrar algum, guarde numa kixikila com kambas de confiança. "
            "Registar tudo num caderno ajuda a perceber onde o dinheiro está a ir."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Gráfico circular mostrando a distribuição típica do salário mínimo angolano: alimentação 40%, renda 25%, transporte 15%, outros 10%, poupança 10%.",
        "audio_transcript": None,
        "cultural_notes": "O salário mínimo em Angola varia por sector. Muitas famílias complementam com rendimentos informais.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-019",
        "question": "O que devo ter em conta ao controlar gastos nas compras do mercado informal?",
        "answer": (
            "Nas praças e mercados informais de Angola, os preços variam bué de um vendedor para outro. "
            "Antes de comprar, dê uma volta e compare os preços em pelo menos três bancas. "
            "Leve sempre dinheiro trocado porque as zungueiras raramente têm troco. "
            "Faça uma lista do que precisa antes de sair de casa para evitar compras desnecessárias. "
            "Compre produtos da época, que são mais baratos e frescos. "
            "Anote quanto gastou em cada ida ao mercado para ter controlo mensal dos gastos com alimentação."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Tabela comparativa de preços de produtos básicos (tomate, cebola, feijão, fuba) em diferentes mercados de Luanda.",
        "audio_transcript": None,
        "cultural_notes": "As zungueiras são vendedoras ambulantes, essenciais na cadeia de distribuição alimentar angolana.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-020",
        "question": "Qual é a diferença no custo de vida entre Luanda e as províncias do interior?",
        "answer": (
            "Luanda é, de longe, a cidade mais cara de Angola. A renda de uma casa em Luanda pode custar "
            "entre Kz 80.000 e Kz 500.000, enquanto nas províncias como Huambo ou Bié pode ser Kz 20.000 a Kz 80.000. "
            "A alimentação também é mais cara em Luanda porque muitos produtos vêm de fora. "
            "O transporte em Luanda consome mais orçamento por causa das distâncias e do trânsito. "
            "Nas províncias, muitas famílias produzem parte da sua alimentação, o que reduz custos. "
            "Ao fazer o orçamento, é importante considerar a realidade da sua província e não copiar padrões de Luanda."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Gráfico de barras comparando o custo de vida mensal em Luanda, Huambo, Benguela e Lubango para uma família de 4 pessoas.",
        "audio_transcript": None,
        "cultural_notes": "Luanda concentra grande parte da actividade económica e populacional, inflacionando os preços locais.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-BUD-021",
        "question": "Como fazer um orçamento quando a minha renda vem do comércio informal?",
        "answer": (
            "Quando se vive do comércio informal, a renda não é fixa, por isso o orçamento tem de ser flexível. "
            "Comece por calcular a média do que ganha por semana durante dois ou três meses. "
            "Separe primeiro as despesas que não podem falhar: renda, comida e transporte. "
            "Guarde pelo menos Kz 5.000 a Kz 10.000 por semana para imprevistos, mesmo nos meses bons. "
            "Não misture o dinheiro do negócio com o dinheiro da casa — use bolsos ou envelopes separados. "
            "Nos meses em que ganhar mais, reforce a poupança em vez de aumentar os gastos."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Diagrama de fluxo mostrando como separar rendimentos variáveis do comércio informal em categorias: despesas fixas, reposição de stock, poupança e gastos pessoais.",
        "audio_transcript": None,
        "cultural_notes": "O sector informal emprega a maioria da população activa angolana, especialmente mulheres vendedoras (zungueiras e quitandeiras).",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-BUD-022",
        "question": "Quanto devo reservar no meu orçamento para despesas com gerador e água?",
        "answer": (
            "Em muitas zonas de Angola, a electricidade e a água não são constantes, por isso o gerador é essencial. "
            "O gasóleo para um gerador pequeno pode custar entre Kz 15.000 e Kz 40.000 por mês, dependendo do uso. "
            "A água comprada a camiões-cisterna pode variar de Kz 5.000 a Kz 20.000 por mês. "
            "Reserve entre 15% e 25% do orçamento mensal para estas despesas de infraestrutura. "
            "Uma alternativa é investir num painel solar pequeno, que a longo prazo sai mais barato que o gerador. "
            "Tente acumular água em bidões nos dias em que há fornecimento para reduzir a compra a terceiros."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Gráfico mostrando o custo mensal estimado de gerador (gasóleo + manutenção) e água em diferentes bairros de Luanda.",
        "audio_transcript": None,
        "cultural_notes": "As falhas no fornecimento de água e electricidade são frequentes, obrigando as famílias a ter soluções alternativas como geradores e cisternas.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-023",
        "question": "Como posso reduzir os gastos com transporte em Luanda usando candongueiros e kupapatas?",
        "answer": (
            "O transporte em Luanda consome bué do orçamento familiar. Os candongueiros são a opção mais barata, "
            "com tarifas a partir de Kz 200, mas é importante conhecer as rotas para evitar transbordos desnecessários. "
            "Os kupapatas (moto-táxis) são mais rápidos no trânsito, mas custam mais e têm riscos de segurança. "
            "Planear as deslocações por dia e agrupar tarefas na mesma zona poupa dinheiro e tempo. "
            "Se possível, compre a quantidade semanal de transporte e veja se vale mais do que pagar dia a dia. "
            "Considere caminhar em distâncias curtas — além de poupar, faz bem à saúde."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "Candongueiros são táxis colectivos (Toyota Hiace azul e branco). Kupapatas são moto-táxis. Ambos são transportes informais essenciais em Luanda.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-024",
        "question": "Quanto devo orçamentar para uma festa de óbito em Angola?",
        "answer": (
            "A festa de óbito é uma tradição cultural muito importante em Angola e pode ter custos significativos. "
            "Um óbito simples, com comida e bebida para a comunidade, pode custar entre Kz 200.000 e Kz 500.000. "
            "As famílias costumam receber contribuições dos kambas, vizinhos e colegas de trabalho. "
            "É aconselhável ter uma reserva específica para estas situações, guardando Kz 10.000 a Kz 20.000 por mês. "
            "Participe em grupos de solidariedade que fazem colectas quando alguém perde um familiar. "
            "Não se endivide para fazer um óbito — o importante é honrar o falecido dentro das possibilidades."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Tabela com estimativa de custos de uma festa de óbito em Luanda: alimentação, bebidas, aluguer de tendas, transporte do corpo, e contribuições comunitárias esperadas.",
        "audio_transcript": None,
        "cultural_notes": "A festa de óbito é um evento social e cultural obrigatório em Angola, que pode durar vários dias e representa uma despesa significativa para as famílias.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-BUD-025",
        "question": "Como posso planificar as despesas de educação dos meus filhos em Angola?",
        "answer": (
            "A educação em Angola envolve várias despesas: propinas, material escolar, uniforme, lanche e transporte. "
            "Numa escola pública, as propinas são baixas, mas o material e o uniforme podem custar Kz 30.000 a Kz 80.000 por ano. "
            "Nas escolas privadas, as propinas podem variar de Kz 15.000 a Kz 100.000 por mês. "
            "Muitos pais recorrem a explicadores particulares, que cobram entre Kz 10.000 e Kz 30.000 por mês. "
            "Reserve uma parcela fixa do orçamento para educação e comece a juntar para o material escolar desde Janeiro. "
            "Compre livros e material em grupo com outros pais para conseguir descontos."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Gráfico de barras empilhadas mostrando os custos anuais de educação por nível (primário, secundário, universitário) incluindo propinas, material e transporte.",
        "audio_transcript": None,
        "cultural_notes": "Os explicadores são tutores privados muito comuns em Angola, usados para complementar o ensino nas escolas públicas, que muitas vezes têm turmas superlotadas.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-BUD-026",
        "question": "Como fazer o orçamento mensal para saúde, considerando hospitais públicos e clínicas privadas?",
        "answer": (
            "A saúde em Angola exige planeamento financeiro porque os hospitais públicos são gratuitos mas têm muitas limitações. "
            "Em muitos casos, mesmo nos hospitais públicos, é preciso comprar medicamentos fora, o que pode custar Kz 5.000 a Kz 50.000. "
            "Uma consulta numa clínica privada custa entre Kz 10.000 e Kz 50.000, sem contar exames. "
            "Reserve entre 5% e 10% do orçamento mensal para despesas de saúde. "
            "Considere aderir a um seguro de saúde se o seu empregador não oferecer, mesmo que seja o mais básico. "
            "Ter uma reserva de emergência médica de pelo menos Kz 100.000 pode evitar endividamento em caso de doença."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "intermediate",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "O sistema de saúde público angolano é gratuito mas enfrenta escassez de materiais e medicamentos, levando muitas famílias a recorrer ao sector privado.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-027",
        "question": "O que é que posso fazer no meu orçamento quando o Kwanza desvaloriza?",
        "answer": (
            "A desvalorização do Kwanza faz com que os preços dos produtos importados subam rapidamente. "
            "Quando há sinais de desvalorização, antecipe compras essenciais como óleo, arroz e sabão. "
            "Ajuste o orçamento mensalmente, porque os preços mudam de semana para semana. "
            "Reduza a dependência de produtos importados e prefira alimentos locais como fuba, mandioca e feijão. "
            "Se possível, mantenha uma pequena parte das poupanças em dólares para proteger o poder de compra. "
            "Renegocie contratos e compromissos que estejam indexados ao dólar para evitar surpresas."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Gráfico de linha mostrando a evolução da taxa de câmbio Kz/USD nos últimos 5 anos e o impacto no cabaz básico de alimentação.",
        "audio_transcript": None,
        "cultural_notes": "A economia angolana é muito dependente do petróleo e do dólar, tornando o Kwanza vulnerável a desvalorizações frequentes.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-028",
        "question": "Quanto custa um alambamento em Angola e como posso orçamentar?",
        "answer": (
            "O alambamento é o casamento tradicional angolano e os custos variam muito conforme a família da noiva. "
            "A lista de pedidos pode incluir dinheiro (Kz 200.000 a Kz 2.000.000), bebidas, tecidos, animais e outros bens. "
            "Comece a planificar com pelo menos um ano de antecedência e poupe mensalmente para este objectivo. "
            "Negocie com a família da noiva de forma respeitosa — muitas famílias aceitam ajustar os pedidos. "
            "Não se endivide para pagar o alambamento; é melhor adiar do que começar a vida a dois com dívidas. "
            "Faça uma lista detalhada de todos os itens pedidos e pesquise preços antes de comprar."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Lista típica de alambamento com os itens mais comuns e respectivos preços estimados em Kwanzas.",
        "audio_transcript": None,
        "cultural_notes": "O alambamento é uma cerimónia de noivado tradicional onde a família do noivo entrega bens e dinheiro à família da noiva. É culturalmente obrigatório em muitas etnias angolanas.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-BUD-029",
        "question": "Qual é o custo do cabaz básico de alimentação em Angola e como posso poupar?",
        "answer": (
            "O cabaz básico para uma família de 4 pessoas em Luanda pode custar entre Kz 80.000 e Kz 150.000 por mês. "
            "Os produtos essenciais incluem fuba de milho, arroz, feijão, óleo alimentar, açúcar, sal, tomate e cebola. "
            "Comprar em quantidade no início do mês nas praças maiores como o Mercado do Hoji Ya Henda sai mais barato. "
            "Substitua produtos importados por nacionais: por exemplo, fuba de milho local em vez de farinha de trigo importada. "
            "Comprar directamente a produtores ou grossistas pode reduzir os custos em 20% a 30%. "
            "Planeie as refeições da semana para evitar desperdício e compras de última hora mais caras."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Tabela com os preços médios dos 10 produtos do cabaz básico angolano em Kwanzas, comparando praça informal e supermercado.",
        "audio_transcript": None,
        "cultural_notes": "A fuba de milho é a base da alimentação angolana, usada para fazer funge, o prato nacional.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-BUD-030",
        "question": "Como posso comparar os custos de morar num musseque e num condomínio em Luanda?",
        "answer": (
            "A renda num musseque pode variar de Kz 15.000 a Kz 50.000 por mês, enquanto num condomínio pode ir de Kz 150.000 a Kz 800.000. "
            "No musseque, os custos de água e electricidade são menores mas a qualidade é inferior e pode precisar de gerador. "
            "Os condomínios geralmente incluem segurança, água e luz mais estáveis, mas têm taxa de condomínio. "
            "No musseque, gasta-se mais com transporte porque muitos ficam longe dos centros de emprego. "
            "Faça uma conta total: renda + água + luz + gerador + transporte para comparar as duas opções de forma justa. "
            "Considere a segurança, o acesso a escolas e hospitais como factores além do custo puro."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Tabela comparativa de custos mensais totais de habitação em musseque vs condomínio em Luanda, incluindo renda, utilidades, transporte e segurança.",
        "audio_transcript": None,
        "cultural_notes": "Musseques são bairros de construção informal que albergam a maioria da população de Luanda. Condomínios fechados surgiram no boom petrolífero.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-031",
        "question": "Sou empreendedor de rua. Como posso fazer um orçamento para o meu negócio?",
        "answer": (
            "Como empreendedor de rua, separe sempre o dinheiro do negócio do dinheiro pessoal — use envelopes ou carteiras diferentes. "
            "Anote diariamente quanto comprou de mercadoria e quanto vendeu para saber o lucro real. "
            "Reserve pelo menos 30% das vendas para repor o stock e 10% para poupança do negócio. "
            "Inclua no orçamento os custos escondidos: transporte da mercadoria, taxa do lugar na praça, e perdas por produtos estragados. "
            "Nos meses bons, não aumente os gastos pessoais — reforce o stock e a poupança. "
            "Mantenha um caderno simples com entradas e saídas; mesmo sem ser contabilista, isso faz bué diferença."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Modelo simples de caderno de registos para empreendedor de rua com colunas: data, mercadoria comprada, valor gasto, vendas do dia, lucro.",
        "audio_transcript": None,
        "cultural_notes": "O comércio de rua é a principal fonte de renda para milhões de angolanos, especialmente mulheres.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-BUD-032",
        "question": "Como gerir as finanças quando recebo parte em dólares e parte em Kwanzas?",
        "answer": (
            "Quando se recebe em duas moedas, é fundamental não misturar os orçamentos. "
            "Pague as despesas em Kwanzas com os Kwanzas e guarde os dólares como reserva de valor. "
            "Troque dólares apenas quando necessário e sempre na taxa mais favorável — compare bancos e casas de câmbio autorizadas. "
            "Evite o mercado paralelo (kinguilas) porque, apesar da taxa ser melhor, há risco de notas falsas e problemas legais. "
            "Mantenha um registo separado para cada moeda e converta tudo para Kwanzas no final do mês para ter uma visão real. "
            "Se possível, mantenha uma conta bancária em dólares para evitar a desvalorização do Kwanza sobre os seus rendimentos."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "advanced",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "Angola tem uma economia parcialmente dolarizada. Kinguilas são cambistas informais de rua que operam fora do sistema bancário oficial.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-033",
        "question": "Como posso ensinar orçamento financeiro aos meus filhos em Angola?",
        "answer": (
            "Comece por dar ao seu filho uma pequena mesada semanal, mesmo que seja Kz 500 a Kz 1.000. "
            "Peça que ele divida o dinheiro em três partes: gastar, poupar e partilhar com alguém. "
            "Leve-o consigo ao mercado para que veja os preços e aprenda a comparar e negociar. "
            "Ensine-o a anotar os gastos num caderno — este hábito vai acompanhá-lo para a vida. "
            "Explique que o dinheiro é limitado e que escolher uma coisa significa abrir mão de outra. "
            "Dê o exemplo: se os filhos virem os pais a planificar, vão fazer o mesmo naturalmente."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "A educação financeira infantil é pouco desenvolvida em Angola, sendo a família o principal canal de transmissão de hábitos financeiros.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-BUD-034",
        "question": "Quais são os principais erros de orçamento que os angolanos cometem?",
        "answer": (
            "Um dos maiores erros é gastar bué logo que recebe o salário sem planificar o mês inteiro. "
            "Outro erro comum é não incluir despesas irregulares como óbitos, festas e reparações no orçamento. "
            "Muitas pessoas não separam o dinheiro do negócio do dinheiro pessoal, o que gera confusão. "
            "Comprar fiado sem controlar o quanto deve é uma armadilha que acumula dívidas invisíveis. "
            "Não ter reserva de emergência faz com que qualquer imprevisto se torne uma crise financeira. "
            "Gastar muito com aparências — roupas de marca, festas — em vez de investir no essencial também prejudica bué."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Infográfico listando os 6 erros financeiros mais comuns dos angolanos com ícones ilustrativos para cada um.",
        "audio_transcript": None,
        "cultural_notes": "Em Angola, a pressão social para demonstrar prosperidade é forte, levando a gastos excessivos com aparências.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-035",
        "question": "Como fazer um orçamento semanal se recebo pagamento diário no comércio?",
        "answer": (
            "Quem recebe diariamente precisa pensar em ciclos semanais em vez de mensais. "
            "Ao final de cada dia, separe o dinheiro em três montes: reposição de mercadoria, despesas da casa e poupança. "
            "Defina um valor fixo diário para gastar consigo e com a família — o que sobrar, guarde. "
            "Ao domingo, faça a conta da semana e veja se está no caminho certo. "
            "Use o Multicaixa Express para transferir a poupança diária para uma conta, tirando a tentação de gastar. "
            "Mantenha um caderninho no bolso onde anota tudo que entra e sai — basta 5 minutos por dia."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Modelo de tabela semanal com linhas para cada dia e colunas: ganho do dia, reposição de stock, despesas da casa, poupança.",
        "audio_transcript": None,
        "cultural_notes": "Grande parte dos trabalhadores informais angolanos recebe rendimentos diários variáveis.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-036",
        "question": "Quanto custa manter um carro em Luanda e vale a pena no orçamento?",
        "answer": (
            "Manter um carro em Luanda envolve combustível (Kz 30.000 a Kz 80.000/mês), seguro obrigatório, manutenção e estacionamento. "
            "O combustível em Angola é subsidiado, mas os custos de manutenção e peças são altos porque são quase todos importados. "
            "O seguro automóvel obrigatório custa a partir de Kz 20.000 por ano, mas o seguro completo é bem mais caro. "
            "Compare o custo total do carro com o transporte público: candongueiros, TCUL e kupapatas. "
            "Se o carro consome mais de 25% do seu orçamento, pode não valer a pena financeiramente. "
            "Considere usar o carro também como fonte de renda, fazendo táxi nas horas livres, para ajudar a cobrir os custos."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Gráfico comparando o custo mensal de ter carro próprio vs usar transporte público em Luanda para deslocações diárias ao trabalho.",
        "audio_transcript": None,
        "cultural_notes": "O combustível em Angola é dos mais baratos de África devido aos subsídios, mas as estradas e peças importadas encarecem a manutenção.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-037",
        "question": "Como posso incluir despesas com a família alargada no meu orçamento?",
        "answer": (
            "Na cultura angolana, apoiar a família alargada é uma obrigação social e emocional importante. "
            "Defina um valor máximo mensal que pode dar aos familiares sem comprometer as suas despesas essenciais. "
            "Comunique abertamente à família o que pode e não pode fazer — não prometa o que não tem. "
            "Em vez de dar dinheiro sempre, ajude com oportunidades: arranje emprego, ensine um ofício, pague uma formação. "
            "Reserve uma linha específica no orçamento para 'apoio familiar', entre 5% e 15% do rendimento. "
            "Aprenda a dizer não quando o pedido ultrapassa as suas capacidades — ajudar os outros não pode destruir as suas finanças."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "intermediate",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "Em Angola, a família alargada tem grande importância cultural. Quem tem rendimento é esperado que apoie pais, irmãos, tios e sobrinhos.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-BUD-038",
        "question": "Como fazer o orçamento para as festas de fim de ano em Angola?",
        "answer": (
            "O Natal e o Ano Novo em Angola envolvem gastos grandes com comida, roupa nova e presentes. "
            "Comece a poupar desde Setembro, guardando pelo menos Kz 15.000 a Kz 30.000 por mês. "
            "Faça uma lista do que realmente precisa: comida para a ceia, roupa para as crianças, e um orçamento para presentes. "
            "Compre os presentes e a comida com antecedência, porque os preços sobem bué em Dezembro. "
            "Não use todo o 13.º salário nas festas — guarde pelo menos metade para Janeiro, que é um mês pesado. "
            "Festas boas não precisam de ser caras; o importante é estar com a família."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Calendário de poupança de Setembro a Dezembro com metas semanais e checklist de compras para as festas de fim de ano.",
        "audio_transcript": None,
        "cultural_notes": "O Natal é a festa mais importante do calendário angolano, com grande pressão social para demonstrar abundância.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-039",
        "question": "Como é que a AGT (Administração Geral Tributária) afecta o meu orçamento pessoal?",
        "answer": (
            "Se trabalha por conta de outrem, o IRT (Imposto sobre o Rendimento do Trabalho) é retido na fonte pelo empregador. "
            "As taxas de IRT variam de 0% para rendimentos até Kz 100.000 a 25% para os mais altos. "
            "Se tem negócio próprio, precisa de se registar na AGT e pagar o Imposto Industrial. "
            "Inclua os impostos no orçamento para não ter surpresas — calcule o salário líquido, não o bruto. "
            "Guarde os recibos de pagamento de impostos porque podem ser pedidos a qualquer momento. "
            "Consulte um contabilista se o seu negócio cresce, para evitar multas e aproveitar deduções legais."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Tabela de escalões do IRT em Angola com as percentagens aplicáveis a cada faixa de rendimento em Kwanzas.",
        "audio_transcript": None,
        "cultural_notes": "A AGT tem vindo a aumentar a fiscalização sobre o sector informal, exigindo maior formalização dos negócios.",
        "sources": ["Banco Nacional de Angola", "AGT"],
    },
    {
        "id": "AO-BUD-040",
        "question": "Como fazer o orçamento para uma refeição diária típica de uma família angolana?",
        "answer": (
            "Uma refeição típica angolana pode incluir funge com mufete de peixe, feijão de óleo de palma ou calulu. "
            "O custo diário de alimentação para uma família de 4 pessoas fica entre Kz 3.000 e Kz 8.000. "
            "Compre os ingredientes base em quantidade no início da semana: fuba, feijão, óleo de palma e peixe seco. "
            "O peixe fresco é mais caro; alterne com peixe seco (bacalhau de terra) e carne nos dias especiais. "
            "Cozinhar em casa sai muito mais barato do que comer em bancas de rua. "
            "Planeie o menu semanal para aproveitar os mesmos ingredientes em diferentes pratos e reduzir o desperdício."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Menu semanal ilustrado com pratos típicos angolanos e o custo estimado de cada refeição em Kwanzas.",
        "audio_transcript": None,
        "cultural_notes": "O funge (pirão de fuba de milho ou mandioca) é o prato base da alimentação angolana, normalmente acompanhado de peixe ou carne com molho.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-041",
        "question": "Como posso usar o telemóvel para controlar o meu orçamento em Angola?",
        "answer": (
            "O Multicaixa Express permite ver o saldo e as transacções da conta bancária directamente no telemóvel. "
            "Use a funcionalidade de transferências para separar dinheiro em diferentes contas por categoria: renda, comida, poupança. "
            "Existem aplicações gratuitas como o Money Manager que funcionam offline para registar gastos. "
            "Defina lembretes no telemóvel para o dia de pagar contas fixas e evitar multas por atraso. "
            "Tire fotos dos recibos de compras importantes para ter um registo digital organizado. "
            "Mesmo que use um telemóvel simples, pode anotar os gastos nas notas do telefone."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Capturas de ecrã do Multicaixa Express mostrando funcionalidades de controlo de saldo e transferências entre contas.",
        "audio_transcript": None,
        "cultural_notes": "O Multicaixa Express tem vindo a revolucionar os pagamentos em Angola, especialmente entre a população sem acesso bancário tradicional.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-042",
        "question": "Quanto devo gastar com comunicações (recargas e internet) no meu orçamento?",
        "answer": (
            "Os gastos com recargas de telemóvel e internet podem acumular-se sem que a pessoa se aperceba. "
            "Um pacote de internet mensal custa entre Kz 2.000 e Kz 15.000 dependendo do operador e do volume de dados. "
            "As recargas para chamadas podem somar Kz 5.000 a Kz 20.000 por mês se não houver controlo. "
            "Defina um limite semanal para recargas e use mais o WhatsApp e chamadas por internet para poupar. "
            "Compare os pacotes dos operadores (Unitel, Movicel, Africell) e escolha o que melhor se adapta ao seu uso. "
            "As comunicações não devem ultrapassar 5% do orçamento mensal."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "Angola tem três operadores de telecomunicações. A Unitel é a maior, mas a concorrência tem vindo a melhorar os preços.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-BUD-043",
        "question": "Como posso orçamentar a compra de roupa para a família em Angola?",
        "answer": (
            "A roupa é uma despesa importante em Angola, especialmente com a pressão social de estar sempre bem apresentado. "
            "Reserve uma verba trimestral em vez de mensal para roupa, entre Kz 20.000 e Kz 60.000 por trimestre. "
            "Os mercados de roupa usada (calamidades) oferecem peças de qualidade a preços muito mais baixos. "
            "Para roupa das crianças que crescem rápido, troque com outros pais ou compre em segunda mão. "
            "Compre peças básicas e versáteis que combinem entre si, em vez de muitas peças que só se usam uma vez. "
            "Evite comprar roupa por impulso; espere pelas promoções e liquidações."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Gráfico comparativo de preços de roupa nova em loja, roupa nova no mercado informal e roupa usada (calamidades) para peças comuns.",
        "audio_transcript": None,
        "cultural_notes": "Os mercados de 'calamidades' (roupa usada importada) são muito populares em Angola e oferecem roupa a preços acessíveis.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-044",
        "question": "Como planificar o orçamento para quem paga renda de casa em dólares?",
        "answer": (
            "Em Luanda, muitas rendas de casa são cobradas em dólares, o que cria risco cambial. "
            "Se ganha em Kwanzas mas paga renda em dólares, reserve sempre um valor acima da renda actual para cobrir flutuações cambiais. "
            "Negocie com o senhorio para pagar em Kwanzas a uma taxa fixa acordada mensalmente. "
            "Se possível, tente mudar para um imóvel com renda em Kwanzas para ter mais previsibilidade. "
            "Acompanhe a taxa de câmbio semanalmente para não ser apanhado de surpresa no dia do pagamento. "
            "A renda não deve ultrapassar 30% do rendimento líquido mensal, considerando a taxa de câmbio mais desfavorável."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Gráfico mostrando o impacto da variação cambial Kz/USD no custo real de uma renda de $500 ao longo de 12 meses.",
        "audio_transcript": None,
        "cultural_notes": "Embora o BNA tenha proibido a indexação de rendas ao dólar, a prática persiste no mercado imobiliário angolano.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-045",
        "question": "Qual é a melhor forma de controlar dívidas no orçamento familiar angolano?",
        "answer": (
            "O primeiro passo é listar todas as dívidas: fiado na loja, empréstimos de kambas, crédito bancário. "
            "Ordene as dívidas da mais cara (com juros mais altos) para a mais barata e pague primeiro as mais caras. "
            "Reserve pelo menos 20% do rendimento mensal para pagar dívidas até ficar livre delas. "
            "Não contraia novas dívidas enquanto estiver a pagar as antigas — quebre o ciclo. "
            "Se tem dívida no banco, negocie a reestruturação para ter uma prestação que caiba no orçamento. "
            "Evite comprar fiado regularmente; pague a pronto mesmo que signifique comprar menos de cada vez."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Diagrama de 'bola de neve' de dívidas mostrando a estratégia de pagar primeiro as dívidas com juros mais altos.",
        "audio_transcript": None,
        "cultural_notes": "O fiado é uma prática muito comum em Angola, tanto no comércio informal como entre famílias e amigos.",
        "sources": ["Banco Nacional de Angola", "CMC"],
    },
    {
        "id": "AO-BUD-046",
        "question": "Como é que um jovem universitário pode gerir o orçamento em Angola?",
        "answer": (
            "Como universitário, o primeiro passo é saber exactamente quanto recebe dos pais ou do trabalho e listar todas as despesas. "
            "As propinas podem variar de Kz 15.000 a Kz 80.000 por mês; pague-as sempre em primeiro lugar. "
            "Partilhe a renda com colegas para dividir custos de habitação e electricidade. "
            "Compre fotocópias e sebéntas em vez de livros novos — são muito mais baratas. "
            "Cozinhe em casa com outros colegas e divida os custos dos ingredientes. "
            "Se possível, arranje um biscate compatível com os estudos para ter rendimento extra."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "Muitos universitários angolanos estudam à noite para poder trabalhar durante o dia e custear os estudos.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-BUD-047",
        "question": "Como posso adaptar o meu orçamento quando os preços do combustível sobem em Angola?",
        "answer": (
            "Quando o preço do combustível sobe, os efeitos propagam-se a toda a economia: transporte, alimentos e serviços ficam mais caros. "
            "Reveja imediatamente o orçamento e identifique despesas que pode reduzir temporariamente. "
            "Use mais o transporte público e reduza viagens desnecessárias de carro. "
            "Compre alimentos em maior quantidade de uma só vez para reduzir deslocações ao mercado. "
            "Antecipe compras de bens essenciais antes que os preços subam ainda mais. "
            "Considere alternativas energéticas: painéis solares pequenos podem reduzir a dependência do gerador a gasóleo."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Gráfico de cascata mostrando como o aumento de 20% no combustível impacta os preços de transporte, alimentação e serviços em Angola.",
        "audio_transcript": None,
        "cultural_notes": "Angola é produtora de petróleo mas importa combustíveis refinados. Os subsídios ao combustível têm sido progressivamente reduzidos.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-048",
        "question": "Qual é a melhor forma de negociar preços nas praças de Luanda para poupar no orçamento?",
        "answer": (
            "Nas praças de Luanda, o preço inicial quase nunca é o preço final — negociar é parte da cultura. "
            "Comece por perguntar o preço sem mostrar demasiado interesse e ofereça 30% a 40% menos. "
            "Compare preços entre vários vendedores antes de comprar — caminhe pela praça toda. "
            "Compre em quantidade: os vendedores dão melhores preços quando compra mais. "
            "Vá ao mercado no final do dia, quando os vendedores querem despachar a mercadoria e fazem melhores preços. "
            "Crie relação com vendedores fixos — ser cliente habitual traz descontos e qualidade garantida."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "A negociação de preços é uma tradição cultural forte nos mercados angolanos, diferente dos supermercados com preços fixos.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-049",
        "question": "Como criar um fundo para despesas imprevistas no orçamento angolano?",
        "answer": (
            "Os imprevistos em Angola podem ser caros: uma avaria no gerador, uma doença, ou uma contribuição para um óbito. "
            "O ideal é ter uma reserva de pelo menos Kz 100.000 a Kz 300.000, mas comece com o que puder. "
            "Separe 5% a 10% do rendimento mensal para este fundo, automaticamente antes de gastar noutras coisas. "
            "Guarde este dinheiro numa conta separada ou num envelope lacrado em casa. "
            "Só use o fundo para verdadeiras emergências, não para oportunidades de compra. "
            "Quando usar o fundo, reponha-o o mais rápido possível nos meses seguintes."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Gráfico mostrando o crescimento de um fundo de emergência com contribuições mensais de Kz 15.000 ao longo de 12 meses.",
        "audio_transcript": None,
        "cultural_notes": "As obrigações sociais imprevistas (óbitos, doenças de familiares) são uma das principais causas de descontrolo financeiro em Angola.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-050",
        "question": "Qual é a importância de definir prioridades no orçamento quando o salário é baixo?",
        "answer": (
            "Quando o salário é baixo, cada Kwanza conta e definir prioridades é fundamental para não passar necessidade. "
            "A ordem de prioridade deve ser: alimentação, habitação, saúde, transporte para o trabalho, e só depois o resto. "
            "Use a regra do essencial primeiro: pague as necessidades antes de pensar em desejos. "
            "Diga não a pressões sociais de gastar em festas e aparências quando o básico não está garantido. "
            "Procure formas de aumentar o rendimento: biscates, venda de produtos, prestação de serviços. "
            "Mesmo com salário baixo, guarde algo todos os meses — Kz 2.000 por mês são Kz 24.000 no final do ano."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Pirâmide de prioridades financeiras para famílias de baixo rendimento em Angola, com alimentação na base e lazer no topo.",
        "audio_transcript": None,
        "cultural_notes": "Muitos angolanos vivem com rendimentos abaixo do limiar de pobreza, exigindo uma gestão financeira muito cuidadosa.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-BUD-051",
        "question": "Como posso usar o crédito Multicaixa Express para gerir melhor o orçamento?",
        "answer": (
            "O Multicaixa Express permite fazer pagamentos, transferências e consultar saldos directamente pelo telemóvel. "
            "Utilize-o para pagar contas de água, luz e telefone sem precisar de ir ao banco ou ficar em filas. "
            "Configure transferências automáticas no dia em que recebe para separar dinheiro por categorias. "
            "O histórico de transacções funciona como um registo automático dos seus gastos. "
            "Compare o saldo antes e depois de cada semana para ver quanto gastou. "
            "Tenha atenção às taxas de transacção para que não comam uma parte significativa do seu orçamento."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Fluxograma mostrando como usar o Multicaixa Express para separar o salário em diferentes categorias de despesa no dia do pagamento.",
        "audio_transcript": None,
        "cultural_notes": "O Multicaixa Express é o principal serviço de pagamentos móveis em Angola, gerido pela EMIS.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-052",
        "question": "Como orçamentar a construção de uma casa em Angola, tijolo a tijolo?",
        "answer": (
            "A construção 'tijolo a tijolo' é a forma mais comum de construir casa em Angola quando não se tem crédito bancário. "
            "Comece por adquirir o terreno e regularize a documentação na administração do município. "
            "Faça um projecto simples com um técnico para saber a quantidade total de material necessário. "
            "Compre material em fases: primeiro a fundação, depois as paredes, depois a cobertura. "
            "Reserve entre 20% e 30% do rendimento mensal para compra de material de construção. "
            "Compare preços entre lojas de material e mercados informais — a diferença pode ser de 30% a 40%."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Cronograma de construção por fases com os custos estimados de cada etapa (fundação, paredes, cobertura, acabamentos) em Kwanzas.",
        "audio_transcript": None,
        "cultural_notes": "A autoconstrução progressiva é a principal forma de acesso à habitação em Angola, dada a escassez de crédito hipotecário acessível.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-053",
        "question": "Como posso fazer um orçamento para quem trabalha na pesca artesanal em Angola?",
        "answer": (
            "A pesca artesanal tem rendimentos muito variáveis, dependendo da época, do mar e da sorte. "
            "Nos meses bons, guarde pelo menos 40% dos ganhos para cobrir os meses em que o mar não dá. "
            "Inclua no orçamento os custos de manutenção do barco, redes e combustível. "
            "Diversifique as fontes de renda: venda peixe fresco, peixe seco e marisco para diferentes mercados. "
            "Forme um grupo com outros pescadores para partilhar custos de combustível e equipamento. "
            "Mantenha um registo diário da pesca e das vendas para saber qual é realmente o seu rendimento médio."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "intermediate",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "A pesca artesanal é uma actividade económica importante em províncias costeiras como Benguela, Namibe e Luanda.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-054",
        "question": "Como incluir contribuições para a igreja ou mesquita no orçamento familiar?",
        "answer": (
            "As contribuições religiosas são importantes para muitas famílias angolanas e devem ser planeadas. "
            "Defina um valor fixo mensal que não comprometa as despesas essenciais — entre 5% e 10% do rendimento é razoável. "
            "Se a igreja pede contribuições extraordinárias frequentes, avalie se estão dentro das suas possibilidades. "
            "Não se endivide para dar ofertas — a fé não exige que passe necessidade. "
            "Inclua as contribuições religiosas como uma linha fixa do orçamento, tal como qualquer outra despesa. "
            "Se o rendimento diminuir, ajuste a contribuição sem culpa; Deus entende as dificuldades."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "basic",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "Angola é um país muito religioso, com forte presença de igrejas cristãs. As contribuições religiosas representam uma parcela significativa do orçamento de muitas famílias.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-BUD-055",
        "question": "Qual é a melhor forma de orçamentar despesas com seguro em Angola?",
        "answer": (
            "Os seguros em Angola ainda são pouco utilizados, mas podem proteger o orçamento de choques financeiros grandes. "
            "O seguro automóvel é obrigatório e custa a partir de Kz 20.000 anuais para responsabilidade civil. "
            "O seguro de saúde pode custar entre Kz 30.000 e Kz 150.000 por mês, dependendo da cobertura. "
            "Compare pelo menos três seguradoras antes de escolher: ENSA, Nossa Seguros, AAA Seguros, entre outras. "
            "Reserve uma linha específica no orçamento para prémios de seguro, pagando idealmente de forma mensal. "
            "Comece pelo seguro mais urgente — saúde se tiver filhos pequenos, automóvel se tiver carro — e vá adicionando conforme possível."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Tabela comparativa de tipos de seguro disponíveis em Angola com custos médios mensais e cobertura oferecida.",
        "audio_transcript": None,
        "cultural_notes": "O mercado de seguros em Angola é pequeno mas está a crescer. A ARSEG é a entidade reguladora.",
        "sources": ["Banco Nacional de Angola", "CMC"],
    },
    {
        "id": "AO-BUD-056",
        "question": "Como adaptar o orçamento familiar quando um membro da família emigra?",
        "answer": (
            "Quando um familiar emigra, o orçamento muda em duas direcções: menos uma boca para alimentar, mas mais despesas com comunicações e envio de dinheiro. "
            "Se o emigrante vai enviar remessas, não conte com esse dinheiro como rendimento fixo — trate como bónus. "
            "Defina com o familiar quanto e quando vai enviar para poder planificar sem ansiedade. "
            "Use canais oficiais de transferência (Western Union, bancos) para ter segurança e comprovativo. "
            "O custo da emigração em si (passaporte, visto, bilhete, dinheiro inicial) deve ser orçamentado com antecedência. "
            "Não endivide a família toda para custear a emigração de um membro — calcule se o retorno financeiro compensa o investimento."
        ),
        "country": "AO",
        "topic": "budget",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Gráfico de fluxo de remessas mostrando os custos de envio por diferentes canais e o valor líquido que chega à família em Angola.",
        "audio_transcript": None,
        "cultural_notes": "A emigração é uma estratégia económica comum em Angola, com destinos principais sendo Portugal, Brasil, Namíbia e África do Sul.",
        "sources": ["Banco Nacional de Angola"],
    },
    # =========================================================================
    # SAVINGS — 39 pairs (AO-SAV-019 .. AO-SAV-057)
    # =========================================================================
    {
        "id": "AO-SAV-019",
        "question": "O que é uma kixikila e como funciona esta poupança rotativa?",
        "answer": (
            "A kixikila é um sistema de poupança rotativa tradicional angolano em que um grupo de pessoas contribui mensalmente com um valor fixo. "
            "Cada mês, um membro diferente recebe o total das contribuições, e o ciclo repete-se até todos receberem. "
            "Por exemplo, 10 kambas contribuem Kz 20.000 cada; a cada mês, um recebe Kz 200.000. "
            "A confiança entre os membros é fundamental — escolha pessoas responsáveis e de palavra. "
            "A kixikila ajuda a juntar uma quantia grande que seria difícil poupar sozinho. "
            "Defina regras claras desde o início: valor, data de pagamento, ordem de recebimento e consequências para quem falhar."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Diagrama circular mostrando como funciona uma kixikila de 10 membros com contribuição mensal de Kz 20.000, indicando a rotação de quem recebe.",
        "audio_transcript": None,
        "cultural_notes": "A kixikila é uma prática ancestral de poupança colectiva presente em todo o território angolano, especialmente entre mulheres.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-020",
        "question": "Vale mais a pena fazer depósito a prazo em Kwanzas ou em dólares?",
        "answer": (
            "O depósito a prazo em Kwanzas geralmente oferece juros mais altos (10% a 18% ao ano) do que em dólares (2% a 5%). "
            "No entanto, se o Kwanza desvalorizar mais do que a diferença de juros, o depósito em dólares sai a ganhar. "
            "Para quem precisa do dinheiro em Kwanzas para despesas diárias, o depósito em Kz é mais prático. "
            "Para poupança de longo prazo, diversifique: parte em Kwanzas para os juros altos e parte em dólares para protecção cambial. "
            "Compare as taxas entre bancos — BFA, BAI, BIC e Standard Bank costumam ter condições diferentes. "
            "Lembre-se que os juros do depósito a prazo são tributados pelo IRT sobre rendimentos de capital."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Gráfico comparativo do rendimento real (após inflação e câmbio) de depósitos a prazo em Kwanzas vs dólares nos últimos 3 anos.",
        "audio_transcript": None,
        "cultural_notes": "A desconfiança no Kwanza leva muitos angolanos a preferirem poupar em dólares, apesar dos juros menores.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-021",
        "question": "Como posso proteger as minhas poupanças da inflação alta em Angola?",
        "answer": (
            "A inflação em Angola pode ultrapassar 15% ao ano, o que significa que o dinheiro parado perde valor rapidamente. "
            "Não guarde grandes quantias em Kwanzas debaixo do colchão — esse dinheiro está a perder valor todos os dias. "
            "Coloque a poupança num depósito a prazo com taxa de juro acima da inflação, se possível. "
            "Invista em bens que acompanham a inflação: terrenos, materiais de construção ou mercadoria para revenda. "
            "Converta parte da poupança em dólares como protecção contra a desvalorização do Kwanza. "
            "Compre Obrigações do Tesouro que oferecem rendimentos indexados à inflação."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Gráfico de linha dupla mostrando a taxa de inflação em Angola vs taxas de juro dos depósitos a prazo nos últimos 5 anos.",
        "audio_transcript": None,
        "cultural_notes": "A inflação persistente em Angola erode rapidamente as poupanças, sendo um dos maiores desafios para a educação financeira.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-022",
        "question": "Como posso usar o Multicaixa Express como ferramenta de poupança?",
        "answer": (
            "O Multicaixa Express permite abrir uma conta e fazer depósitos pelo telemóvel sem ir ao banco. "
            "Configure uma transferência automática mensal do saldo principal para uma conta poupança separada. "
            "Use a funcionalidade de pagamento de serviços para evitar gastos com deslocações ao banco. "
            "O dinheiro no Multicaixa Express é mais difícil de gastar por impulso do que o dinheiro no bolso. "
            "Verifique o saldo regularmente para acompanhar o crescimento da sua poupança. "
            "Combine o Multicaixa Express com uma conta poupança formal num banco para obter juros sobre o valor depositado."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Passo a passo ilustrado de como configurar uma transferência automática mensal para poupança no Multicaixa Express.",
        "audio_transcript": None,
        "cultural_notes": "O Multicaixa Express tem sido fundamental na inclusão financeira de angolanos sem acesso a agências bancárias.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-023",
        "question": "É melhor guardar dinheiro em casa ou no banco em Angola?",
        "answer": (
            "Guardar dinheiro em casa é arriscado: pode ser roubado, destruído por inundações ou incêndios, e perde valor com a inflação. "
            "No banco, o dinheiro está protegido pelo Fundo de Garantia de Depósitos até um certo limite. "
            "Os bancos pagam juros sobre depósitos a prazo, o que ajuda a combater a inflação. "
            "A desconfiança nos bancos é compreensível em Angola, onde já houve problemas de liquidez em alguns bancos. "
            "Se não confia no banco, use pelo menos o Multicaixa Express como alternativa a ter bazuka em casa. "
            "Uma solução equilibrada: mantenha um valor pequeno em casa para emergências imediatas e o resto no banco."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Tabela de prós e contras de guardar dinheiro em casa vs no banco, incluindo riscos, rendimento e acessibilidade.",
        "audio_transcript": None,
        "cultural_notes": "Muitos angolanos ainda preferem guardar dinheiro em casa devido à desconfiança no sistema bancário e à dificuldade de acesso a agências.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-024",
        "question": "O que são Obrigações do Tesouro angolanas e como posso investir nelas?",
        "answer": (
            "As Obrigações do Tesouro (OT) são títulos de dívida emitidos pelo Estado angolano para financiar o orçamento. "
            "Ao comprar uma OT, está basicamente a emprestar dinheiro ao governo, que lhe paga juros periódicos. "
            "As OT em Kwanzas podem oferecer taxas de 15% a 22% ao ano, superiores aos depósitos a prazo. "
            "Para investir, precisa de ter conta num banco e pedir ao gestor de conta que compre as OT por si. "
            "Existem também OT em dólares, com juros menores mas com protecção cambial. "
            "O investimento mínimo varia, mas alguns bancos permitem a partir de Kz 1.000.000."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Gráfico mostrando o rendimento das Obrigações do Tesouro angolanas em Kz e USD nos últimos anos, comparado com depósitos a prazo.",
        "audio_transcript": None,
        "cultural_notes": "As OT angolanas são consideradas de baixo risco doméstico mas têm risco cambial para investidores em dólares.",
        "sources": ["Banco Nacional de Angola", "CMC"],
    },
    {
        "id": "AO-SAV-025",
        "question": "Como poupar para construir casa 'tijolo a tijolo' em Angola?",
        "answer": (
            "A construção progressiva é a forma mais realista de ter casa própria em Angola para a maioria das famílias. "
            "Defina um valor mensal fixo para compra de material de construção: Kz 30.000 a Kz 100.000 conforme o rendimento. "
            "Compre material aos poucos e guarde-o — o preço dos materiais de construção acompanha a inflação. "
            "O terreno deve ser a primeira compra; sem terreno legalizado, todo o investimento fica em risco. "
            "Faça um orçamento total da obra com um pedreiro ou engenheiro para saber quanto precisa ao todo. "
            "Junte-se a uma kixikila com o objectivo específico de comprar material de construção em quantidade."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Cronograma de poupança para construção progressiva de uma casa simples em Angola, dividido por fases e com valores estimados em Kwanzas.",
        "audio_transcript": None,
        "cultural_notes": "A autoconstrução 'tijolo a tijolo' é a principal via de acesso à habitação em Angola, dado o custo proibitivo do mercado imobiliário formal.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-026",
        "question": "Como poupar para abrir o meu próprio negócio em Angola, como uma quitanda ou loja?",
        "answer": (
            "Primeiro, calcule quanto precisa para começar: stock inicial, renda do espaço, licença e reserva para os primeiros meses. "
            "Uma quitanda simples pode começar com Kz 100.000 a Kz 300.000 de investimento inicial. "
            "Poupe pelo menos Kz 20.000 a Kz 50.000 por mês e defina uma data-alvo para abrir o negócio. "
            "Comece a vender em pequena escala enquanto poupa — teste o mercado antes de investir tudo. "
            "Use a kixikila para juntar o capital inicial mais rapidamente com os seus kambas. "
            "Não peça empréstimo para começar; é melhor começar pequeno com dinheiro próprio e crescer organicamente."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Plano de poupança em 12 meses para abrir uma quitanda, com metas mensais e lista de investimento inicial necessário.",
        "audio_transcript": None,
        "cultural_notes": "As quitandas são pequenas mercearias de bairro, essenciais na cadeia de distribuição alimentar angolana.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-SAV-027",
        "question": "Quanto devo ter numa reserva de emergência no contexto angolano?",
        "answer": (
            "O ideal internacional é 3 a 6 meses de despesas, mas em Angola, comece com o que for possível. "
            "Calcule as suas despesas mensais essenciais: renda, comida, transporte, água e luz — esse é o valor mensal de referência. "
            "Tente juntar pelo menos 1 mês de despesas essenciais como primeiro objectivo — para muitas famílias, isso pode ser Kz 80.000 a Kz 200.000. "
            "Guarde a reserva de emergência numa conta separada, preferencialmente com algum rendimento. "
            "Não toque neste dinheiro para nada que não seja uma verdadeira emergência: doença, perda de emprego ou reparação urgente. "
            "Depois de atingir 1 mês, continue a poupar até chegar a 3 meses de despesas."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Gráfico de termómetro mostrando metas progressivas de reserva de emergência: Kz 50.000, Kz 100.000, Kz 200.000 até Kz 600.000.",
        "audio_transcript": None,
        "cultural_notes": "A instabilidade económica angolana torna a reserva de emergência ainda mais importante, mas também mais difícil de constituir.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-028",
        "question": "Como poupar dinheiro para estudar no exterior estando em Angola?",
        "answer": (
            "Estudar no exterior exige planeamento financeiro de longo prazo — comece a poupar pelo menos 2 a 3 anos antes. "
            "Calcule todos os custos: propinas (variam muito por país), bilhete de avião, seguro, alojamento e alimentação. "
            "Poupe em dólares ou euros para evitar a desvalorização do Kwanza ao longo do tempo. "
            "Candidate-se a bolsas de estudo: INAGBE, Fundação Calouste Gulbenkian, Chevening e outras oferecem bolsas para angolanos. "
            "Trabalhe em part-time ou faça biscates para reforçar a poupança sem depender apenas da família. "
            "Portugal e Brasil são destinos populares porque não há barreira linguística, mas compare custos com outros países."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Tabela comparativa de custos anuais de estudo em Portugal, Brasil, África do Sul e Cuba para estudantes angolanos.",
        "audio_transcript": None,
        "cultural_notes": "Estudar no exterior é uma aspiração forte para muitos jovens angolanos e é visto como investimento familiar.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-SAV-029",
        "question": "Como funciona um clube de poupança comunitário em Angola?",
        "answer": (
            "Um clube de poupança comunitário é um grupo organizado de pessoas que poupam juntas e podem fazer empréstimos internos. "
            "Cada membro contribui com um valor fixo semanal ou mensal, que é guardado numa caixa comum. "
            "O dinheiro acumulado pode ser emprestado a membros com juro baixo, e os juros revertem para o grupo. "
            "No final do ciclo (geralmente 1 ano), o fundo é dividido entre os membros com os lucros dos juros. "
            "São necessárias regras claras: estatutos, livro de registos, e pelo menos 3 pessoas para gerir a caixa. "
            "O clube é diferente da kixikila porque o dinheiro fica guardado e cresce, em vez de rodar."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Diagrama mostrando a estrutura de um clube de poupança: contribuições semanais, fundo central, empréstimos com juros e distribuição anual.",
        "audio_transcript": None,
        "cultural_notes": "Os clubes de poupança comunitários são promovidos por ONGs em Angola como forma de inclusão financeira nas zonas rurais.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-030",
        "question": "Como aforrar se o meu rendimento vem todo do mercado informal?",
        "answer": (
            "No mercado informal, o segredo é poupar primeiro e gastar depois — antes mesmo de contar o lucro do dia. "
            "Defina um valor mínimo diário de poupança: mesmo Kz 500 por dia são Kz 15.000 por mês. "
            "Use o método do envelope: ao fechar o dia, coloque a poupança num envelope lacrado e não mexa. "
            "Semanalmente, deposite o conteúdo dos envelopes no banco ou no Multicaixa Express. "
            "Nos dias de vendas fracas, poupe menos mas poupe sempre — a consistência é mais importante que o valor. "
            "Não conte com a poupança para repor stock; o dinheiro do negócio e a poupança devem ser separados."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Ilustração do método dos envelopes de poupança diária com um calendário mensal mostrando os depósitos acumulados.",
        "audio_transcript": None,
        "cultural_notes": "A maioria dos trabalhadores informais angolanos não tem conta bancária, tornando métodos simples de poupança essenciais.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-031",
        "question": "Como funciona a poupança conjunta familiar em Angola?",
        "answer": (
            "A poupança conjunta familiar é quando vários membros da família contribuem mensalmente para um objectivo comum. "
            "Pode ser para construir casa, pagar estudos de um filho, ou cobrir emergências médicas da família. "
            "Defina regras claras: quanto cada um contribui, quem guarda o dinheiro, e como se decide sobre o uso. "
            "Use uma conta bancária em nome de dois membros da família para maior segurança e transparência. "
            "Façam uma reunião mensal para prestar contas e decidir os próximos passos. "
            "A contribuição deve ser proporcional ao rendimento de cada um — quem ganha mais, contribui mais."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "A família alargada angolana funciona como unidade económica, e a poupança conjunta reforça os laços familiares e a solidariedade.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-032",
        "question": "Como proteger as minhas poupanças da desvalorização cambial em Angola?",
        "answer": (
            "A desvalorização do Kwanza é um risco constante para quem poupa em moeda nacional. "
            "Diversifique: mantenha parte das poupanças em dólares, parte em Kwanzas com juros altos e parte em bens reais. "
            "Compre Obrigações do Tesouro em dólares para ter rendimento com protecção cambial. "
            "Invista em bens que acompanham ou superam a inflação: terrenos, material de construção, equipamentos. "
            "Não troque todo o dinheiro para dólares de uma vez — faça compras regulares para diluir o risco cambial. "
            "Acompanhe as decisões do BNA sobre a taxa de câmbio para antecipar movimentos."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Gráfico mostrando o valor real de Kz 1.000.000 poupados ao longo de 5 anos em três cenários: debaixo do colchão, depósito a prazo em Kz, e convertido em USD.",
        "audio_transcript": None,
        "cultural_notes": "A memória de grandes desvalorizações do Kwanza faz com que muitos angolanos prefiram activos em dólares ou bens tangíveis.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-033",
        "question": "Como poupar para emergências de saúde em Angola?",
        "answer": (
            "Uma emergência de saúde em Angola pode custar de Kz 50.000 a Kz 5.000.000 dependendo da gravidade. "
            "Separe pelo menos 5% do rendimento mensal para um fundo exclusivo de saúde. "
            "Considere aderir a um seguro de saúde, mesmo que seja o plano mais básico — é mais barato do que pagar tudo do bolso. "
            "Mantenha este fundo numa conta de fácil acesso, porque emergências não esperam. "
            "Junte-se a um grupo de solidariedade para saúde com kambas e familiares. "
            "Invista em prevenção: vacinas, check-ups e alimentação saudável custam menos do que tratar doenças."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Tabela com custos estimados de procedimentos médicos comuns em Angola: consulta, internamento, cirurgia simples, parto, em hospital público vs clínica privada.",
        "audio_transcript": None,
        "cultural_notes": "O sistema de saúde público angolano é gratuito mas tem limitações graves, forçando muitas famílias a recorrer ao sector privado em emergências.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-034",
        "question": "É possível poupar ganhando o salário mínimo em Angola?",
        "answer": (
            "Sim, é possível poupar com salário mínimo, mas exige disciplina e criatividade. "
            "Comece com valores muito pequenos: Kz 1.000 por semana já são Kz 52.000 por ano. "
            "Corte gastos não essenciais: reduza recargas, use transporte mais barato, cozinhe em casa. "
            "Procure formas de rendimento extra: lavar carros ao fim-de-semana, vender bolinhos, fazer biscates. "
            "Use a kixikila para forçar a disciplina de poupar todos os meses. "
            "O mais importante é criar o hábito — mesmo Kz 500 por dia fazem a diferença ao longo do tempo."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "O salário mínimo em Angola é muito baixo face ao custo de vida, especialmente em Luanda, tornando a poupança um desafio real.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-035",
        "question": "Quais são os bancos mais seguros para poupar em Angola?",
        "answer": (
            "Os maiores bancos em Angola são o BAI, BFA, BIC, BPC e Banco de Poupança e Crédito. "
            "A segurança de um banco depende do seu capital, liquidez e supervisão pelo BNA (Banco Nacional de Angola). "
            "Verifique se o banco é membro do Fundo de Garantia de Depósitos, que protege até um certo valor. "
            "Não coloque todas as poupanças num só banco — diversifique entre dois ou três. "
            "Compare as taxas de juro oferecidas: podem variar bué entre bancos para o mesmo tipo de depósito. "
            "Escolha um banco com boa rede de ATMs e Multicaixa na sua zona para facilitar o acesso."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Tabela comparativa dos 5 maiores bancos angolanos com taxas de juro de poupança, número de agências e rede ATM.",
        "audio_transcript": None,
        "cultural_notes": "O sector bancário angolano passou por consolidação nos últimos anos, com o BNA a revogar licenças de bancos frágeis.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-036",
        "question": "Como posso poupar para a reforma (aposentadoria) em Angola?",
        "answer": (
            "O sistema de segurança social angolano (INSS) paga pensões, mas os valores são geralmente baixos. "
            "Se trabalha formalmente, confirme que o seu empregador está a fazer os descontos para o INSS. "
            "Complemente a pensão do INSS com poupança privada: depósitos a prazo, OT ou investimento em bens. "
            "Comece a poupar para a reforma o mais cedo possível — o efeito dos juros compostos é poderoso. "
            "Investir em terrenos ou imóveis para arrendar pode gerar rendimento passivo na reforma. "
            "Fale com o seu banco sobre planos de poupança-reforma voluntários, que alguns já oferecem."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Gráfico mostrando o crescimento de uma poupança mensal de Kz 20.000 ao longo de 20 e 30 anos com juros compostos de 12% ao ano.",
        "audio_transcript": None,
        "cultural_notes": "Muitos angolanos não contribuem para o INSS por trabalharem no sector informal, ficando sem protecção na velhice.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-SAV-037",
        "question": "O que é o Depósito a Prazo Não Mobilizável e quando compensa?",
        "answer": (
            "O Depósito a Prazo Não Mobilizável (DPNM) é um depósito em que não pode levantar o dinheiro antes do prazo. "
            "Em troca da menor liquidez, os bancos oferecem taxas de juro mais altas — geralmente 2% a 5% acima do depósito normal. "
            "Compensa quando tem dinheiro que sabe que não vai precisar durante 6 meses, 1 ano ou mais. "
            "É ideal para poupança de objectivos de médio prazo: construção de casa, casamento ou educação. "
            "Compare as taxas entre bancos e preste atenção ao prazo mínimo e às penalizações por levantamento antecipado. "
            "Não coloque neste tipo de depósito dinheiro que possa precisar para emergências."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "Os depósitos a prazo são um dos instrumentos financeiros mais utilizados em Angola, dado o mercado de capitais ainda em desenvolvimento.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-038",
        "question": "Como posso ensinar os meus filhos a poupar desde pequenos em Angola?",
        "answer": (
            "Dê ao seu filho um mealheiro (porquinho) e incentive-o a guardar moedas todos os dias. "
            "Quando receber mesada, ensine a regra dos três montes: gastar, poupar e partilhar. "
            "Defina objectivos concretos: 'se poupares Kz 5.000, podes comprar aquele brinquedo'. "
            "Leve as crianças ao banco para abrirem a primeira conta poupança — muitos bancos angolanos têm contas jovens. "
            "Conte histórias de pessoas que conseguiram coisas boas porque pouparam com disciplina. "
            "Dê o exemplo: se os filhos virem os pais a poupar, vão imitar naturalmente o comportamento."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Ilustração de três recipientes coloridos para crianças: 'Gastar', 'Poupar' e 'Dar', com moedas e notas de Kwanzas.",
        "audio_transcript": None,
        "cultural_notes": "A educação financeira infantil em Angola é informal e passa principalmente pela família.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-039",
        "question": "Como funciona uma conta poupança no banco em Angola?",
        "answer": (
            "Uma conta poupança é uma conta bancária que paga juros sobre o saldo depositado. "
            "Para abrir, precisa de BI (bilhete de identidade), NIF (número de identificação fiscal) e comprovativo de residência. "
            "Os juros das contas poupança em Angola variam de 3% a 8% ao ano em Kwanzas. "
            "Pode fazer depósitos a qualquer momento, mas os levantamentos podem ter limites mensais. "
            "Alguns bancos exigem um saldo mínimo; verifique para evitar cobranças de manutenção. "
            "A conta poupança é ideal para a reserva de emergência porque combina rendimento com acessibilidade."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Checklist ilustrada dos documentos necessários para abrir uma conta poupança em Angola e comparação de taxas entre bancos.",
        "audio_transcript": None,
        "cultural_notes": "A penetração bancária em Angola é baixa, com muitos angolanos a nunca terem tido uma conta bancária.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-040",
        "question": "Vale a pena comprar ouro como forma de poupança em Angola?",
        "answer": (
            "O ouro é considerado um refúgio de valor em tempos de instabilidade económica e cambial. "
            "Em Angola, não existe um mercado formal de ouro para pequenos investidores, o que dificulta a compra e venda. "
            "Comprar jóias de ouro não é o mesmo que investir em ouro — as jóias têm margem do ourives que se perde na revenda. "
            "Se quiser investir em ouro, procure fundos de investimento internacionais acessíveis através de bancos angolanos. "
            "O ouro não paga juros nem dividendos — o ganho vem apenas da valorização do preço. "
            "Para a maioria dos angolanos, há alternativas mais acessíveis e práticas como depósitos a prazo e OT."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "advanced",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "Angola tem reservas de ouro mas o mercado de investimento em ouro para o público em geral é praticamente inexistente.",
        "sources": ["Banco Nacional de Angola", "CMC"],
    },
    {
        "id": "AO-SAV-041",
        "question": "Como posso poupar nas despesas diárias para ter mais dinheiro no final do mês?",
        "answer": (
            "Pequenas poupanças diárias acumulam bué ao longo do mês. "
            "Leve marmita de casa em vez de comprar almoço na rua — pode poupar Kz 2.000 a Kz 5.000 por dia. "
            "Use recargas de dados em vez de pacotes grandes se não gasta tudo — evite o desperdício. "
            "Caminhe em distâncias curtas em vez de apanhar candongueiro para tudo. "
            "Compre a granel com vizinhos ou kambas e dividam os custos e os produtos. "
            "Anote tudo que gasta durante uma semana — vai ficar surpreendido com quanto vai em pequenas compras."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Infográfico com 6 dicas práticas de poupança diária com o valor estimado que se pode poupar por mês em cada uma.",
        "audio_transcript": None,
        "cultural_notes": "Em Angola, os pequenos gastos diários com comida de rua, recargas e transporte somam valores significativos ao fim do mês.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-042",
        "question": "Como funciona o Certificado de Aforro em Angola?",
        "answer": (
            "Os Certificados de Aforro são instrumentos de poupança emitidos pelo Estado, semelhantes às Obrigações do Tesouro mas mais acessíveis. "
            "São destinados ao pequeno investidor e geralmente têm valores mínimos de investimento mais baixos. "
            "Os juros são pagos periodicamente e costumam ser competitivos face à inflação. "
            "Para subscrever, dirija-se ao seu banco e peça informação sobre as emissões disponíveis. "
            "São considerados seguros porque são garantidos pelo Estado angolano. "
            "Têm prazos definidos, mas geralmente permitem resgate antecipado com alguma penalização."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "O desenvolvimento do mercado de títulos públicos acessíveis ao público em geral é uma prioridade do governo angolano para fomentar a poupança interna.",
        "sources": ["Banco Nacional de Angola", "CMC"],
    },
    {
        "id": "AO-SAV-043",
        "question": "Devo poupar para comprar carro ou é melhor usar transporte público?",
        "answer": (
            "Antes de poupar para um carro, faça as contas: o custo total mensal de ter carro em Luanda pode ser Kz 100.000 a Kz 250.000. "
            "Se gasta menos de Kz 60.000 por mês em transporte público, o carro pode não compensar financeiramente. "
            "Um carro pode ser fonte de renda extra se fizer táxi ou entregas nas horas vagas. "
            "Se decidir comprar, poupe para pagar a pronto — os juros do crédito automóvel em Angola são muito altos. "
            "Considere um carro usado em bom estado em vez de novo — a desvalorização do carro novo é brutal. "
            "Não sacrifique a reserva de emergência para comprar carro; poupe separadamente para cada objectivo."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Calculadora visual comparando custo total mensal de ter carro vs usar transporte público em Luanda durante 1 ano.",
        "audio_transcript": None,
        "cultural_notes": "Em Luanda, ter carro é também um símbolo de estatuto social, criando pressão para comprar mesmo quando não compensa financeiramente.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-044",
        "question": "Como posso usar a agricultura urbana para poupar com alimentação em Angola?",
        "answer": (
            "A agricultura urbana é uma forma inteligente de reduzir os gastos com alimentação, mesmo em Luanda. "
            "Um quintal pequeno pode produzir tomate, cebola, couve e pimenta suficientes para reduzir as compras na praça. "
            "O investimento inicial é baixo: sementes, terra e ferramentas básicas custam menos de Kz 10.000. "
            "A poupança pode ser de Kz 5.000 a Kz 15.000 por mês só em legumes e temperos. "
            "Se produzir em excesso, pode vender aos vizinhos e ter rendimento extra. "
            "Junte-se a cooperativas urbanas para trocar conhecimentos, sementes e técnicas."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Plano ilustrado de uma pequena horta urbana com os produtos mais fáceis de cultivar em Angola e a poupança mensal estimada.",
        "audio_transcript": None,
        "cultural_notes": "A agricultura urbana tem crescido em Angola, especialmente nos musseques de Luanda, como forma de complementar a alimentação familiar.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-045",
        "question": "Qual é a diferença entre poupança de curto prazo e longo prazo no contexto angolano?",
        "answer": (
            "A poupança de curto prazo (até 1 ano) serve para objectivos próximos: festas, material escolar, reparações. "
            "A poupança de longo prazo (mais de 1 ano) é para objectivos grandes: casa, carro, reforma, educação dos filhos. "
            "Para curto prazo, use conta poupança ou depósito a prazo de 3 a 6 meses, com boa liquidez. "
            "Para longo prazo, use depósitos a prazo maiores, OT ou investimentos em bens reais. "
            "No contexto angolano de inflação alta, a poupança de longo prazo em Kwanzas perde valor se os juros não superarem a inflação. "
            "Diversifique a poupança de longo prazo entre Kwanzas, dólares e bens para proteger contra diferentes riscos."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Diagrama de dois caminhos: poupança de curto prazo (objectivos e instrumentos) e poupança de longo prazo (objectivos e instrumentos) no contexto angolano.",
        "audio_transcript": None,
        "cultural_notes": "A instabilidade macroeconómica angolana torna a poupança de longo prazo mais complexa e exige diversificação.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-046",
        "question": "Como poupar se tenho muitas obrigações sociais (óbitos, casamentos, baptizados)?",
        "answer": (
            "As obrigações sociais em Angola são uma despesa real e frequente que deve ser incluída no planeamento. "
            "Crie um fundo específico para eventos sociais, separado da poupança principal, com contribuição mensal de Kz 10.000 a Kz 30.000. "
            "Aprenda a dosear a contribuição: não precisa dar o mesmo valor em todos os eventos. "
            "Participe em grupos de solidariedade que recolhem contribuições para quando é a sua vez. "
            "Estabeleça um limite máximo por evento e comunique com respeito quando não pode contribuir mais. "
            "Não retire da poupança de objectivos (casa, educação) para cobrir eventos sociais — use apenas o fundo dedicado."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "As obrigações sociais (óbitos, alambamentos, baptizados) são uma parte essencial da vida comunitária angolana e representam uma parcela significativa das despesas.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-SAV-047",
        "question": "Como posso investir as minhas poupanças na Bolsa de Dívida e Valores de Angola (BODIVA)?",
        "answer": (
            "A BODIVA é a bolsa de valores de Angola, onde se podem negociar títulos de dívida pública e, futuramente, acções. "
            "Para investir, precisa de abrir conta num intermediário financeiro registado na CMC (Comissão do Mercado de Capitais). "
            "Actualmente, os principais produtos disponíveis são Obrigações do Tesouro e papel comercial de empresas. "
            "Os rendimentos das OT podem ser atractivos: entre 12% e 22% ao ano em Kwanzas. "
            "O investimento mínimo varia conforme a emissão, mas geralmente começa a partir de Kz 100.000. "
            "Informe-se junto do seu banco sobre como aceder à BODIVA — os maiores bancos são intermediários autorizados."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Fluxograma mostrando os passos para investir na BODIVA: abrir conta, escolher intermediário, seleccionar título, fazer a ordem de compra.",
        "audio_transcript": None,
        "cultural_notes": "A BODIVA foi criada em 2014 e ainda está em fase de desenvolvimento, com poucos produtos disponíveis para o público geral.",
        "sources": ["Banco Nacional de Angola", "CMC"],
    },
    {
        "id": "AO-SAV-048",
        "question": "Como posso poupar energia e reduzir custos com electricidade em Angola?",
        "answer": (
            "A electricidade em Angola é cara, especialmente para quem depende de gerador. "
            "Substitua lâmpadas incandescentes por LED — consomem 80% menos energia e duram mais. "
            "Desligue aparelhos da tomada quando não estiver a usar; o consumo em standby acumula-se. "
            "Use o gerador apenas para o essencial: frigorífico, luz e carregar telemóvel — evite ligar ferro e ar condicionado. "
            "Investir num painel solar de Kz 50.000 a Kz 200.000 pode poupar bué em gasóleo a médio prazo. "
            "A poupança com energia pode ser de Kz 10.000 a Kz 30.000 por mês que vão directamente para a conta poupança."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Infográfico comparando o custo mensal de electricidade com gerador a gasóleo vs painel solar ao longo de 24 meses.",
        "audio_transcript": None,
        "cultural_notes": "A energia solar tem crescido em Angola como alternativa ao fornecimento instável da rede pública e ao custo do gasóleo para geradores.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-049",
        "question": "Como posso juntar dinheiro para iniciar uma criação de animais em Angola?",
        "answer": (
            "A criação de animais (galinhas, porcos, cabras) é uma forma popular de investimento e poupança em Angola. "
            "Para começar uma criação de galinhas de quintal, precisa de cerca de Kz 50.000 a Kz 150.000 para o galinheiro e as primeiras aves. "
            "Poupe mensalmente Kz 10.000 a Kz 25.000 e em 6 meses terá capital para começar. "
            "A vantagem é que os animais se reproduzem e geram rendimento: ovos, carne e crias para venda. "
            "É uma forma tangível de poupança que não é afectada directamente pela desvalorização do Kwanza. "
            "Comece pequeno, aprenda o ofício e vá expandindo à medida que tem retorno."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Plano de poupança e investimento para criação de 20 galinhas com custos iniciais, custos mensais e retorno estimado em ovos e carne.",
        "audio_transcript": None,
        "cultural_notes": "A criação de animais de quintal é uma tradição angolana que funciona como reserva de valor e complemento alimentar.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-050",
        "question": "O que acontece às minhas poupanças se o banco falir em Angola?",
        "answer": (
            "Angola tem o Fundo de Garantia de Depósitos (FGD) que protege os depositantes em caso de falência bancária. "
            "O FGD garante depósitos até um determinado limite por titular e por banco. "
            "Para maior segurança, distribua as poupanças entre dois ou três bancos diferentes. "
            "Escolha bancos bem capitalizados e supervisionados pelo BNA — consulte os relatórios públicos de estabilidade financeira. "
            "Os grandes bancos (BAI, BFA, BIC) são geralmente considerados mais seguros por terem maior base de capital. "
            "Em caso de problemas, o BNA intervém antes da falência para proteger os depositantes."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "advanced",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "O BNA revogou licenças de alguns bancos pequenos nos últimos anos, reforçando a importância de escolher bancos sólidos.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-051",
        "question": "Como posso poupar em grupo usando um sistema de caderneta comunitária?",
        "answer": (
            "A caderneta comunitária é um sistema onde cada membro do grupo tem uma caderneta com o registo das suas contribuições. "
            "O grupo reúne-se semanalmente ou mensalmente para depositar as contribuições na presença de todos. "
            "O tesoureiro do grupo guarda o dinheiro em local seguro ou deposita no banco em nome do grupo. "
            "Cada membro pode pedir empréstimo do fundo comum com juros baixos, que revertem para o grupo. "
            "No final do ano, o fundo é distribuído com os lucros dos juros proporcionalmente às contribuições. "
            "A transparência é essencial: todas as movimentações devem ser registadas na caderneta e confirmadas por todos."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Modelo de página de caderneta comunitária com colunas: data, contribuição, empréstimo, pagamento, saldo, assinatura.",
        "audio_transcript": None,
        "cultural_notes": "Os sistemas de poupança comunitária são fundamentais em Angola, especialmente nas zonas rurais sem acesso bancário.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-052",
        "question": "Devo poupar ou pagar dívidas primeiro em Angola?",
        "answer": (
            "A regra geral é: pague primeiro as dívidas com juros altos, porque o custo dos juros é maior que o ganho da poupança. "
            "Se a dívida tem juros de 30% ao ano e a poupança rende 12%, está a perder 18% — pague a dívida. "
            "No entanto, mantenha sempre uma mini-reserva de Kz 20.000 a Kz 50.000 para emergências enquanto paga as dívidas. "
            "Depois de pagar as dívidas, redirecione o valor que pagava de prestação directamente para a poupança. "
            "Não contraia novas dívidas enquanto estiver a pagar as antigas — quebre o ciclo. "
            "Se a dívida for informal (fiado com amigos), pague rapidamente para manter os relacionamentos e a sua reputação."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Árvore de decisão visual: 'Tenho dívida?' -> 'Juros altos?' -> 'Pagar dívida primeiro' vs 'Juros baixos?' -> 'Poupar e pagar em paralelo'.",
        "audio_transcript": None,
        "cultural_notes": "O crédito ao consumo em Angola tem juros muito elevados, tornando o pagamento rápido de dívidas essencial.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-053",
        "question": "Como posso usar a compra de terrenos como forma de poupança em Angola?",
        "answer": (
            "Os terrenos em Angola, especialmente nas zonas periurbanas, têm valorizado consistentemente acima da inflação. "
            "Comprar um terreno é uma forma tangível de poupar que protege contra a desvalorização do Kwanza. "
            "Antes de comprar, verifique a documentação: título de concessão, autorização da administração local e ausência de litígios. "
            "Terrenos nas províncias são mais baratos que em Luanda e podem valorizar muito com o desenvolvimento. "
            "Não compre terreno sem verificar in loco e sem acompanhamento de alguém que conheça a zona. "
            "A desvantagem é que o terreno não é líquido — vender rapidamente pode significar perder valor."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Mapa simplificado de Luanda e arredores mostrando zonas de valorização imobiliária e preços médios por metro quadrado.",
        "audio_transcript": None,
        "cultural_notes": "A questão fundiária em Angola é complexa, com sobreposição entre direito costumeiro e legislação formal, gerando frequentes conflitos de terra.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-054",
        "question": "Que estratégias de poupança existem para mulheres empreendedoras angolanas?",
        "answer": (
            "As mulheres empreendedoras angolanas podem usar a kixikila entre mulheres do mesmo mercado ou bairro. "
            "Muitos bancos e microfinanças oferecem produtos específicos para mulheres com condições especiais. "
            "Separe sempre o lucro do negócio em três: reposição de stock, despesas pessoais e poupança. "
            "Forme grupos de poupança com outras mulheres empreendedoras para partilhar custos e conhecimentos. "
            "Invista os lucros em diversificar o negócio em vez de gastar tudo — compre mais variedade de produtos. "
            "Procure programas de capacitação financeira de organizações como a ADRA, Care Angola ou IRSEA."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text", "image"],
        "image_description": "Infográfico com as diferentes opções de poupança disponíveis para mulheres empreendedoras angolanas: kixikila, microfinanças, conta poupança, reinvestimento.",
        "audio_transcript": None,
        "cultural_notes": "As mulheres angolanas dominam o comércio informal e são peças centrais na economia familiar.",
        "sources": ["Banco Nacional de Angola", "IRSEA"],
    },
    {
        "id": "AO-SAV-055",
        "question": "Como funcionam as instituições de microfinanças para poupança em Angola?",
        "answer": (
            "As instituições de microfinanças (IMF) em Angola oferecem serviços bancários simplificados para quem não tem acesso aos bancos tradicionais. "
            "Muitas IMF permitem abrir conta com valores mínimos mais baixos e menos documentação. "
            "As taxas de juro de poupança podem ser competitivas com as dos bancos comerciais. "
            "As IMF estão regulamentadas pelo BNA, o que dá alguma segurança aos depositantes. "
            "Antes de depositar, verifique se a IMF está licenciada pelo BNA — existem entidades ilegais que se fazem passar por microfinanças. "
            "As IMF são uma boa porta de entrada para o sistema financeiro formal para quem nunca teve conta bancária."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "intermediate",
        "modalities": ["text"],
        "image_description": None,
        "audio_transcript": None,
        "cultural_notes": "O sector de microfinanças em Angola está a crescer mas ainda é pequeno comparado com outros países africanos.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-056",
        "question": "Como posso poupar para abrir um negócio de venda de recargas telefónicas em Angola?",
        "answer": (
            "O negócio de venda de recargas é um dos mais acessíveis em Angola, exigindo pouco capital inicial. "
            "Precisa de Kz 20.000 a Kz 50.000 para o stock inicial de recargas e um telemóvel com Multicaixa Express. "
            "Poupe Kz 5.000 a Kz 10.000 por semana e em 1 a 2 meses terá o capital necessário. "
            "A margem de lucro é pequena (5% a 8%), mas o volume de vendas compensa. "
            "Comece vendendo aos kambas, vizinhos e colegas; depois expanda para um ponto fixo no bairro. "
            "Reinvista os lucros para aumentar o stock e diversificar com venda de dados móveis e pagamento de serviços."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "basic",
        "modalities": ["text", "image"],
        "image_description": "Plano simples de negócio de venda de recargas: investimento inicial, custos fixos, margem por transacção e projecção de lucro mensal.",
        "audio_transcript": None,
        "cultural_notes": "A venda de recargas telefónicas é um negócio muito popular em Angola, especialmente entre jovens, dada a alta penetração de telemóveis.",
        "sources": ["Banco Nacional de Angola"],
    },
    {
        "id": "AO-SAV-057",
        "question": "Quais são os riscos de participar em esquemas de multiplicação de dinheiro em Angola?",
        "answer": (
            "Os esquemas de multiplicação de dinheiro (pirâmides financeiras) prometem rendimentos irreais como 50% a 100% por mês. "
            "Estes esquemas pagam os primeiros investidores com o dinheiro dos novos — não há investimento real por trás. "
            "Quando não entram novos membros suficientes, o esquema colapsa e quem entrou por último perde tudo. "
            "Em Angola, muitas pessoas já perderam bué de bazuka em esquemas como estes, especialmente através de redes sociais. "
            "A regra é simples: se parece bom demais para ser verdade, provavelmente é burla. "
            "Invista apenas em produtos regulados pelo BNA e CMC, e denuncie esquemas suspeitos às autoridades."
        ),
        "country": "AO",
        "topic": "savings",
        "difficulty": "advanced",
        "modalities": ["text", "image"],
        "image_description": "Diagrama de pirâmide financeira mostrando como o dinheiro flui dos novos participantes para os antigos até o colapso inevitável.",
        "audio_transcript": None,
        "cultural_notes": "Esquemas piramidais têm sido recorrentes em Angola, aproveitando a baixa literacia financeira e o desejo de ganhos rápidos.",
        "sources": ["Banco Nacional de Angola", "CMC"],
    },
]

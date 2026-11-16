# Lint dos slides da reunião com pesquisadores (2026-11-15)

Alvo: `slides.template.html` (23 slides). Lentes: estilo (skill no-ai-slop), relevância para 30 minutos, contexto (termos usados antes de explicados), coerência com a dissertação (`Wiki/dissertacao/capitulos-v2`) e gráficos. Cinco revisores independentes e um juiz que conferiu as citações e juntou os repetidos; 12 apontamentos foram descartados.

Total: 56 apontamentos (12 altos, 34 médios, 10 baixos). Nada foi aplicado.

## Conferências feitas depois do juiz

- A referência do TRIDENT confere com a tese: `rigueira2026trident`, Rigueira et al., BRACIS 2025 (anais publicados em 2026), da UFMG.
- MIMIC-wide sob MCAR: o extrato do E176 registra −0,0044 [−0,0056; −0,0032] como "better (under the 1% margin)". Hoje o deck diz "empatam"; antes de reescrever, decidir entre "melhor, por menos de 1%" e "equivalente".
- A "H1c" da tese é a liderança sobre MLP, SAINT e TabNet. O deck não deve nomeá-la; basta dizer o resultado, como já faz.

## Recomendações gerais

- Tempo: as notas marcam cerca de 24,5 minutos, e os slides 3 a 7 (cerca de 25 bullets, sem tempo nas notas) ficam com uns 5 minutos. Para caber em 30 minutos com perguntas, reduzir os slides 6 e 7 à tela mínima e mandar o resto para as notas, juntar os slides 17 e 18 e pôr tempo em todas as notas (slide 3: cerca de 1 min; slides 5 a 7: cerca de 3 min; slide 22: cerca de 1 min).
- Vocabulário fixo, definido uma vez nos slides 5 a 7 e mantido até o fim: o TRIDENT (nunca 'decodificador'), a referência simples, os clássicos fortes, o conjunto forte (clássicos fortes + TabPFN v2), K e H, e melhor/pior/equivalente/sem diferença. Cortar da tela todo nome interno: fork, tickets, barra, tmatch, regra table, sementes da tese, sufixos _20nan e sweep/defaults.
- Ids de experimento: no máximo um por bullet na tela, e só quando sustenta a afirmação; o resto vai para as notas. A plateia não conhece o registro, e listas como (E65, E67, E81) pesam sem informar.
- Oráculo contra implantável: em cada slide de resultado, dizer qual leitura (K ou H) sustenta o número e preferir H onde ela existe. Na MIMIC-IV, dizer que o primeiro desenho pontua células medidas e que o transplante só tem K. É a pergunta mais provável de quem é de fora.
- Enquadramento da classificação: a tese chama o balanço de misto, com H1c sustentada sobre MLP, SAINT e TabNet. 'Insuficiente' (slides 1, 2 e 11) é escolha do autor, mas, se ficar, os slides 11 e 21 devem dizer também a parte sustentada e o 'sem ajuste de nenhum dos lados' do TabPFN, para não parecer comparação desigual.
- Para uma plateia com gente de informática em saúde, a seção da MIMIC-IV é o gancho. Apresentar a MIMIC-IV e o SOFA, levar para a tela o resultado de que preencher melhor não ajudou a prever óbito (E150) e marcar a mimic-wide (E176 a E180) como estudo posterior à dissertação. Pode valer abrir com o exemplo do lactato, que hoje só aparece nas notas dos slides 3 e 8.

## Apontamentos por slide

### Slide 1

- **media** · tela · contexto: origem do modelo
  - Trecho: “Partimos do TRIDENT, um Transformer tabular que representa cada célula faltante por um token aprendido”
  - Problema: O deck nunca diz de onde vem o TRIDENT. Quem é de fora não sabe se o modelo é do aluno ou publicado por outros, e o slide 3 cita "O artigo original" sem dizer qual.
  - Correção: Nas notas do slide 1: "O TRIDENT foi proposto por Rigueira et al. (BRACIS 2025), aqui na UFMG; partimos do código publicado e o estendemos." No slide 3: "O artigo original (Rigueira et al., 2025) avaliou o modelo só sob falta completamente aleatória (MCAR)."

### Slide 2

- **media** · tela · contexto: termos usados antes de definir
  - Trecho: “o TRIDENT vence a referência simples na maioria das variantes e mantém a posição sob MAR e MNAR.”
  - Problema: O resumo usa referência simples, MAR/MNAR, decode por coluna, TabPFN v2 e "contexto do TabPFN" antes de qualquer definição. As siglas só são explicadas no slide 8, que nem define MCAR. A referência simples só aparece no slide 7, e o ouvinte pode supor que é a média, quando é o melhor entre cinco imputadores, uma afirmação mais forte.
  - Correção: Nas notas do slide 2: "A referência simples é o melhor, em cada execução, entre média/moda, kNN, HGB e LightGBM. O TabPFN v2 é um modelo de fundação pré-treinado em tabelas sintéticas, que lê até 10.000 linhas de treino como contexto e prevê sem ajuste. MCAR é a falta ao acaso; no MAR, a falta depende de outras colunas; no MNAR, do próprio valor. Tudo isso volta adiante."

- **media** · tela · causalidade não sustentada / fora da tese
  - Trecho: “na tabela maior que o contexto do TabPFN, vence o TabPFN também.”
  - Problema: A frase sugere que o tamanho da tabela causa a vitória. As notas do slide 20 dizem o contrário: a vantagem "acompanha os padrões reais de lacuna mais do que o tamanho da tabela" (E177), e sob MCAR os dois empatam. A mimic-wide (E176–E180) é posterior à dissertação, que vai até E174. O slide 21 repete a ideia ("numa tabela grande, do TabPFN").
  - Correção: Slide 2: "...e, numa tabela clínica maior (mimic-wide, estudo posterior à dissertação), também o TabPFN v2, uma vantagem que acompanha os padrões reais de lacuna." Slide 21: "...e, com padrões reais numa tabela maior, do TabPFN." No slide 19, marcar a mimic-wide como estudo posterior ao texto.

- **media** · tela · alternância de nomes do modelo
  - Trecho: “Na MIMIC-IV, com lacunas reais, o decodificador vence a referência simples”
  - Problema: O deck alterna "TRIDENT", "o decodificador" (slides 2, 12 e 20, no título e no gráfico) e "decoder" (notas do 15). Na tese, "decodificador" nomeia os decodificadores do MAE e da difusão, e o modelo é sempre "o TRIDENT". O ouvinte pode achar que na MIMIC entrou outro modelo.
  - Correção: Usar sempre "o TRIDENT". No slide 2, "o TRIDENT vence a referência simples"; título do 20, "Com padrões reais, o TRIDENT fica à frente"; legenda do gráfico, "TRIDENT menos cada oponente".

- **media** · notas · promessa adiada / inferência como fato
  - Trecho: “ele divide as variantes com o TRIDENT de um jeito que conseguimos explicar”
  - Problema: O padrão é prometido aqui e de novo na última nota do slide 21 ("segue um padrão que conseguimos explicar") e nunca é dito em nenhum dos dois lugares. A explicação pelo prior é marcada na tese como inferência (05:67, "coerente com"; 07:43, "parece refletir"), e o slide 14 a apresenta como fato.
  - Correção: Notas do 2 e do 21: "Contra o TabPFN v2, o TRIDENT lidera nas tabelas numéricas até 60% de falta; o TabPFN, a 80% e nas duas tabelas mais categóricas, o que é coerente com o prior sintético dele, que pesa quando a linha tem pouco contexto." Slide 14: "A divisão é coerente com o prior do TabPFN, aprendido em tabelas sintéticas...".

### Slide 3

- **baixa** · tela · contexto: token não explicado
  - Trecho: “aberta por [CLS]; categóricas e numéricas têm embeddings próprios.”
  - Problema: O [CLS] é usado aqui e no bullet da classificação sem explicação, e quem não vem de NLP não sabe que a saída dele resume a linha.
  - Correção: Nas notas: "[CLS] é um token posto no início de cada linha; a saída dele no encoder resume a linha inteira e alimenta o classificador."

### Slide 4

- **alta** · tela · correção (tese)
  - Trecho: “a busca de hiperparâmetros nunca superou a configuração padrão (E21, E39).”
  - Problema: A frase contradiz a tese. Em 04-metodologia.tex:352, a busca no perfil completo (E21) superou os padrões em pendigits e letter a 20%, mas não venceu a referência simples em nenhuma das sete variantes. Quem nunca superou os padrões foi a busca refeita (E39). As notas repetem o erro: "nunca venceu a referência simples nem superou a configuração padrão". Dita sem âncora temporal, ainda parece contradizer o slide 13, em que K vence a referência simples em 16 de 21.
  - Correção: Tela: "buscar hiperparâmetros não levou o modelo à frente da referência simples (E21) e, refeita sobre a configuração atual, a busca não superou os padrões (E39)." Fazer a mesma correção nas notas.

- **media** · grafico · rótulos antes da definição / jargão
  - Trecho: “KC: K calibrado (≥40%)”
  - Problema: K, H e KC entram no slide 4 sem definição: as próprias notas adiam K e H para o slide 7, e KC não volta em nenhum slide. A linha de baixo usa "Fork" (jargão de git) e "fine-tuning", enquanto o slide 10 diz "ajuste fino". A caixa "Pré-treino" está em azul, a cor do que estendemos, quando o pré-treino é do artigo e só o alvo normalizado é nosso.
  - Correção: Tirar a caixa "Leituras" (K/KC/H) e a frase da legenda sobre K e H; elas ficam no slide 7. Linha de baixo: "Artigo: pré-treino → ajuste fino do classificador sobre [CLS]. Nossa versão: o mesmo encoder, com decode no lugar do classificador." Tirar o azul da caixa do pré-treino e escrever nela "alvo normalizado (nosso)".

- **media** · tela · contexto: argumento de rigor sem definição
  - Trecho: “A configuração atual saiu de estudos pré-registrados”
  - Problema: "Pré-registrado" é o argumento de rigor do deck (slides 4, 12 e 22), mas o deck nunca diz o que se fixou antes nem onde. Quem é de informática em saúde associa o termo a registros externos de ensaios clínicos.
  - Correção: Nas notas: "Pré-registrar, aqui, é fixar a hipótese, os braços, as variantes, as sementes e a regra de decisão numa entrada do nosso registro de experimentos antes da primeira execução."

### Slide 5

- **media** · tela · contexto: unidade 'par' e rótulos de conjunto
  - Trecho: “Cada comparação: cinco dobras × três sementes = quinze pares.”
  - Problema: O slide não diz o que é um par. No slide 15, "7 pares" e "9 de 10 pares" contam casos de tabela e taxa, outra unidade. O slide também não chama as 21 variantes de "desenvolvimento", termo que os slides 13, 14 e 17 usam, nem explica o nome "biodeg_20nan" que aparece nos gráficos.
  - Correção: "Cada comparação: cinco dobras × três sementes = quinze pares (TRIDENT e oponente na mesma dobra e semente)." E: "21 variantes de desenvolvimento, para as decisões de desenho; 15 de confirmação, lidas uma única vez."

- **media** · grafico · tabela incompleta / inglês
  - Trecho: “kr-vs-kp3196360—”
  - Problema: A coluna "tarefa (classes)" deixa "—" em kr-vs-kp, kc2 e electricity, o que sugere que essas tabelas não têm rótulo, mas a tab:dados da tese (04-metodologia.tex:192-200) dá 2 classes para as três. As outras células estão em inglês ("Binary", "four classes").
  - Correção: Coluna "classes" com números, como na tese: credit-g 2, kr-vs-kp 2, spambase 2, vehicle 4, kc2 2, biodeg 2, pendigits 10, letter 26, electricity 2.

- **baixa** · tela · calque no título
  - Trecho: “Nove tabelas, falta cortada com a verdade conhecida”
  - Problema: "Falta cortada" não é termo da área; a tese diz "falta injetada". As notas do 19 repetem o termo ("cortadas por um mecanismo").
  - Correção: Título: "Nove tabelas completas, com falta injetada". Notas do 19: "injetadas por um mecanismo que nós escrevemos". Nas notas do 5: "tabelas públicas do OpenML".

### Slide 6

- **baixa** · tela · exagero + sigla ambígua + métrica não definida
  - Trecho: “Painel secundário (MAE, viés, dispersão, Wasserstein) confere cada afirmação.”
  - Problema: Um painel não confere nada, e "cada afirmação" promete mais do que o deck mostra. "MAE" pode ser lido como masked autoencoder, família do ReMasker. No mesmo slide, "Na classificação, macro-F1." fica sem definição, e ela importa para o balanceamento por classe do slide 11.
  - Correção: "Um painel secundário (erro absoluto, viés, razão de dispersão, distância de Wasserstein) mostra se o ganho vem de encolher os preenchimentos para o centro." E: "Na classificação, macro-F1: a média do F1 de cada classe, sem pesar pelo tamanho da classe."

### Slide 7

- **alta** · tela · contexto: oráculo sem explicação
  - Trecho: “K: época escolhida pela verdade das lacunas de validação. Um oráculo; dá o teto.”
  - Problema: Escolher a época na validação é prática padrão, e o ouvinte não entende por que isso seria um oráculo. Falta dizer que K usa os valores verdadeiros das células apagadas, que só conhecemos porque nós mesmos as apagamos, enquanto H usa só a perda nas células observadas. Essa distinção sustenta metade dos números do deck e o primeiro limite do slide 22.
  - Correção: "K: época escolhida pelos valores verdadeiros das lacunas de validação, que só conhecemos porque nós as apagamos e que não existem em dados reais. Um oráculo; dá o teto." / "H: época escolhida pela perda de reconstrução nas células observadas da validação. É a leitura implantável."

- **media** · tela · contexto: oponente principal mal apresentado
  - Trecho: “TabPFN v2, à parte: modelo de fundação tabular, coluna a coluna, sem ajuste.”
  - Problema: O oponente mais forte do deck é apresentado em seis palavras. O ouvinte não sabe que ele foi pré-treinado em tabelas sintéticas, que lê as linhas de treino como contexto (até 10.000) nem que imputa tomando cada coluna como alvo das demais. O prior sintético do slide 14 e o limite de contexto dos slides 2 e 19 dependem disso.
  - Correção: Tela: "TabPFN v2, à parte: modelo de fundação pré-treinado em tabelas sintéticas; lê até 10.000 linhas de treino como contexto e prevê cada coluna a partir das demais, sem ajuste." Nas notas, a frase do cap. 5: "no nosso protocolo, ele prevê cada coluna a partir das demais, com as linhas de treino como contexto".

- **media** · tela · contexto: métodos só pelo nome
  - Trecho: “Imputadores clássicos fortes: missForest, condexp, SoftImpute e ICE.”
  - Problema: São quatro métodos só pelo nome, e as notas explicam apenas o condexp. HGB, no bullet da referência simples, também não é expandido. A plateia não vê por que esses formam o degrau "forte".
  - Correção: Nas notas: "missForest imputa com florestas aleatórias, coluna a coluna, em rodadas; condexp usa a esperança condicional gaussiana dados os observados da linha; SoftImpute completa a matriz supondo posto baixo; ICE são rodadas de regressões encadeadas, no espírito do MICE. HGB é o gradient boosting por histograma do scikit-learn." A tela fica como está.

### Slide 8

- **media** · tela · relevância + definição ausente
  - Trecho: “MNAR: a chance de sumir cresce com o próprio valor, Φ(inclinação · z); padrão 0,5, de 0,25 a 2,0 em biodeg a 60%.”
  - Problema: Este é o único slide que poderia definir as siglas usadas desde o slide 2, e MCAR nem tem bullet. Em vez disso, a tela traz fórmulas (Φ, sigmoid(1,5 · s)), a faixa de inclinação, três desvios e o desenho do contraste de interação, que só será usado sete slides depois. Em 2 minutos, isso não fica na cabeça.
  - Correção: Bullets: "MCAR: toda célula tem a mesma chance de sumir." / "MAR: a chance depende de outras colunas; 30% delas nunca perdem células e decidem quais somem nas outras." / "MNAR: a chance cresce com o próprio valor escondido; a inclinação regula a força (padrão 0,5)." / "Sob MNAR, as células escondidas ficam 0,358 desvio-padrão acima das observadas, contra 0,008 sob MCAR." Fórmulas, faixa de 0,25 a 2,0, o 0,788 e o bullet da interação vão para as notas (este último também para as notas do slide 15).

- **baixa** · grafico · termo interno / eixo sem título
  - Trecho: “z (valor padronizado da célula) ou s (escore das motoras)”
  - Problema: "Motoras" é jargão do gerador e não aparece no texto. O eixo y (0,0 / 0,5 / 1,0) não tem título. A legenda repete os rótulos da ponta das curvas e não diz que o número é a inclinação. O bullet "Funciona:" é um colon reveal que não diz qual mecanismo.
  - Correção: Eixo x: "z (valor padronizado da célula, MNAR) ou s (escore das colunas que decidem a falta, MAR)". Eixo y: "chance de a célula ser sorteada". Rótulos: "MNAR, inclinação 0,5 (padrão)" etc.; tirar a legenda de cima. Bullet: "Sob MNAR, as células escondidas ficam 0,358 desvio-padrão acima das observadas...".

### Slide 9

- **alta** · grafico · clareza: sinal não definido e invertido
  - Trecho: “B − A, macro-F1 (negativo: o pré-treino ajuda)”
  - Problema: A e B não aparecem em nenhum lugar da tela. A legenda diz "Sem menos com pré-treino", outra notação. No slide 10, o "+0,010" é positivo a favor do pré-treino, de modo que o sinal favorável troca entre dois slides seguidos. A tese (06-estudos-de-caso.tex:19) usa A − B, com menos sem, e escreve o artigo como +0,017.
  - Correção: Sem trocar nenhum valor, escrever o contraste por extenso. Eixo: "sem pré-treino − com pré-treino, macro-F1 (negativo: o pré-treino ajuda)". Legenda: "Diferença sem − com pré-treino, IC 95% sobre 450 pares; abaixo de zero, o pré-treino ajuda. A faixa é a margem de ±0,01; a linha, o ganho do artigo (1,7 ponto)." No slide 10: "o pré-treino acrescenta +0,010 (com − sem)". Inverter a figura inteira para a convenção da tese fica a critério do autor.

- **media** · grafico · jargão interno / inglês
  - Trecho: “E87, sweep do fork (300/150 épocas)E118, defaults do código original (40/40)”
  - Problema: "sweep", "fork" e "defaults" são jargão interno, e "300/150" e "40/40" não dizem o que contam. O bullet chama o mesmo braço de "Configuração antiga do fork", e o eixo usa os sufixos de arquivo "00nan…80nan", enquanto o texto fala em níveis de falta.
  - Correção: Legenda: "nossa configuração anterior (300 épocas de pré-treino, 150 de ajuste fino; E87)" e "padrões do código original (40 épocas em cada fase; E118)", conforme a tese (06:31). Eixo: "0% | 20% | 40% | 60% | 80%". No bullet: "Nossa configuração anterior: ...".

- **baixa** · notas · tom defensivo + vocabulário da tese
  - Trecho: “Isso não é uma replicação fracassada: os hiperparâmetros ajustados por tabela no artigo não estão no repositório.”
  - Problema: Dita em voz alta, abrir negando uma acusação que ninguém fez põe o ouvinte na defensiva, e o fato basta sozinho. Mais adiante, a mesma nota usa "rodar", e o guia da tese pede "executar".
  - Correção: "Os hiperparâmetros ajustados por tabela no artigo não estão no repositório; medimos as duas configurações que o código permite executar: 300 épocas de pré-treino na nossa versão e 40 em cada etapa no original."

### Slide 10

- **media** · tela · ressalva da tese omitida
  - Trecho: “Única brecha: com um quarto dos rótulos, o pré-treino acrescenta +0,010, com 8 variantes melhores e nenhuma pior (E141).”
  - Problema: A tese acompanha esse +0,010 de uma ressalva: o intervalo de Nadeau–Bengio cruza o zero (07-conclusoes.tex:49). O slide a omite. A frase da caixa ("perder três quartos dos rótulos custa muito mais que esse ganho") não tem fonte na tese. O rótulo "Única brecha:" repete o colon reveal do título ("Por quê:").
  - Correção: "Com um quarto dos rótulos, o pré-treino acrescenta +0,010 (com − sem), com 8 variantes melhores e nenhuma pior (E141); com a correção de Nadeau–Bengio, o intervalo cruza o zero." Cortar a frase sobre o custo dos rótulos ou levá-la às notas com a fonte. Título: "O ajuste fino apaga o efeito do pré-treino".

- **baixa** · tela · inglês + termos truncados
  - Trecho: “Taxas menores, congelamento inicial, orçamentos e schedules não recuperaram o efeito.”
  - Problema: "schedules" fica em inglês, e a tese diz "agendamento da taxa de aprendizado". "Taxas" e "orçamentos" ficam sem complemento.
  - Correção: "Taxas de aprendizado menores, encoder congelado no início, outros números de épocas e outros agendamentos da taxa não recuperaram o efeito."

### Slide 11

- **alta** · tela · contexto + correção (tese)
  - Trecho: “O token [NULL] lê a falta informativa com o sinal certo, mas o teste de que a aproveita tão bem quanto imputar mais indicador ficou inconclusivo (E128, E155).”
  - Problema: O item junta conceitos que o deck nunca definiu: falta informativa, "com o sinal certo" e "imputar mais indicador". O E128 não é um teste inconclusivo. É a falha do portão de validade que parou o protocolo (07-conclusoes.tex:27). O E155 é o teste inconclusivo, e a estimativa pende contra o [NULL] (06:117). Ninguém de fora consegue reconstruir o que foi testado.
  - Correção: "O teste de se o [NULL] aproveita a falta informativa (quando faltar já diz algo sobre o rótulo) tão bem quanto preencher a célula e acrescentar uma coluna 0/1 que marca a falta ficou sem resposta, com a estimativa pendendo contra o [NULL] (E155)." Nas notas: "Antes, o protocolo parou duas vezes numa checagem prévia, fixada no pré-registro, de que o desenho detectava o efeito (E128)."

- **media** · tela · sinal e simetria do ajuste (tese)
  - Trecho: “TabPFN v2, sem ajuste: à frente em todos os níveis, por +0,018 a +0,037 de macro-F1 com probabilidades balanceadas por classe (E92).”
  - Problema: O sinal positivo não diz qual termo vem primeiro, e no resto do deck a diferença é TRIDENT − oponente. A expressão "sem ajuste" sugere um TRIDENT ajustado contra um TabPFN cru, mas a tese diz "sem ajuste de nenhum dos lados", com o TRIDENT no orçamento padrão (06:43). O slide 2 repete o problema.
  - Correção: "TabPFN v2, sem ajuste de nenhum dos lados: à frente em todos os níveis; TabPFN − TRIDENT de +0,018 a +0,037 de macro-F1, com probabilidades balanceadas por classe (E92)." Slide 2: "o TabPFN v2, sem ajuste de nenhum dos lados, fica à frente...".

- **media** · tela · clareza: aparente contradição
  - Trecho: “o TRIDENT lidera de 40% a 80%, mas sobre o MLP por cerca de um ponto, empata a 80% e perde na tabela completa (E99, E107).”
  - Problema: Em voz alta, "lidera de 40% a 80%" seguido de "empata a 80%" soa contraditório, e não fica claro que o empate é só com o MLP. "Empata" sugere equivalência, mas a tese diz "sem diferença" (07:49), e o slide 6 separa as duas coisas. SAINT e TabNet não são apresentados.
  - Correção: "MLP, SAINT e TabNet, com o mesmo orçamento de ajuste: de 40% a 80%, o TRIDENT lidera os três; sobre o MLP, só por cerca de um ponto e sem diferença a 80%. Na tabela completa, fica atrás (E99)." Nas notas: E107 e "SAINT é um Transformer tabular com atenção entre colunas e entre linhas; TabNet, uma rede que escolhe atributos por atenção sequencial".

- **media** · tela · recapitulação + calque + enquadramento
  - Trecho: “Para classificar, o TRIDENT é insuficiente: com todos os rótulos, o pré-treino não paga”
  - Problema: A caixa repete os três bullets acima, e o slide 21 repete tudo de novo. "Não paga" é calque de "pays off" (o slide 15 repete: "não pagaram"). A tese chama o balanço da classificação de "misto", com H1c sustentada (06:47; 07:49), e a caixa omite a parte sustentada.
  - Correção: Cortar a caixa, ou deixar uma frase: "Para classificar, a posição é modesta: o pré-treino não ajuda com todos os rótulos e o TabPFN v2 fica à frente; sobre MLP, SAINT e TabNet, o TRIDENT lidera, por pouco sobre o MLP." No 15, trocar "não pagaram" por "não trouxeram ganho".

### Slide 12

- **alta** · tela · consistência de oponentes
  - Trecho: “o primeiro decodificador só errava menos que o melhor imputador clássico em 6 das 21 variantes (E19)”
  - Problema: O oponente do E19 era a referência simples da época (média/moda, kNN, HGB). O slide 7 reserva o rótulo "clássicos fortes" para missForest e os demais, e "o melhor imputador clássico" faz o ouvinte pensar neles. O ponto de partida parece pior do que foi, e a comparação com o "13 de 21 contra clássicos fortes" do slide 13 fica errada. "Decodificador" é, além disso, um terceiro nome para o TRIDENT.
  - Correção: "Ponto de partida: o primeiro TRIDENT de imputação só errava menos que a referência simples em 6 das 21 variantes (E19)."

- **media** · tela · terceiro rótulo de leitura + sementes
  - Trecho: “0,7416 (saída padrão, oráculo)”
  - Problema: "Saída padrão" é um terceiro nome, ao lado de K e H. Na tese, é K a 20% e KC (K calibrado) de 40% para cima (04:170), e KC nunca é explicado no deck. A trajetória usa as sementes 42/7/13 e é "o caminho até a calibração, e não a configuração de hoje" (04:354, 364), mas o slide 13 diz que o desenvolvimento usa 101/202/303. O bullet "Época pelas lacunas de validação" não diz que essa é a leitura K.
  - Correção: Legenda: "impute_score médio nas 21 variantes (sementes 42/7/13): de 0,7929 para 0,7542 (H, implantável) e 0,7416 (K calibrado, oráculo)." Bullet: "Época escolhida pelas lacunas de validação (a leitura K): melhorou 3, todas a 80%." Nas notas: "KC é K com uma calibração lida nas mesmas lacunas de validação; a figura é o caminho até a calibração, e o encoder maior e o lote grande vieram depois."

- **baixa** · tela · conceito não definido
  - Trecho: “Alvo de pré-treino normalizado com decode longo: melhorou 9 das 21.”
  - Problema: O alvo normalizado aparece aqui, no diagrama do slide 4 e nas notas do 10, sem dizer o que se normaliza, e "decode longo" não tem duração. O título ("Três mudanças de método mudaram o quadro") repete a palavra e não diz o que mudou.
  - Correção: Nas notas: "Alvo normalizado: o embedding que o pré-treino prevê passa por uma normalização de camada; decode longo: 450 épocas." Título: "Três mudanças de método baixaram o impute_score médio de 0,7929 para 0,7542 (H)".

### Slide 13

- **alta** · tela · clareza: sujeito ambíguo, números fora do gráfico
  - Trecho: “Ajustada: melhora em 17 de 21 variantes, mas move poucos vereditos; a regra da confirmação deu misto, 9 vitórias e 3 derrotas (E44).”
  - Problema: O ouvinte entende que é o TRIDENT que melhora, mas quem melhora é a referência ajustada em relação à sem ajuste (05-imputacao.tex:41). O 9/3 vem da regra primária da confirmação, escrita sobre K e exigindo vencer as duas referências simples (05:45). Não é a linha "K − ref. simples ajustada" (13 | 5 | 3) do gráfico ao lado. Nenhum número do bullet aparece no gráfico, e "misto" não está entre as categorias do slide 6.
  - Correção: Dividir em dois bullets. "Referência simples ajustada: K vence 13 de 21 e perde 3." e "A regra fixada para a confirmação, que exige vencer as duas referências simples, deu resultado misto: 9 vitórias e 3 derrotas em 15 (E44)." Nas notas: "O ajuste melhora a própria referência em 17 de 21 variantes, mas muda poucos vereditos."

- **media** · grafico · leituras misturadas / linhas fora do título
  - Trecho: “confirmação · H − TabPFN v2 | 5 | 5 | 5 | n = 15”
  - Problema: As linhas alternam K (oráculo) e H (implantável) sem aviso, e quem compara linhas mistura um teto com a leitura implantável. As duas linhas do TabPFN (9/6/6 e 5/5/5) não sustentam o título "vence na maioria" e só são discutidas no slide seguinte, que não tem figura.
  - Correção: Legenda: "Variantes por veredito, sob MCAR. K = época pelo oráculo (teto); H = época implantável; onde só há K, a contagem é um teto. Desenvolvimento: 21 variantes; confirmação: 15, lidas uma vez." Separar visualmente as duas linhas do TabPFN, com uma nota do tipo "→ próximo slide", ou movê-las para o slide 14. As sementes vão para as notas.

- **media** · grafico · categoria de veredito com outro nome
  - Trecho: “melhorsem vereditoequivalentepior”
  - Problema: O slide 6 e o glossário da tese (glossario.tex:43) usam "sem diferença". As legendas dos slides 13 e 18 ("equivalente ou sem veredito", "sem veredito 6") usam "sem veredito", que soa como uma quarta categoria ou como uma comparação não lida.
  - Correção: Trocar "sem veredito" por "sem diferença" nas legendas e títulos de painel dos slides 13 e 18: "melhor · sem diferença · equivalente · pior"; "híbrido − H: melhor 19 · equivalente 1 · sem diferença 6 · pior 0"; "equivalente ou sem diferença".

### Slide 14

- **media** · tela · contradição interna + título vago
  - Trecho: “TabPFN lidera: a 80% e, a partir de 40%, em kr-vs-kp e credit-g.”
  - Problema: O bullet anterior diz que o TRIDENT lidera em spambase em todas as taxas e em kc2 a 80%, e em voz alta "TabPFN lidera a 80%" contradiz o mesmo slide. As duas listas de tabelas e taxas são minúcia para ler em voz alta e escondem o padrão. O título ("um quadro dividido, mas não aleatório") anuncia um padrão sem dizer qual.
  - Correção: Título: "Contra o TabPFN v2, o TRIDENT lidera nas numéricas até 60%; o TabPFN, a 80% e nas mais categóricas". Um bullet só: "O TRIDENT lidera nas tabelas numéricas até 60% de falta; o TabPFN, a 80% (exceto spambase e kc2) e, a partir de 40%, nas duas mais categóricas, kr-vs-kp e credit-g." A lista detalhada vai para as notas.

- **media** · tela · palavra diferente da categoria
  - Trecho: “Contra ele, H vence, empata e perde 5 cada na confirmação”
  - Problema: No gráfico do slide 13, 5 | 5 | 5 é melhor / sem diferença / pior. "Empata" sugere "equivalente", que o slide 6 define como outra categoria.
  - Correção: "Contra ele, na confirmação, H é melhor em 5, sem diferença em 5 e pior em 5; no desenvolvimento, K é melhor em 9 e pior em 6."

- **media** · tela · modelos não apresentados + ressalvas omitidas
  - Trecho: “MaCoDE: melhor em 20 de 21 e 14 de 15; ReMasker: melhor nas 12 numéricas (E48, E159).”
  - Problema: Os dois modelos aparecem pela primeira vez aqui, só pelo nome, e não fica claro quem é melhor. A tese ressalva que os dois foram executados sem ajuste, enquanto o TRIDENT foi ajustado no desenvolvimento (05:77), e que contra o ReMasker só há a leitura K, nas 12 variantes numéricas da confirmação (05:79). O slide já tem seis bullets e uma caixa.
  - Correção: Cortar da tela. Nas notas: "Também comparamos com dois imputadores profundos recentes, sem ajuste, nas configurações dos seus artigos: o MaCoDE, um Transformer mascarado sobre valores discretizados em faixas, perde para o TRIDENT em 20 de 21 e 14 de 15 variantes; o ReMasker, um autoencoder mascarado, perde nas 12 numéricas da confirmação, na leitura K (E48, E159)."

### Slide 15

- **alta** · tela · contexto: termos e estudo pressupostos
  - Trecho: “A família completa refutou a perda de terreno para o conjunto forte: piora só em biodeg a 60%, um caso em doze (E64).”
  - Problema: "Família completa" e "conjunto forte" nunca foram definidos. Na tese, conjunto forte são os clássicos fortes mais o TabPFN v2 (glossario.tex:26), e o slide 7 trata os dois separados. "Refutou a perda de terreno" pressupõe um estudo anterior que só aparece nas notas, e a dupla negação obriga a decifrar a frase.
  - Correção: "Contra os clássicos fortes e o TabPFN v2, o TRIDENT não perde terreno do MCAR para o MAR em seis tabelas: a vantagem encolhe só em biodeg a 60%, um caso em doze (E64)." Nas notas: "Um primeiro estudo, em quatro tabelas, sugeria perda de terreno; com seis tabelas e sementes novas, isso não se sustentou."

- **media** · tela · unidade trocada + verbo vago
  - Trecho: “H vence os 7 pares da inclinação padrão; sem diferença na 1,0 e inverte na 2,0 (E46).”
  - Problema: No slide 5, "pares" são dobra × semente. Aqui, "7 pares" e "9 de 10 pares" são casos de tabela e taxa. "Inverte" não diz o que se inverte, e na tese é uma derrota para a referência simples (05:139).
  - Correção: "Contra a referência simples: H vence nos 7 casos de tabela e taxa com a inclinação padrão; não há diferença na inclinação 1,0, e H perde na 2,0 (E46)." No primeiro bullet, trocar "em 9 de 10 pares" por "em 9 de 10 casos".

- **media** · tela · correção (tese): rótulo e veredito
  - Trecho: “Três tentativas de explorar o mecanismo não pagaram (E65, E67, E81).”
  - Problema: A frase está sob MAR, mas só o E65 é MAR; o E67 e o E81 são estudos sob MNAR (06:59, 65). O E81 teve veredito misto e é, na tese, "a primeira evidência de que o [NULL] carrega a informação que o TRIDENT lhe atribui" (06:75), ao custo de acurácia em 11 de 11. O deck apaga o principal achado positivo sobre o token. "Não pagaram" é calque.
  - Correção: Cortar da tela. Nas notas, separar: "Sob MAR, máscaras de decode por propensão não melhoraram (E65). Sob MNAR, o [NULL] sozinho nas lacunas do decode foi pior (E67); um decoder que distingue a lacuna real da própria máscara lê o padrão de lacunas sob MNAR moderado, mas custa acurácia em 11 de 11 variantes (E81)." Se houver tempo, dizer o E81 em voz alta no slide 11, onde se fala do [NULL].

- **media** · tela · abstração sem referente
  - Trecho: “O custo aparece contra a referência simples, a 60%, sem ficar atrás dela em nenhum caso.”
  - Problema: "O custo" não tem referente na tela. O fato, que a vantagem encolhe nas três tabelas numéricas pequenas, está só nas notas.
  - Correção: "Contra a referência simples, a vantagem encolhe a 60% nas três tabelas numéricas pequenas, mas H nunca fica atrás dela."

- **baixa** · tela · ressalva da tese omitida
  - Trecho: “Contra o TabPFN v2: H vence 6 de 10 e não perde nenhum (E58).”
  - Problema: A tese apresenta o número com uma ressalva: os pares estão a 20% e 60%, onde o TRIDENT já empatava ou vencia sob MCAR (05:139). Sem ela, o ouvinte entende uma vantagem criada pelo MNAR.
  - Correção: "Contra o TabPFN v2: H é melhor em 6 de 10 e nunca pior, em taxas (20% e 60%) onde já não perdia sob MCAR (E58)."

### Slide 16

- **alta** · tela · clareza: termo ambíguo
  - Trecho: “a média simples foi melhor que ambos em 10 de 15”
  - Problema: Desde o slide 6, "média" é preencher com a média da coluna, o 1,0 do impute_score. Aqui é a média dos dois preenchimentos, mas o ouvinte vai entender que preencher com a média venceu os dois modelos.
  - Correção: "TRIDENT e TabPFN erram em lugares diferentes: a média meio a meio dos dois preenchimentos foi melhor que cada um em 10 de 15, mas só em 3 de 9 numa replicação (E60, E66)."

- **media** · tela · nome de código + conceito não definido
  - Trecho: “A regra table”
  - Problema: "table" é o nome da regra no código e não diz nada ao ouvinte; a tese a chama de "híbrido implantável" (glossario.tex:25). "Nada nela lê a tabela completa" é obscuro: o que se quer dizer é que a regra não usa a verdade das lacunas. "Variantes difíceis" não é definida aqui, e o slide 17 a define de outro modo. A tese a define como "aquelas em que a mistura igual não superava o TabPFN" (05:89).
  - Correção: Subtítulo: "O híbrido implantável: um peso por tabela". Último bullet: "Não usa a verdade das lacunas; por isso é implantável." Bullet do E82: "...pior que o TabPFN em 5 de 10 variantes difíceis, aquelas em que a média meio a meio não superava o TabPFN (E82)." No slide 17, usar a mesma definição.

### Slide 17

- **media** · tela · jargão + tela duplicada nas notas
  - Trecho: “Sementes da tese, 26 variantes: nunca pior que H”
  - Problema: "Sementes da tese" contra "sementes novas" é distinção do registro; o slide 13 chamou 101/202/303 de sementes de desenvolvimento. As duas caixas da tela repetem o bullet de limites e são copiadas palavra por palavra nas notas, e o apresentador acaba lendo o slide. As notas também usam "auto-mascaramento", que o deck nunca definiu.
  - Correção: "Sementes do desenvolvimento (101/202/303), 26 variantes: nunca pior que H...". Cortar as duas caixas da tela. Nas notas, deixar só o que acrescenta: "Sob MNAR na inclinação 2,0, o híbrido perde para H em dois conjuntos de sementes."

### Slide 18

- **media** · tela · slide sem mensagem + rótulos internos
  - Trecho: “7 · O híbrido com o TabPFN v2 · E174”
  - Problema: O slide não tem título, só o eyebrow com um id interno, e a mensagem da figura não está escrita. Os 26 rótulos de linha usam o sufixo "_20nan", e a legenda repete "sementes da tese". É um terceiro slide para uma seção de 3 minutos.
  - Correção: Juntar ao slide 17, ou acrescentar o título "Nunca pior que H; pior que o TabPFN só em credit-g a 80%" e tirar o E174 do eyebrow. Rótulos: "biodeg 20%", "credit-g 80%" etc. Legenda: "26 variantes MCAR do desenvolvimento: híbrido menos H (esq.) e menos TabPFN v2 (dir.), IC 95% sobre quinze pares; negativo favorece o híbrido."

### Slide 19

- **alta** · tela · correção (tese): o que é pontuado
  - Trecho: “Lacunas naturais: ficam como entrada; escondemos células medidas para pontuar.”
  - Problema: A tese chama esse desenho de "células medidas escondidas" (05-imputacao.tex:178; glossario.tex:10) e avisa que ele "diz que o modelo aprendeu a estrutura das linhas medidas, e não que preenche bem as lacunas reais" (05:194). O rótulo "Lacunas naturais", repetido no slide 20 ("_natural", "naturais"), faz o ouvinte crer que o escore mede as lacunas reais.
  - Correção: "Células medidas escondidas: as lacunas naturais ficam como entrada, e pontuamos células medidas que escondemos." Usar "células medidas" no slide 20 (bullets e rótulos do gráfico).

- **media** · tela · contexto clínico + ressalvas da tese
  - Trecho: “em duas visões: sinais vitais e exames (22 atributos) e componentes do SOFA (17).”
  - Problema: Nem a MIMIC-IV nem o SOFA são apresentados, e a sigla SOFA não é expandida. Os 22 atributos incluem vasopressores, idade, sexo e tipo de admissão. O deck chama a mesma tabela de "sinais vitais", "de sinais vitais" e "mimic-cardio". As notas dizem que o transplante "preserva a geometria real da falta", mas omitem as duas ressalvas da tese (05:198): os doadores são mais graves (mortalidade de 33,5% e 25,4%, contra 16,5%), e o transplante é, no máximo, MAR.
  - Correção: Tela: "...em duas tabelas: cardio (sinais vitais, exames e vasopressores; 22 atributos) e os componentes do escore de disfunção orgânica SOFA (17)." Nas notas: "A MIMIC-IV é uma base pública de prontuários de UTI de um hospital de Boston. O transplante mantém quais exames faltam juntos num paciente real, e conhecemos o valor de cada lacuna; mas os doadores são mais graves que a coorte, e o transplante é, no máximo, MAR."

### Slide 20

- **alta** · tela · correção (tese) + unidades misturadas
  - Trecho: “Referência simples: melhor por cerca de 1,5% do escore dela nas lacunas naturais, e por −0,0455 e −0,0272 com transplante (E148, E149).”
  - Problema: (1) O 1,5% é das células medidas escondidas, não das lacunas naturais (05:190). (2) Os números do transplante só existem na leitura K, de oráculo (05:196; 07-conclusoes.tex:45), e o deck disse no slide 7 que H é a leitura para dados reais. (3) O bullet põe uma porcentagem relativa ao lado de diferenças absolutas de quatro casas, sem dizer de que tabela é cada uma.
  - Correção: "Referência simples: melhor por cerca de 1,5% do escore dela nas células medidas escondidas (E148) e por cerca de 5% e 3% com padrões transplantados, só na leitura K, de oráculo (E149)." As diferenças −0,0455 (sinais vitais) e −0,0272 (SOFA) vão para as notas.

- **alta** · tela · clareza: sujeito ambíguo + escopo
  - Trecho: “TabPFN, coorte: à frente na de sinais vitais, atrás na do SOFA; ali o híbrido fecha a distância (E163, E166).”
  - Problema: Pelo rótulo, quem fica à frente parece ser o TabPFN, mas é o TRIDENT. A frase só vale para o transplante. Nas células medidas, o TRIDENT fica à frente nas duas tabelas, mas só no erro quadrático, e a tese não tira daí afirmação contra o TabPFN (05:192). Nas tabelas da coorte, a comparação é descritiva. Sobre o híbrido, a tese registra só "nunca pior que H", e pior que o TabPFN na variante permutada do SOFA (B-experimentos.tex:101). O E166 não está na tese.
  - Correção: "Contra o TabPFN, na coorte e de forma descritiva: com transplante, o TRIDENT fica à frente nos sinais vitais e atrás no SOFA; ali o híbrido nunca fica pior que H (E163, E166)." Manter "fecha a distância" só se o registro do E163/E166 mostrar o híbrido no nível do TabPFN na variante principal.

- **alta** · grafico · jargão interno no gráfico
  - Trecho: “decodificador − barradecodificador − TabPFN v2”
  - Problema: "barra" é o apelido interno da referência simples e não aparece em nenhum outro lugar do deck. Os rótulos das linhas são nomes de arquivo ("mimic-cardio_natural", "tmatch", "mcar") que não batem com os do slide 19 ("sinais vitais e exames", "Padrões transplantados"). O ouvinte não sabe quem é o oponente nem qual linha é qual.
  - Correção: Legenda: "TRIDENT − referência simples" e "TRIDENT − TabPFN v2". Rótulos: "sinais vitais · células medidas", "SOFA · células medidas", "sinais vitais · transplante", "SOFA · transplante", "mimic-wide · células medidas", "mimic-wide · transplante", "mimic-wide · MCAR (controle)". Os ids E saem dos rótulos e vão para a legenda da figura.

- **media** · grafico · gráfico desenha descritivo como testado
  - Trecho: “Decodificador menos cada oponente (sementes 42/7/13; negativo favorece o TRIDENT). Só se comparam contrastes dentro de uma linha.”
  - Problema: Na coorte, os contrastes contra o TabPFN são descritivos, mas aparecem desenhados iguais aos testados; só as notas dizem isso. O bullet diz que os dois "empatam sob MCAR" na mimic-wide, mas o ponto (−0,0044) tem um IC que exclui o zero, e pela regra do slide 6 o ouvinte leria "melhor". A legenda alterna "Decodificador" e "TRIDENT".
  - Correção: Legenda: "TRIDENT menos cada oponente, IC 95%; negativo favorece o TRIDENT. Contra o TabPFN, só as linhas da mimic-wide têm teste; nas da coorte, a comparação é descritiva. Sob MCAR, a diferença para o TabPFN (−0,0044) fica dentro da margem de 1%: lemos como equivalente." Conferir no E176 que o critério foi de fato a margem. As sementes vão para as notas.

- **media** · notas · relevância para a plateia
  - Trecho: “Preencher melhor não ajudou a prever óbito”
  - Problema: Para pesquisadores de informática em saúde, este talvez seja o resultado mais útil da MIMIC-IV (E150), e a tese o destaca na conclusão (07:45). Ele está no fim de uma nota longa, depois de um experimento em andamento (E180). A nota também diz que E177 está "em andamento", mas os 84%/82% citados antes são dele, já concluído.
  - Correção: Levar para a tela, no slide 20 ou no 22: "Preencher melhor não ajudou a prever óbito: numa regressão logística, só os indicadores de falta ajudaram, na tabela do SOFA (E150)." Nas notas: "o E180, em andamento, separa o efeito das linhas do efeito dos passos de otimização". O E177 sai do parêntese de "em andamento".

### Slide 21

- **media** · tela · conclusão incompleta / ambígua
  - Trecho: “Imputação: com um decode por coluna, é competitivo; vence os clássicos na maioria das variantes e mantém a posição sob MAR e MNAR.”
  - Problema: "Os clássicos" é ambíguo entre a referência simples e os clássicos fortes. A conclusão omite a divisão com o TabPFN v2, que o slide 14 chama de "o adversário mais forte", e lida sozinha sugere que o TRIDENT vence tudo na imputação. Falta também o segundo eixo da tese, o [NULL] diante da falta informativa (07:54, 113).
  - Correção: "Imputação: com um decode por coluna, é competitivo; vence a referência simples e os clássicos fortes na maioria das variantes e mantém a posição sob MAR e MNAR. Diante do TabPFN v2, lidera nas numéricas até 60% e perde a 80% e nas mais categóricas." Acrescentar: "Falta informativa: o [NULL] lê a falta, mas não mostramos que vença a imputação com indicador."

- **baixa** · tela · grafia
  - Trecho: “O mesmo encoder, dois veredictos”
  - Problema: Os slides 6, 11 e 13 e a tese escrevem "veredito", e a forma diferente aparece justo no título da conclusão.
  - Correção: "O mesmo encoder, dois vereditos".

### Slide 22

- **media** · tela · generalização errada + limites omitidos
  - Trecho: “Os números de benchmark usam o oráculo K; H é mais modesta, sobretudo a 80%.”
  - Problema: Os slides 13, 15 e 17 relatam H, e a frase parece desqualificar todos os números. Faltam dois limites que a tese declara (07:79, 81). O primeiro é o ajuste assimétrico: TabPFN, MaCoDE e ReMasker foram executados sem ajuste, e o TRIDENT foi ajustado no desenvolvimento. O segundo: os intervalos de 15 pares são otimistas, e com Nadeau–Bengio 3 de 16 vitórias de K no desenvolvimento e 1 de 11 na confirmação virariam "sem diferença".
  - Correção: "Parte dos números usa o oráculo K; H, a leitura implantável, é mais modesta, sobretudo a 80%." Acrescentar: "O TabPFN, o MaCoDE e o ReMasker foram executados sem ajuste; o TRIDENT foi ajustado no desenvolvimento." Nas notas: o caso de Nadeau–Bengio (3 de 16 e 1 de 11) e a volta das dobras da confirmação em estudos posteriores.

### Slide 23

- **baixa** · tela · jargão de projeto
  - Trecho: “Números do registro de experimentos do fork e dos tickets”
  - Problema: "Fork" e "tickets" são jargão do projeto (as notas do 2 e os slides 4 e 9 também usam "fork"), e a plateia não sabe o que são. Os ids E aparecem na tela desde o slide 4 e só aqui são explicados.
  - Correção: "Os números vêm do nosso registro de experimentos; o identificador entre parênteses, como (E87), remete a cada experimento." Pôr a mesma frase como rodapé curto no slide 4 e trocar "fork" por "nossa versão" nos demais slides.

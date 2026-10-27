# Rascunho: TRIDENT como imputador — fundamentação, metodologia, resultados e direções

> **Status:** rascunho para discussão, 2026-10-27 (v4). Não é o artefato; é o esqueleto que decidiu o
> que entra nele. O artefato está publicado (link no fim).
>
> - **Para quê:** a reunião com a orientadora, para apresentar o que foi feito e decidir direções.
> - **Cobre:** E01–E81 e os ADRs até 0031 (`feat/imputation-task` em `8e74064`).
> - **Substitui:** o artefato "Resultados da imputação TRIDENT" (2026-10-23), que só lista
>   resultados. Este acrescenta a ligação com as RQs da tese (wiki: `projects/trident/thesis-goals`,
>   `claims`, `thesis-tldr`) e, para cada experimento central, o percurso pergunta → hipótese →
>   resultado → conclusão.
> - **Fontes que o artefato vai reaproveitar:**
>   - os relatórios `2026-10-21-thesis-table-with-tabpfn` e `2026-10-22-results-consolidated`;
>   - o ledger de claims C1–C17 da wiki;
>   - os Outcomes dos tickets.
> - **Números:** todos serão reconstruídos por script a partir do store, nunca copiados à mão.
>   Os que aparecem aqui servem só para orientar a discussão.

## Decisões de 2026-10-27

| # | Pergunta | Decisão | Efeito no artefato |
|---|---|---|---|
| 1 | Público | A orientadora é coautora do TRIDENT e conhece pouco do fork. | Não explicar o TRIDENT; explicar bem **o que o fork mudou** (nova seção 2.1). |
| 2 | Tom | Defender a RQ1 e apresentar a RQ2 (e as outras) como trabalho futuro planejado. | A RQ1 é o corpo; RQ2–RQ4 aparecem como plano, com protocolo e primeiros sinais. |
| 3 | Seção 3 e ADRs | A trajetória em forma concisa, sem citar ADRs; os ADRs vão para o apêndice. | A seção 3 fala em decisões, não em números de ADR. |
| 4 | Classificação | Entra. | Nova seção 5 (E70/E71). |
| 5 | Figura de trajetória | Recuperar do store: os passos existem nas mesmas seeds (42/7/13). | Uma curva de `impute_score` sem runs novas (seção 3). |
| 4b | E71 | Fica como está; replicação com os parâmetros antigos e mais folds como tarefa futura. | T41 em `docs/TASKS.md`; seção 5. |
| 6 | Posicionamento | Entra como tabela, explicando o que cada método faz. | Seção 1.3. |
| 7 | Decisão de posicionamento | Opções neutras, sem recomendação. | Seção 10. |

## Como ler este rascunho

Cada seção traz:

- **Objetivo:** o que a seção precisa convencer;
- **Conteúdo candidato:** os experimentos (E##) que a sustentam;
- **Figura:** a proposta e de onde vem;
- **Decisão:** `[entra]` (proponho que entre no corpo), `[apêndice]`, `[?]` (você decide).

**Formato de cada experimento central (o "card"):**

1. **Pergunta:** a RQ ou hipótese a que responde.
2. **O que foi pensado:** a hipótese pré-registrada e o porquê.
3. **Desenho:** braços, variantes, seeds e regra de decisão.
4. **Resultado:** a figura e a contagem melhor / sem diferença / pior.
5. **Conclusão:** o veredito e o que mudou no modelo (ou nada).

Os números E## ficam nos cards como referência pequena. Os ADRs não aparecem no corpo; o
apêndice traz a tabela "decisão → ADR".

---

## 0. Abertura `[entra]`

- **Objetivo:** em uma tela, dizer o que já foi demonstrado e qual é o plano.
- **Conteúdo candidato:**
  - **"O que o TRIDENT já demonstra como imputador"**, em 4–5 linhas:
    - vence a barra clássica na maioria das variantes, inclusive na confirmação lida uma única
      vez;
    - fica empatado ou à frente do TabPFN v2 até 60% de faltantes;
    - mantém a posição sob MNAR e MAR;
    - preserva a dispersão dos dados melhor que o TabPFN.
  - **"O que vem a seguir":** RQ2 (MNAR na predição), RQ3 (alvo latente) e RQ4 (o split), cada uma
    com o protocolo já desenhado.
  - **As três decisões a encaminhar**, como opções neutras: posicionamento, a afirmação
    `[NULL]`/`[MASK]` e os dados clínicos (seção 10).
- **Figura:** nenhuma; um bloco de texto.

## 1. A tese `[entra]`

### 1.1 Motivação

- **Objetivo:** situar a tese.
- **Conteúdo candidato**, curto (wiki, `thesis-goals` § Motivation):
  - em dados clínicos, a falta é muitas vezes informativa;
  - sob MNAR, as conclusões mudam (Le Morvan & Varoquaux 2025);
  - o TRIDENT original só foi avaliado sob MCAR, e a sua máscara de pré-treino é do tipo MCAR.

### 1.2 RQs e cobertura

- **Objetivo:** mostrar que a RQ1 está respondida e que as demais têm um plano.
- **Conteúdo candidato:** a grade abaixo. A última coluna, "plano", é o que transforma uma lacuna
  em trabalho futuro.

| RQ / hipótese | Estado | Evidência | Plano |
|---|---|---|---|
| **RQ1, imputação (H1d)** | respondida, com escopo a fixar | MCAR, MNAR, MAR, sorteios, seeds, confirmação (seções 4 e 6) | fixar a margem contra os imputadores deep |
| **RQ1, classificação (H1, H1c)** | primeiro passo dado | E70/E71 (seção 5) | refazer a Table 3 do artigo com ≥5 seeds e orçamentos iguais (t3/t24); TabPFN na classificação (T40) |
| **RQ2 / H2, H2b** (`[NULL]` contra imputação + indicador) | planejada | nenhuma run ainda | protocolo C15 (wiki): não-inferioridade a −0,01 de macro-F1 e +2% de log loss, com as margens já aceitas |
| **RQ2.2** (o preditor usa a falta?) | sinal inicial | E81: um decoder com `[NULL]` lê o padrão de lacunas sob MNAR moderado e não sob MCAR | Shapley no preditor (protocolo C15) |
| **RQ3 / H3** (latente vs valor) | parcial | o alvo normalizado repara o objetivo original; o decode apaga o pré-treino (seção 7.1) | contraste latente × valor com encoder congelado, e na classificação |
| **RQ4** (máscara por mecanismo, o split) | parcial | qual token preenche a lacuna importa (`[MASK]` vence); a máscara por propensão e o `[NULL]` sozinho foram refutados (seção 7.4) | protocolo C1: a ablação tied vs untied |

- **Figura 1:** a grade como matriz colorida (respondida / parcial / planejada).
- **Como dizer o C1 sem enfraquecer a RQ1:** a ablação limpa do split ainda não rodou. Os dois
  testes vizinhos (E67, E81) dão sinais mistos, e por isso o protocolo C1 propõe rodá-la na
  classificação, onde o TRIDENT foi desenhado para ser usado.

### 1.3 Posicionamento: os parentes e o que cada um faz `[entra]`

- **Objetivo:** situar o TRIDENT e os oponentes do fork. A orientadora conhece parte deles pelo
  artigo, mas não os que vieram depois.
- **Conteúdo candidato** (das páginas da wiki; a versão final confere cada linha com a página da
  referência):

| Método | O que faz | Em comum com o TRIDENT | O que muda | No fork |
|---|---|---|---|---|
| **SAINT** (2021) | Transformer tabular com atenção entre colunas e entre linhas | um token de "faltante" aprendido por coluna | no artigo do TRIDENT, rodou só com entradas imputadas | — |
| **MaCoDE** (AAAI 2025) | gerador estilo BERT que também imputa: prevê a faixa (quantil) de cada célula mascarada | um vetor por coluna, aprendido, para a célula faltante | o mesmo vetor serve para lacunas reais e para buracos de pré-treino (sem split); alvo em faixas, não latente | oponente externo (E48) |
| **MFAN** (2025) | Transformer que prevê a partir de tabelas incompletas; pré-treina restaurando valores e a máscara de células apagadas | embedding de faltante por coluna, mais pré-treino e fine-tuning | as células apagadas recebem o mesmo embedding de faltante; taxa fixa de 10%; alvo em valor; só MCAR | — |
| **Knockout** (TMLR 2025) | treina trocando entradas, ao acaso, por valores fora do suporte, para aprender p(Y \| X_obs) | separa dois tipos de "ausente" | dois valores fixos (+10 e −10), não tokens aprendidos; MLP sem pré-treino; o único ablation do split encontrado | — |
| **MaskTab** (2026) | Transformer industrial (risco de crédito) com `[MASK]` e `[MISS]` separados e reconstrução em valor | o split `[MASK]`/`[MISS]` | tokens compartilhados entre colunas; alvo em valor; máscara do tipo MCAR | — |
| **SMART** (NeurIPS 2024) | classificador de séries temporais de EHR com atenção ciente da falta, pré-treinado reconstruindo no espaço latente | alvo latente, `[CLS]`, falta explícita | séries temporais; usa um encoder-alvo EMA e um preditor (guardas contra colapso) | — |
| **T-JEPA** (ICLR 2025) | JEPA para tabelas: prevê o embedding das colunas escondidas a partir das visíveis | alvo latente | dados completos (sem faltantes); precisou de um token `[REG]` para não colapsar | — |
| **ReMasker** (ICLR 2024) | autoencoder mascarado em Transformer: re-mascara células observadas, codifica só as visíveis e reconstrói todas as colunas com MSE | aprender a imputar escondendo células observadas | sem token para a lacuna real (lacunas saem do encoder); alvo em valor; sem fine-tuning | — (citado) |
| **CACTI** (ICML 2025) | um ReMasker que copia máscaras de padrões reais de falta e usa embeddings dos nomes das colunas | idem | as máscaras seguem a falta real (relaciona-se à RQ4); usa o contexto dos nomes das colunas | — (citado) |
| **TabPFN v2** (Nature 2025) | Transformer pré-treinado em ~100M tabelas sintéticas que prevê em uma passada, usando as linhas de treino como contexto | — | sem treino na tabela-alvo; limite de 10.000 linhas; prior só MCAR; no fork, imputa coluna por coluna, sem tuning | tier forte, offline (E55) |
| **not-MIWAE** (ICLR 2021) | VAE com pesos de importância que modela explicitamente a falta, p(m \| x) | — | modela o mecanismo MNAR de forma explícita | oponente MNAR (E72/E73) |

- **Fecho da seção:** depois de MaCoDE e MFAN, o token de faltante por coluna não é mais
  exclusivo. O que continua distinto é a combinação:
  - `[NULL]` por coluna separado de `[MASK]` por coluna;
  - alvo latente em tabelas estáticas;
  - taxa de máscara adaptada à falta.

  Nenhum dos três foi ablado ainda (seção 10).
- **Decisão:** `[?]` 11 linhas é bastante. Proponho manter no corpo os que rodaram no fork mais
  MFAN e Knockout, e o resto no apêndice.

## 2. Metodologia `[entra]`

- **Objetivo:** que a orientadora consiga avaliar a força de cada resultado sem ler os ADRs.

### 2.1 O que o fork mudou em relação ao TRIDENT do artigo `[entra]`

O artigo é classificação: pré-treino latente seguido de fine-tuning de um classificador. O fork
mantém o caminho de classificação intacto por padrão e acrescenta:

| No artigo | No fork | Por quê |
|---|---|---|
| fine-tuning de um classificador | **uma tarefa de imputação:** no lugar do classificador entra uma etapa de *decode*, com uma cabeça por coluna (cross-entropy para as categóricas, MSE ponderado para as numéricas) | para responder a RQ1 |
| objetivo latente: regredir a saída do encoder nas células `[MASK]` contra o embedding de entrada destacado | **`embedding_normalized`:** alvo com layer norm e um preditor de duas camadas | o objetivo original terminava 4–8× pior que uma constante por coluna no próprio objetivo |
| — | **no decode, as lacunas reais entram como `[MASK]`** (e não como `[NULL]`) | melhor em 11 e 13 de 21 variantes, pior em nenhuma |
| época final | **checkpoint e calibração escolhidos nas lacunas de validação** (leitura K/KC), com H como leitura implantável | ganho em variantes de alta falta |
| — | **encoder maior nas tabelas grandes**, a 20nan | ajuda nas grandes, atrapalha nas pequenas |
| avaliação sob MCAR, contra classificadores | **protocolo de imputação:** células induzidas, MCAR/MNAR/MAR, dois tiers de baselines, pré-registro, desenvolvimento separado da confirmação | para dar peso aos vereditos |

- **Figura 2:** diagrama do pipeline (pré-treino → decode → leitura K/KC/H), com o que é novo no
  fork destacado em cor.

### 2.2 Dados e variantes

- **Desenvolvimento:** 9 tabelas OpenML do artigo (kr-vs-kp, spambase, credit-g, vehicle, biodeg,
  kc2, pendigits, letter, electricity) e 21 variantes `<tabela>_<taxa>nan`.
- **Confirmação:** as 15 variantes não usadas no desenvolvimento, lidas uma única vez (E44).
- **Seeds:** cinco conjuntos de seeds; 101/202/303 é o primário.

### 2.3 Mecanismos

- **MCAR** induzido;
- **MNAR:** a família dependente do valor, com um ajuste de inclinação no biodeg;
- **MAR:** 30% de colunas motoras sempre observadas decidem as lacunas, comparadas a controles
  MCAR pareados.

### 2.4 O que é medido

- **`impute_score`:** menor é melhor; 1.0 é o preenchimento por média/moda.
- **Onde:** nas células induzidas, as mesmas para todos os métodos.
- **Painel secundário:** MAE, Wasserstein e dispersão.

### 2.5 Baselines

- **Barra fraca:** mean/mode, kNN 5/10, HGB e LightGBM.
- **Tier forte:** missForest, condexp, SoftImpute, ICE e o TabPFN v2.
- **Externos:** MaCoDE e not-MIWAE.
- **Orçamento de tuning:** igual para todos, só na validação.

### 2.6 Regra de leitura e disciplina

- **O teste:** t pareado por fold, com concordância entre seeds.
- **Equivalência:** ±1%.
- **Disciplina:**
  - pré-registro antes da primeira célula;
  - sem parada opcional;
  - confirmação lida uma única vez.
- **Por que importa:** é o que dá peso a um "refutado".

### 2.7 As leituras K / KC / H

- **K** é um oráculo: escolhe a época com a verdade das lacunas de validação.
- **KC:** o K calibrado, a saída default a partir de 40% de faltantes.
- **H:** o checkpoint da loss, a leitura implantável.
- **Ressalva visível:** o que precisa transferir (MNAR, implantação) é lido no H.

**Decisão:** `[?]` proponho 2.1 inteira no corpo, com 2.2–2.7 em quadros curtos.

## 3. Como chegamos no default `[entra]`, conciso e sem ADRs

- **Objetivo:** mostrar que o default não é tuning cego. Cada peça entrou por um teste
  pré-registrado, e várias ideias foram descartadas.
- **Conteúdo:** uma linha do tempo de 6 a 8 passos, cada um com uma frase. Por exemplo:
  1. **Busca de hiperparâmetros:** não bateu os defaults; a alavanca estava no método, não nos
     hiperparâmetros.
  2. **Objetivo latente normalizado + decode longo:** melhor em 9 de 21 variantes, pior em
     nenhuma.
  3. **Lacunas mostradas como `[MASK]` no decode:** melhor em 11 e depois 13 de 21, pior em
     nenhuma.
  4. **Checkpoint escolhido pelas lacunas de validação:** 6 variantes melhores, nenhuma pior.
  5. **Encoder maior:** só nas tabelas grandes.
  6. **Calibração:** a partir de 40nan.
- **Descartadas** (uma linha): `[MASK]` também no pré-treino, alavancas contra o washout, máscaras
  por propensão e `[NULL]` sozinho no decode.
- **Os ADRs** de cada passo vão para o apêndice, na tabela "decisão → ADR → experimento".

### Figura 4: a trajetória vem do store, sem runs novas

Verificado no store em 2026-10-27, num script read-only de agregação.

- **O conjunto de seeds:** os passos 1 a 3 rodaram todos no conjunto 42/7/13 (E24–E26, E30, E32),
  nas 21 variantes de desenvolvimento.
- **K e KC não precisam de runs próprias.** Cada run loga métricas por leitura:
  - H em `impute_score`;
  - K em `induced_checkpoint/impute_score`;
  - KC em `induced_checkpoint_calibrated/impute_score`.

  Por isso K e KC saem das mesmas runs do passo 3.
- **A curva fica:** primeiro decoder → + alvo normalizado e decode 450 → + lacunas como `[MASK]` →
  + checkpoint K → + calibração KC, com as 21 variantes nas mesmas três seeds.
- **Leitura preliminar** (média simples de `impute_score`, não é número para o artefato):
  0,792 → 0,776 → 0,754 → 0,745 → 0,742.

**Ressalvas a resolver no script final:**

- **Variantes com arquivo promovido:** cinco variantes (kr-vs-kp_20/40nan, spambase_20/40nan,
  credit-g_40nan) tinham arquivos de hiperparâmetros promovidos em parte desse período. É preciso
  conferir se todos os passos usaram a mesma configuração nelas.
- **Calibração:** a curva calibra em todas as taxas, mas o default só calibra a partir de 40nan. O
  último passo deve seguir a regra do default.
- **Encoder maior:** não tem runs em 42/7/13 nas tabelas grandes. Ou fica fora da curva, ou entra
  como um ponto à parte em 101/202/303.
- **Runs duplicadas:** uma mesma configuração aparece em mais de um experimento. É preciso fixar a
  regra (média, ou a run do experimento que decidiu o passo).
- **Versão do código:** muda entre os passos (por exemplo, as cabeças batched). Ver se isso move
  números.

**Desenho:** a curva, com a contagem pré-registrada de cada passo anotada no ponto. A curva mostra
o quanto o modelo melhorou no total, e a contagem mostra o veredito de cada passo.

## 4. RQ1, imputação sob MCAR `[entra]`

- **Objetivo:** onde o TRIDENT está contra cada nível de oponente, com franqueza sobre o TabPFN.
- **Cards:**
  - **4.1 Contra a barra e o tier forte clássico:**
    - no desenvolvimento, K − forte é 13 / 7 / 1;
    - na confirmação, 12 / 1 / 2;
    - o resultado se mantém com seeds novas e sorteios novos.
  - **4.2 Contra o TabPFN:**
    - misto, por tabela e taxa;
    - o TRIDENT lidera em spambase, letter, pendigits a 20–60nan, biodeg abaixo de 80nan e kc2 a
      60–80nan;
    - o TabPFN lidera a 80nan e em kr-vs-kp e credit-g a partir de 40nan.
  - **4.3 Confirmação:** lida uma única vez; KC − TabPFN é 6 / 4 / 1 equivalente / 4, e todas as
    derrotas são a 80nan.
  - **4.4 Distribuição:** a dispersão do TRIDENT fica mais perto da verdade em 14 de 15; as
    vitórias do TabPFN a 80nan vêm com preenchimentos mais encolhidos.
  - **4.5 MaCoDE:** 20 de 21 e 14 de 15. É um oponente sem tuning. `[apêndice?]`
- **Figura 5 (a principal):** heatmap tabela × taxa, com uma célula por variante.
  - **Cor:** a chamada de K − TabPFN.
  - **Segunda camada:** K − barra.
  - **Conjuntos:** desenvolvimento e confirmação lado a lado.
  - **Fonte:** o script da tabela da tese, numa passada só.
- **Figura 6:** as contagens por conjunto de seeds, reaproveitadas do artefato de 10-23.

## 5. RQ1, classificação: o primeiro passo `[entra]`

- **Objetivo:** mostrar que a segunda metade da RQ1 começou e o que o primeiro passo achou. Deve
  ser apresentado com cuidado, porque toca num resultado do artigo.
- **Cards:**
  - **E70, reprodutibilidade:**
    - **Pergunta:** o código atual, no servidor, reproduz o sweep de classificação antigo?
    - **Resultado:** não. As diferenças chegam a 0,03–0,09 por fold, nos dois sentidos, e as
      médias se movem pouco (−0,006 e +0,012).
    - **Consequência:** o sweep antigo passa a ser só referência descritiva, e o braço A foi
      rodado de novo.
  - **E71, o que o pré-treino compra na classificação:**
    - **Desenho:** 45 variantes, 3 folds, 2 seeds completas, com e sem pré-treino.
    - **Resultado:** nenhuma diferença detectável em nenhum nível (de −0,005 a +0,004 de
      macro-F1). Os +1,7 pontos da Table 4 do artigo ficam fora dos intervalos de 0 a 60nan.
- **Ressalvas a dizer junto:**
  - são os defaults do fork, com 3 folds e 2 seeds, e não o protocolo do artigo;
  - é a mesma conclusão que o E24 teve na imputação;
  - o E71 já usou os parâmetros do sweep antigo (300/150 épocas, `cosine_legacy`), mas não o
    protocolo do artigo (40/40 épocas, hiperparâmetros por dataset, split 80/10/10).
- **Trabalho futuro, a T41 (registrada em `docs/TASKS.md`):** replicar o ganho da Table 4 com os
  parâmetros do artigo e com mais folds (10) e ≥5 seeds, em dois braços lidos separadamente.
  - **O que pedir à orientadora:** os hiperparâmetros por dataset do artigo e o commit que
    representa "o código do artigo". Eles não estão no repositório.
- **Relacionadas:** t3 (a Table 3 com ≥5 seeds) e T40 (o TabPFN como baseline de classificação).
- **Figura 7:** B − A por nível de falta, com intervalos; uma linha em +1,7 como a referência do
  artigo.

## 6. Robustez: seeds, sorteios e mecanismos `[entra]`

- **Objetivo:** a posição da seção 4 não é artefato de uma seed, de um sorteio ou do MCAR.
- **Conteúdo candidato:**
  - **6.1 Seeds e sorteios:** nenhuma inversão.
    - **Ressalva:** o sorteio move um contraste tanto quanto o intervalo, logo os intervalos
      valem para um sorteio fixo.
  - **6.2 MNAR:**
    - a vantagem sobre a barra se mantém em 7 de 7 pares;
    - o TabPFN não ganha terreno em relação ao MCAR;
    - o desvio com sinal do TRIDENT é o que menos muda entre 10 métodos, em 9 de 10 pares.
  - **6.3 MAR:**
    - a família de seis tabelas refutou o "perde terreno" do smoke (pior só em biodeg_60nan);
    - a barra ganha a 60nan nas tabelas numéricas pequenas.
  - **6.4 Um imputador que modela o mecanismo (not-MIWAE):** misto.
    - Com o preenchimento pela média, o termo do mecanismo encolhe o viés mais que o TRIDENT.
    - Aponta para a RQ4.
- **Figura 8:** o ajuste de inclinação do MNAR no biodeg, com o desvio com sinal por método; o
  TRIDENT destacado.
- **Figura 9:** um forest plot de MAR menos controle, 12 linhas com intervalos.

## 7. Por que o modelo se comporta assim `[entra]`

- **7.1 O decode apaga o pré-treino** (E24, E49–E53; ligar ao E71):
  - a representação pré-treinada é real (não colapsa; probes lineares ganham);
  - mas o decode longo reconstrói, a partir do aleatório, um encoder tão bom quanto.
  - **Relevância:** RQ3 e o risco do Mazrouei 2026.
  - **Figura 10:** as curvas de decode, pré-treinado contra aleatório.
- **7.2 Onde perde para o TabPFN, e o que não resolveu:**
  - 80nan e as tabelas categóricas;
  - nem capacidade, nem ponderação por coluna (o déficit é uniforme), nem as alavancas contra o
    washout;
  - electricity tem mecanismo conhecido: `date` como chave.
- **7.3 Complementaridade e o híbrido:**
  - os erros se complementam onde os dois estão próximos;
  - o híbrido por coluna KC + TabPFN nunca é pior que o default;
  - **status:** opt-in, offline, como medida de viabilidade.
- **7.4 O que se sabe sobre `[NULL]`/`[MASK]`:**
  - qual token preenche a lacuna importa no imputador (`[MASK]` vence);
  - `[MASK]` no pré-treino não muda nada;
  - `[NULL]` sozinho custa mais sob MNAR íngreme;
  - o decoder `[NULL]` lê o padrão de lacunas, mas perde acurácia.
  - **Por que importa:** é a ponte para o trabalho futuro da RQ4 e do C1.

## 8. Saídas novas: incerteza e interpretabilidade `[?]`

- **Spread de re-masking:**
  - ordena os erros dentro de uma coluna até 60nan;
  - não estreita intervalos com a mesma cobertura;
  - cai quando as lacunas da linha crescem, embora o erro suba.
- **Mapa de oclusão:** de quais colunas o decoder preenche cada coluna. É fiel em 9 de 9 a 40nan e
  em 7 de 9 a 60nan, com AUROC 0,979 no sintético.
- **Figura 11:** o mapa de oclusão do sintético, com a verdade conhecida.
- **Decisão:** `[?]` corpo, curto, ou apêndice? Ligam-se à RQ2.2 e ao uso clínico.

## 9. Limites `[entra]`

- os benchmarks são OpenML, com mecanismos injetados e nenhum dado clínico;
- K e KC são oráculos;
- 80nan e as tabelas categóricas ficam com o TabPFN;
- o pré-treino é apagado pelo decode;
- a classificação tem só o passo zero;
- vários resultados se apoiam num só conjunto de seeds ou num só sorteio.

## 10. Trabalho futuro e decisões `[entra]`, com opções neutras

- **Trabalho futuro, por RQ**, cada item com o protocolo e o custo estimado:
  - **RQ1, classificação:** t3 (Table 3 com ≥5 seeds) e T40 (TabPFN).
  - **RQ2 / RQ2.2:** protocolo C15.
  - **RQ3:** contraste latente × valor com encoder congelado.
  - **RQ4 / C1:** protocolo C1.
  - **Dados clínicos:** PhysioNet 2012/2019, de acesso aberto.
- **Decisão 1, posicionamento**, sem ordem de preferência:
  - **(a)** RQ1 como capítulo principal, e RQ2 como estudo menor;
  - **(b)** RQ2 como contribuição principal, com a RQ1 como base;
  - **(c)** as duas, cada uma com menos profundidade.
  - **Para cada opção:** o que já está pronto, o que falta, e o argumento contra (por exemplo, Le
    Morvan & Varoquaux 2025 contra a "imputação como contribuição").
- **Decisão 2, a afirmação principal:**
  - **(a)** manter o split e rodar o C1;
  - **(b)** rebaixar o split a componente e centrar a afirmação em outra coisa, como robustez a
    MNAR, ou um imputador com incerteza e mapa de fontes.
- **Decisão 3, dados clínicos:** qual base e qual clínico, e como avaliar sem a verdade dos valores
  faltantes (Digitale 2025).
- **Riscos para a banca:** Mazrouei 2026 e o resultado próprio do E71, que contesta a Table 4.
- **Figura 12** `[?]`: uma árvore "decisão → próximo experimento → custo".

## Apêndice `[apêndice]`

- **Decisão → ADR → experimento:** a tabela que sai da seção 3.
- **Glossário:** `Nnan`, `impute_score`, K/KC/H, a barra, o tier forte, melhor / sem dif. / pior.
- **Os parentes restantes** da seção 1.3.
- **As linhas de resumo dos leitores** (`results_digest.py`).
- **A lista E01–E81**, com o veredito de cada um.

## Formato do artefato final

- **Página:** HTML em português, com os identificadores em inglês e um índice lateral, na mesma
  família visual dos artefatos anteriores.
- **Experimentos:** os experimentos centrais como cards.
- **Figuras:** geradas por um script que lê o store numa passada só, em SVG inline, para claro e
  escuro.

## Ainda em aberto

Nada. Resolvido em 2026-10-27:

- **Figura 4:** a curva vem do store (seção 3).
- **E71:** entra como está, com a T41 como trabalho futuro de replicação.
- **Seção 1.3:** as 11 linhas no corpo.
- **Seção 8:** no corpo.

**Artefato publicado:** https://claude.ai/artifact/47T9yMLRK6FybHtt5UNeDw.

- **Página:** `docs/reports/2026-10-27-reuniao-orientadora.html`, gerada por
  `scripts/experiments/meeting_figures.py`.
- **Dados:** a saída dos leitores fica em `docs/reports/2026-10-27-reuniao-orientadora/`.

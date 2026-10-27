# Rascunho: TRIDENT como imputador — fundamentação, metodologia, resultados e direções

> **Status:** rascunho para discussão, 2026-10-27, 12:58 GMT-3. Não é o artefato; é o esqueleto
> que decide o que entra nele.
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

## Como ler este rascunho

Cada seção traz:

- **Objetivo:** o que a seção precisa convencer;
- **Conteúdo candidato:** os experimentos (E##) e ADRs que a sustentam;
- **Figura:** a proposta e de onde vem;
- **Decisão:** `[entra]` (proponho que entre no corpo), `[apêndice]`, `[?]` (você decide).

**Formato proposto para cada experimento central (o "card"):**

1. **Pergunta:** a RQ ou hipótese a que responde.
2. **O que foi pensado:** a hipótese pré-registrada e o porquê.
3. **Desenho:** braços, variantes, seeds e regra de decisão.
4. **Resultado:** a figura e a contagem melhor / sem diferença / pior.
5. **Conclusão:** o veredito e o que mudou (ADR, default, ou nada).

---

## 0. Abertura: o que quero da reunião `[entra]`

- **Objetivo:** em uma tela, dizer o que já dá para afirmar e o que preciso decidir com ela.
- **Conteúdo candidato:**
  - **"O que dá para afirmar hoje"**, em 4–5 linhas, a partir do "What the ledger supports today"
    da wiki (`claims.md`), atualizado até E81. Por exemplo:
    - o TRIDENT vence a barra clássica na maioria das variantes;
    - fica empatado ou à frente do TabPFN até 60nan;
    - sua posição se mantém sob MNAR e MAR;
    - o pré-treino quase não contribui.
  - **As três decisões em aberto** (`thesis-tldr`):
    1. **Posicionamento:** o TRIDENT como imputador (RQ1/RQ3) ou predição robusta a MNAR
       (RQ2/RQ4)?
    2. **A afirmação principal:** o par `[NULL]`/`[MASK]` por coluna. Ainda é nova? Se sustenta?
    3. **Dados clínicos e avaliação.**
- **Figura:** nenhuma; um bloco de texto.

## 1. A tese: motivação e research questions `[entra]`

- **Objetivo:** ancorar tudo o que vem depois nas RQs, e mostrar com franqueza onde há evidência
  e onde não há.
- **Conteúdo candidato:**
  - **Motivação**, curta (wiki, `thesis-goals` § Motivation):
    - em dados clínicos, a falta é muitas vezes informativa;
    - sob MNAR, as conclusões mudam (Le Morvan & Varoquaux 2025);
    - o TRIDENT original só foi avaliado sob MCAR.
  - **As RQs e hipóteses:**
    - RQ1 / H1d (imputação) e H1, H1c (classificação);
    - RQ2 / H2, H2b, H2.2 (`[NULL]` contra imputação + indicador);
    - RQ3 / H3 (alvo latente contra alvo em valor);
    - RQ4 (máscara consciente do mecanismo).
  - **A grade de cobertura** (abaixo), a peça mais útil para uma reunião sobre direções.

| RQ / hipótese | Estado da evidência | Experimentos | O que falta |
|---|---|---|---|
| **RQ1, imputação (H1d)** | muito testada: MCAR, MNAR, MAR, sorteios, seeds, confirmação | E19–E66, E72–E74 (C3–C10, C13) | margem de não-inferioridade contra imputadores deep (ReMasker, CACTI, DiffPuter, TabImpute); decisão de escopo |
| **RQ1, classificação (H1, H1c)** | não testada | E70/E71: só o passo zero (o fork não reproduz o sweep antigo; pré-treino sem ganho detectável) | t3/t24: Table 3 refeita, ≥5 seeds |
| **RQ2 / H2, H2b (`[NULL]` vs imputação + indicador)** | não testada | nenhuma run (C15) | protocolo C15, rascunho na wiki |
| **RQ2.2 (o preditor usa a falta?)** | só um proxy, na imputação | E81: o decoder com `[NULL]` lê o padrão de lacunas sob MNAR moderado, não sob MCAR; custa acurácia em 11 de 11 | Shapley no preditor (protocolo C15) |
| **RQ3 / H3 (latente vs valor)** | parcial | E25/E30 (alvo normalizado repara o original, ADR 0013); T28: E24, E49–E53 (o decode apaga o pré-treino); E71 | contraste latente × valor com encoder congelado; classificação |
| **RQ4 (máscara por mecanismo, o split)** | parcial | E01, E32/E33 (gaps como `[MASK]`, ADR 0014); E36; E65 (máscara por propensão, refutada); E67 (`[NULL]` sozinho, refutado) | **C1, a ablação limpa do split: nunca rodou** |

- **Ponto a deixar explícito:** a afirmação principal escolhida em 2026-10-15, o par
  `[NULL]`/`[MASK]` por coluna (C1), não tem teste limpo.
  - E67 é o mais próximo e foi refutado: `[NULL]` sozinho no decode custa mais à medida que a
    inclinação do MNAR aumenta.
  - Isso pesa diretamente na decisão 2.
- **Figura 1:** a grade acima, como matriz colorida RQ × estado (testada / parcial / não testada),
  com os E## clicáveis.
- **Posicionamento contra os parentes** (MFAN, Knockout, MaskTab, ReMasker/CACTI, TabPFN): uma
  tabela curta "o que cada um já tem". `[?]` entra no corpo, vira apêndice, ou fica só na fala?

## 2. Metodologia `[entra]`

- **Objetivo:** que a orientadora consiga avaliar a força de cada resultado sem ler os ADRs.
- **2.1 O modelo e a tarefa** (ADR 0004, 0011, 0013, 0014):
  - **Embedder:** o TabularEmbedder, com `[NULL]` e `[MASK]` por coluna.
  - **Pré-treino:** `embedding_normalized`.
  - **Decode:**
    - heads por coluna, 450 épocas, batched (ADR 0012);
    - as lacunas mostradas como `[MASK]`;
    - encoder maior nas tabelas grandes a 20nan (ADR 0018).
  - **Figura 2:** diagrama do pipeline (pré-treino → decode → leitura K/KC/H), em SVG feito à
    mão.
- **2.2 Dados e variantes:**
  - **Desenvolvimento:** 9 tabelas OpenML (kr-vs-kp, spambase, credit-g, vehicle, biodeg, kc2,
    pendigits, letter, electricity) e 21 variantes `<tabela>_<taxa>nan`.
  - **Confirmação:** as 15 variantes não usadas no desenvolvimento (E44), lidas uma única vez.
  - **Seeds:** os conjuntos 101/202/303 (primário), 42/7/13, 404/505/606, 1001/2002/3003 e
    707/808/909.
- **2.3 Mecanismos:**
  - **MCAR** induzido;
  - **MNAR:** a família dependente do valor (E45/E46), com o ajuste de inclinação no biodeg;
  - **MAR:** 30% de colunas motoras sempre observadas, contra controles MCAR pareados
    (E63/E64).
- **2.4 O que é medido** (ADR 0007, 0008, 0025):
  - **`impute_score`:** menor é melhor; 1.0 é o preenchimento por média/moda.
  - **Onde:** nas células induzidas, as mesmas para todos os métodos.
  - **Painel secundário:** MAE, Wasserstein e dispersão.
- **2.5 Baselines** (ADR 0016, 0017, 0021, 0022, 0028):
  - **Barra fraca:** mean/mode, kNN 5/10, HGB, LightGBM.
  - **Tier forte:** missForest, condexp, SoftImpute, ICE, e o TabPFN v2 como membro offline.
  - **Externos:** MaCoDE, not-MIWAE.
  - **Orçamento de tuning:** igual para todos, só na validação.
- **2.6 Regra de leitura e disciplina** (ADR 0020):
  - **O teste:** t pareado por fold, com concordância entre seeds.
  - **Equivalência:** ±1%.
  - **Disciplina:**
    - pré-registro antes da primeira célula;
    - desenvolvimento separado da confirmação;
    - sem parada opcional.
  - **Por que importa:** é o que dá peso a um "refutado".
- **2.7 As leituras K / KC / H** (ADR 0015, 0024, 0027):
  - **K:** o checkpoint escolhido pelas lacunas de validação. É um oráculo, pois usa a verdade
    delas.
  - **KC:** o K calibrado, a saída default a partir de 30–40% de faltantes.
  - **H:** o checkpoint da loss, a leitura implantável.
  - **Ressalva a deixar visível:** as afirmações que precisam transferir (MNAR, implantação) são
    lidas no H.
- **Figura 3** `[?]`: esquema folds × seeds × sorteio, mostrando o que cada intervalo cobre.
  - O E62 mostrou que o sorteio MCAR move um contraste tanto quanto o intervalo.
  - Talvez caiba numa frase em vez de uma figura.
- **Decisão:** `[?]` a metodologia fica toda no corpo, ou só 2.1, 2.4 e 2.7 no corpo e o resto
  num quadro recolhível?

## 3. Como chegamos no default: a trajetória de design `[?]`

- **Objetivo:** mostrar que o default atual não é tuning cego. Cada peça entrou por um experimento
  pré-registrado, e várias ideias foram refutadas.
- **Conteúdo candidato** (uma linha por decisão: hipótese → resultado → ADR):

| Decisão | Experimento | Resultado | ADR |
|---|---|---|---|
| A busca Optuna não é a alavanca | E02–E18, E21, E39 | busca reduzida e completa não batem os defaults; re-tune nunca bate | 0005, 0019 |
| Objetivo de busca nas lacunas induzidas | E22/E23 | o vencedor por máscara fica em 40º de 40 pelo induzido | 0008 |
| Pré-treino normalizado + decode 450 + cosine | E24–E26, E30 | melhor em 9 de 21, pior em nenhuma | 0011, 0013 |
| Lacunas mostradas como `[MASK]` no decode | E32, E33 | 11/21 e 13/21 melhores, nenhuma pior | 0014 |
| Checkpoint pelas lacunas de validação (K) | E34, E35 | 6 melhores, 0 piores | 0015 |
| Encoder maior só nas tabelas grandes | E37, E38, E59 | ajuda nas grandes, atrapalha nas pequenas, nada a 80nan | 0018 |
| Calibração (KC) a partir de 40nan | E40, E47 | 6 de 6 melhores; a versão implantável ganha ≤0,4% | 0027 |
| Leitura implantável (H) | E43 | — | 0024 |
| Refutadas | E36, E50, E65, E67 | `[MASK]` no pré-treino, alavancas contra o washout, máscara por propensão, `[NULL]` sozinho | — |

- **Figura 4:** uma "escada" de adoções, com a contagem melhor / sem dif. / pior de cada passo
  contra o anterior, em barras empilhadas.
  - É honesta porque cada passo tem sua própria contagem pré-registrada.
  - Uma curva única de `impute_score` ao longo dos passos misturaria conjuntos de seeds
    diferentes. Só valeria com novas runs de todos os passos nas mesmas seeds.
- **Decisão:** `[?]` corpo ou apêndice? Para direções, proponho o corpo, curto: a tabela e a
  figura, sem cards.

## 4. Resultado principal: RQ1 sob MCAR `[entra]`

- **Objetivo:** onde o TRIDENT está contra cada nível de oponente, com franqueza sobre o TabPFN.
- **Conteúdo candidato** (cards):
  - **4.1 Contra a barra e o tier forte sem TabPFN:**
    - no desenvolvimento, K − forte é 13 / 7 / 1;
    - na confirmação, 12 / 1 / 2;
    - o resultado se mantém com seeds novas (E51) e sorteios novos (E62).
  - **4.2 Contra o TabPFN:**
    - misto, por tabela e taxa (E55, E57, E58);
    - o TRIDENT lidera em spambase, letter, pendigits a 20–60nan, biodeg abaixo de 80nan e kc2
      a 60–80nan;
    - o TabPFN lidera a 80nan e em kr-vs-kp e credit-g a partir de 40nan.
  - **4.3 Confirmação (E44):** lida uma única vez; KC − TabPFN é 6 / 4 / 1 equivalente / 4, e
    todas as derrotas são a 80nan.
  - **4.4 Distribuição (E56):** a dispersão do TRIDENT fica mais perto da verdade em 14 de 15; as
    vitórias do TabPFN a 80nan vêm com preenchimentos mais encolhidos.
  - **4.5 MaCoDE (E48):** 20 de 21 e 14 de 15. É um oponente fraco (sem tuning).
    `[apêndice?]`
- **Figura 5 (a principal):** heatmap tabela × taxa, com uma célula por variante.
  - **Cor:** a chamada de K − TabPFN, ou de KC − TabPFN.
  - **Rótulo:** a diferença.
  - **Uma segunda camada:** K − barra.
  - **Conjuntos:** desenvolvimento e confirmação lado a lado.
  - **Fonte:** o script da tabela da tese (`confirmation_report.py` e os leitores de E55/E57),
    numa passada só.
- **Figura 6:** as contagens por conjunto de seeds, reaproveitadas do artefato de 10-23.

## 5. Robustez: seeds, sorteios e mecanismos `[entra]`

- **Objetivo:** a posição da seção 4 não é artefato de uma seed, de um sorteio ou do MCAR.
- **Conteúdo candidato:**
  - **5.1 Seeds e sorteios** (E51, E58, E62): nenhuma inversão.
    - **Ressalva:** o sorteio move um contraste tanto quanto o intervalo, logo os intervalos
      valem para um sorteio fixo.
  - **5.2 MNAR** (E45, E46, E58; C8, C9):
    - a vantagem sobre a barra se mantém em 7 de 7 pares;
    - o TabPFN não ganha terreno (1 / 9 / 0);
    - o desvio com sinal do TRIDENT é o que menos muda entre 10 métodos, em 9 de 10 pares.
  - **5.3 MAR** (E63, E64, E65, E68):
    - o "perde terreno" do smoke foi refutado pela família (pior só em biodeg_60nan);
    - a barra ganha a 60nan nas tabelas numéricas pequenas;
    - a máscara por propensão foi refutada.
  - **5.4 Um imputador que modela o mecanismo** (E72, E73, not-MIWAE): misto.
    - Com o preenchimento pela média, o termo do mecanismo encolhe o viés mais que o TRIDENT.
    - É contraponto direto ao H1d sob MNAR. `[?]` aqui ou na seção 6?
- **Figura 7:** o ajuste de inclinação do MNAR no biodeg, com H − barra e o desvio com sinal por
  inclinação; uma linha por método, o TRIDENT destacado.
- **Figura 8:** um forest plot de MAR menos controle (E64), 12 linhas com intervalos, para
  H − forte e H − barra.

## 6. Por que o modelo se comporta assim `[entra]`

- **Objetivo:** os diagnósticos que viram direção de pesquisa.
- **6.1 O pré-treino é apagado pelo decode** (T28: E24, E49–E53; E71 na classificação):
  - o pré-treino é real (não colapsa; probes lineares ganham em pendigits e kr-vs-kp);
  - mas o decode de 450 épocas reconstrói, a partir do aleatório, um encoder tão bom quanto.
  - **Relevância:** RQ3 e o risco do Mazrouei 2026.
  - **Figura 9:** as curvas de decode, pré-treinado contra aleatório (E49), e o encoder congelado
    contra o aleatório (E52).
- **6.2 Onde perde para o TabPFN, e o que não resolveu:**
  - 80nan e as tabelas categóricas;
  - nem capacidade (E59), nem ponderação por coluna (E61, déficit uniforme), nem as alavancas
    contra o washout (E50);
  - electricity tem mecanismo conhecido: `date` como chave (E54).
- **6.3 Complementaridade e o híbrido** (E60, E66, E69, E74; ADR 0031 aceito hoje):
  - os erros se complementam onde os dois estão próximos;
  - o híbrido por coluna KC + TabPFN nunca é pior que o default;
  - **status:** opt-in, offline, como medida de viabilidade.
- **6.4 O que sabemos sobre `[NULL]`/`[MASK]`** (E01, E32/E33, E36, E67, E81):
  - **Já se sabe:** qual token preenche a lacuna importa no imputador (C2: `[MASK]` vence);
    `[MASK]` no pré-treino não muda nada.
  - **Contra:** `[NULL]` sozinho custa mais sob MNAR íngreme (E67); o decoder `[NULL]` lê o padrão
    de lacunas, mas perde acurácia (E81).
  - **O que não se sabe:** se o par vale a pena (C1).
  - **Por que importa:** é a ponte para a decisão 2; proponho fechar a seção com essa pergunta.

## 7. Saídas novas: incerteza e interpretabilidade `[?]`

- **Objetivo:** mostrar capacidades que um imputador clássico não dá.
- **Conteúdo candidato:**
  - **Spread de re-masking** (E75–E77, E80; ADR 0030):
    - ordena os erros dentro de uma coluna até 60nan (replicado em 14 de 15);
    - não estreita intervalos com a mesma cobertura (E77, refutado);
    - cai quando as lacunas da linha crescem, embora o erro suba (E80).
  - **Mapa de oclusão** (E78, E79): "de quais colunas o decoder preenche cada coluna", fiel em
    9 de 9 a 40nan e em 7 de 9 a 60nan; AUROC 0,979 no sintético.
- **Figura 10:** o mapa de oclusão de uma tabela como heatmap coluna × coluna; o sintético, com a
  verdade conhecida, é o mais didático.
- **Decisão:** `[?]` corpo, curto, ou apêndice? Ligam-se à RQ2.2 e à proposta clínica
  (interpretabilidade), mas não respondem uma RQ diretamente.

## 8. Limites que a tese declara `[entra]`

Vem da seção 4 do relatório consolidado, atualizada:

- os benchmarks são OpenML, com mecanismos injetados e nenhum dado clínico (C17);
- K e KC são oráculos;
- 80nan e as tabelas categóricas ficam com o TabPFN;
- o pré-treino é apagado;
- a classificação não foi testada;
- vários resultados se apoiam num só conjunto de seeds ou num só sorteio.

## 9. Direções e decisões para a reunião `[entra]`

- **Objetivo:** sair da reunião com as três decisões encaminhadas.
- **Conteúdo candidato:**
  - **Decisão 1, posicionamento:**
    - **A favor da RQ1:** a evidência está madura.
    - **Contra:**
      - Le Morvan & Varoquaux 2025 (retornos decrescentes da imputação para a predição);
      - a RQ2 está vazia e é onde a motivação clínica mora.
    - **Opções a mostrar:**
      - (a) RQ1 como capítulo principal e RQ2 como estudo menor;
      - (b) virar para a RQ2 com o protocolo C15;
      - (c) as duas, cada uma com menos profundidade (a inclinação de 10-15).
  - **Decisão 2, afirmação principal:**
    - rodar o protocolo C1 (a ablação tied vs untied) na classificação, onde o TRIDENT foi
      desenhado para ser usado;
    - ou rebaixar o split a componente e trocar a afirmação principal: por exemplo, por "um
      imputador com incerteza e mapa de fontes", ou por "robustez a MNAR".
  - **Decisão 3, dados clínicos:**
    - PhysioNet 2012/2019 (acesso aberto) para uma primeira célula com falta natural;
    - o desafio de avaliar sem a verdade dos valores faltantes (Digitale 2025: preencher e
      apagar de novo sob um mecanismo conhecido).
  - **Pendências concretas:**
    - T27 (passo zero da classificação, estágios 0 e 2 feitos);
    - T40 (TabPFN na classificação);
    - T32 (o encoder maior a 40–60nan);
    - a proposta de calibração;
    - os protocolos C1 e C15 da wiki.
  - **Riscos a levantar antes da banca:** Mazrouei 2026 (o pré-treino ajuda pouco em tabelas
    com falta nativa) e o resultado próprio do T28.
- **Figura 11** `[?]`: um mapa das direções, uma árvore "decisão → próximo experimento → custo
  estimado".

## Apêndice candidato `[apêndice]`

- glossário: `Nnan`, `impute_score`, K/KC/H, a barra, o tier forte, melhor / sem dif. / pior;
- a lista de ADRs, com uma linha cada;
- as linhas de resumo dos leitores (`results_digest.py`), como no artefato de 10-23;
- a lista completa E01–E81, com o veredito de cada um.

## Formato do artefato final (proposta)

- **Página:** HTML em português, com os identificadores em inglês e um índice lateral, na mesma
  família visual dos artefatos anteriores.
- **Experimentos:** cada experimento central como card (pergunta → hipótese → resultado →
  conclusão).
- **Figuras:**
  - geradas por um script que lê o store e os leitores numa passada só;
  - em SVG inline, para funcionarem em tema claro e escuro.
- **Tamanho estimado:** com as seções 3 e 7 no corpo, cerca de 25–30 min de leitura; sem elas,
  cerca de 15.

## Perguntas para você

1. **Público e tempo:** quanto a orientadora já conhece do TRIDENT, do MNAR e do fork? Quanto dura
   a reunião? Isso decide a profundidade da seção 2.
2. **Tom:** o artefato deve *defender* a RQ1 ou *expor o vazio* da RQ2 e do C1 para provocar a
   discussão? Proponho a segunda opção, com a RQ1 como base sólida.
3. **Seções 3 e 7:** corpo ou apêndice?
4. **Classificação** (E70/E71): entra como "o primeiro passo dado", ou fica de fora até a T27
   avançar?
5. **A figura de trajetória:** contagens por passo (sem runs novas), ou vale a pena rodar todos os
   passos nas mesmas seeds para ter uma curva única de `impute_score`?
6. **Literatura:** quanto de posicionamento contra os parentes? Uma tabela no corpo, ou só a fala?
7. **Opções da decisão 1:** quer que eu proponha uma recomendação, ou que só apresente as opções
   neutras?

# Análise de Qualidade do Sistema de Busca Semântica

## 1. Resumo Executivo

O sistema de busca semântica apresenta uma arquitetura sofisticada, utilizando modelos de linguagem (BERT, BioWordVec) e Algoritmos Genéticos (AG) para expansão de consulta e seleção de features. No entanto, sua eficácia é drasticamente limitada por uma falha fundamental no design: **a busca inicial por tabelas é semanticamente "cega" ao conteúdo real das colunas.**

O ranqueamento principal de tabelas é feito comparando a consulta do usuário contra vetores que representam apenas os **nomes** da tabela e de suas colunas. A análise do conteúdo só ocorre em uma segunda etapa, sobre as tabelas já ranqueadas. Isso leva a resultados de baixa relevância, como visto na busca por "pré-eclâmpsia", onde tabelas contextualmente pobres (`tb_tipo_situacao_moradia`) superam tabelas ricas (`tb_fat_cad_individual`), simplesmente porque as últimas não possuem os termos da busca em seus metadados (nomes de tabela/coluna).

## 2. Análise Detalhada da Arquitetura e Fluxo

O processo atual ocorre em duas macro-etapas:

**Etapa 1: Ranqueamento de Tabelas**
1.  A consulta (ex: "pré-eclâmpsia") é expandida para termos relacionados usando BioWordVec (`['placenta', 'natimorto', ...]`).
2.  Um AG (`otimizador_ga`) otimiza os pesos desses termos para criar um vetor de consulta final (`V_QUERY_FINAL`).
3.  Este vetor é comparado (via similaridade de cosseno) com vetores pré-calculados em `v_tabelas.npy`.
4.  **PONTO CRÍTICO:** O arquivo `v_tabelas.npy`, gerado por `etapa1_script1_vetorizar_tabelas.py`, contém uma representação vetorial de um "documento" que é apenas a concatenação do **nome da tabela com os nomes de suas colunas**.

Isso significa que a tabela `tb_fat_cad_individual`, que contém as colunas `st_hipertensao_arterial` e `st_diabete` (altamente relevantes para "pré-eclâmpsia"), não será bem ranqueada porque nenhum desses nomes de metadados se parece semanticamente com "pré-eclâmpsia" ou seus termos expandidos.

**Etapa 2: Seleção de Features nas Tabelas Ranqueadas**
1.  O sistema pega as N tabelas mais bem ranqueadas da Etapa 1.
2.  Para cada uma delas, um segundo AG (`otimizador_ga_features`) é executado para encontrar as colunas mais relevantes para a consulta.

Esta etapa é poderosa, mas inútil se as tabelas corretas nunca chegarem até ela. Ela é capaz de encontrar as "agulhas" (`st_hipertensao_arterial`), mas somente se ela já estiver procurando no "palheiro" certo (`tb_fat_cad_individual`).

## 3. Análise dos Resultados para "pré-eclâmpsia"

A discrepância entre os resultados do sistema atual e os do sistema de referência baseado em LLM ilustra perfeitamente a falha:

-   **Sistema Atual:** Retorna `tb_processo`, `tb_tipo_paridade`, `tb_cronicidade`. São tabelas com nomes vagamente relacionados à área da saúde ou a processos genéricos, mas sem conexão direta e forte com a condição clínica da pré-eclâmpsia.
-   **Sistema de Referência (LLM):** Retorna `ta_exame_colesterol_hdl`, `tb_fat_cad_individual`, `tb_fat_visita_domiciliar`. Por quê? Porque ele analisou o **conteúdo ou o propósito** das colunas e entendeu que `vl_colesterol_hdl`, `st_hipertensao_arterial`, e `nu_medicao_pressao_arterial` são semanticamente ligados à pré-eclâmpsia, que é uma forma de hipertensão gestacional com implicações cardiovasculares.

O sistema atual não tem a capacidade de fazer essa conexão na sua etapa mais crucial de ranqueamento.

## 4. Análise do Algoritmo Genético (GA)

Existem dois AGs, e ambos são peças importantes, mas seu potencial está sendo subutilizado.

-   **AG de Otimização da Consulta (`otimizador_ga.py`):**
    -   **Qualidade do Fitness (`best_fitness=20.244692`):** O valor absoluto do fitness não é, por si só, um indicador de qualidade. Ele representa o score máximo de similaridade que o AG conseguiu obter entre a consulta ponderada e as tabelas-alvo.
    -   **Análise:** O AG está funcionando. Ele aprendeu a dar mais peso a termos como "pré-eclâmpsia" (36%), "não gravidez" (18.5%), "iugr" (7.4%), que são semanticamente relevantes. O problema é que ele está otimizando uma excelente consulta para ser usada contra um **conjunto de alvos (vetores de tabela) de baixa qualidade**. A estagnação do fitness a partir da geração 32 indica que o AG convergiu para a melhor solução possível *dado o conjunto de dados de entrada*, mas o teto de qualidade é baixo devido à pobreza dos vetores das tabelas.

-   **AG de Seleção de Features (`otimizador_ga_features.py`):**
    -   **Análise:** Este AG é conceitualmente correto e funciona bem. Ele consegue identificar as colunas mais importantes *dentro de uma única tabela*. No entanto, como mencionado, ele opera tarde demais no fluxo. Sua utilidade será massivamente amplificada quando ele passar a trabalhar sobre tabelas que são genuinamente relevantes.

## 5. Conclusão

A arquitetura do sistema é o seu principal ponto fraco. A separação rígida entre "ranqueamento por metadados" e "análise de conteúdo" impede que a riqueza semântica dos dados seja aproveitada no momento mais importante.

**A melhoria mais impactante, e necessária, é reestruturar o processo de busca para que a relevância das colunas seja o fator principal na determinação do ranking inicial das tabelas.** O Algoritmo Genético, uma vez alimentado com alvos de alta qualidade, terá seu desempenho e utilidade significativamente melhorados.

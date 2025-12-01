# SAÍDA NO CONSOLE:
2025-12-01 09:37:17,580 - INFO - Inicializando o BuscadorSemantico...
2025-12-01 09:37:17,581 - INFO - Usando dispositivo: cpu
2025-12-01 09:37:17,581 - INFO - Carregando modelo BERT principal (pt): pucpr/biobertpt-all …
2025-12-01 09:37:20,404 - INFO - ✅ Modelo BERT principal carregado com sucesso.
2025-12-01 09:37:20,405 - INFO - Carregando modelo BERT inglês: dmis-lab/biobert-base-cased-v1.1 …
2025-12-01 09:37:22,607 - INFO - ✅ Modelo BERT inglês carregado com sucesso.
2025-12-01 09:37:22,607 - INFO - 🧬 Carregando modelo W2V (BioWordVec reduzido) de /home/denilsonbj/projetos/query-cnat/modelos/biowordvec_500k.kv …
2025-12-01 09:37:22,607 - INFO - loading KeyedVectors object from /home/denilsonbj/projetos/query-cnat/modelos/biowordvec_500k.kv
2025-12-01 09:37:22,781 - INFO - loading vectors from /home/denilsonbj/projetos/query-cnat/modelos/biowordvec_500k.kv.vectors.npy with mmap=r
2025-12-01 09:37:22,784 - INFO - KeyedVectors lifecycle event {'fname': '/home/denilsonbj/projetos/query-cnat/modelos/biowordvec_500k.kv', 'datetime': '2025-12-01T09:37:22.783273', 'gensim': '4.4.0', 'python': '3.12.3 (main, Nov  6 2025, 13:44:16) [GCC 13.3.0]', 'platform': 'Linux-6.6.87.1-microsoft-standard-WSL2-x86_64-with-glibc2.39', 'event': 'loaded'}
2025-12-01 09:37:22,784 - INFO - ✅ W2V carregado em 0.18s (vocab: 500,000)
2025-12-01 09:37:22,785 - INFO - Carregando vetores das tabelas de /home/denilsonbj/projetos/query-cnat/v_tabelas.npy...
2025-12-01 09:37:22,791 - INFO - Carregando índice de /home/denilsonbj/projetos/query-cnat/tabelas_index.json...
2025-12-01 09:37:22,792 - INFO - Vetores das tabelas (shape: (1100, 768)) e índices carregados.
2025-12-01 09:37:22,793 - INFO - BuscadorSemantico pronto para uso.
2025-12-01 09:38:10,790 - INFO - 🌐 Traduzindo consulta (PT→EN): 'pré-eclâmpsia' → 'preeclampsia'
2025-12-01 09:38:10,790 - INFO - Gerando candidatos W2V para 'preeclampsia' (topn=300)...
2025-12-01 09:38:11,923 - INFO - 270 candidatos após filtragem lexical/diversidade.
2025-12-01 09:38:11,924 - INFO - Calculando scores híbridos (W2V-cosine, co-occurrence, contextual BERT)...
2025-12-01 09:38:56,237 - INFO - Expandido 'preeclampsia' para 20 termos (incluindo query).
2025-12-01 09:38:56,237 - INFO - 🌐 Traduzindo termos expandidos EN→PT para exibição e uso no pipeline...
2025-12-01 09:39:09,501 - INFO - ✅ Tradução concluída.
2025-12-01 09:39:09,503 - INFO - Vetorizando 20 termos candidatos com BERT (PT)...
2025-12-01 09:39:10,100 - INFO - GAOptimizer inicializado com 20 candidatos e 1100 tabelas
2025-12-01 09:39:10,100 - INFO - --- Iniciando GA ---
2025-12-01 09:39:11,415 - INFO - [GA] Geração 1/50 — best_fitness=15.187220
2025-12-01 09:39:12,217 - INFO - [GA] Geração 2/50 — best_fitness=16.239406
2025-12-01 09:39:13,309 - INFO - [GA] Geração 3/50 — best_fitness=17.862115
2025-12-01 09:39:14,405 - INFO - [GA] Geração 4/50 — best_fitness=17.862115
2025-12-01 09:39:15,109 - INFO - [GA] Geração 5/50 — best_fitness=18.712097
2025-12-01 09:39:16,307 - INFO - [GA] Geração 6/50 — best_fitness=18.801989
2025-12-01 09:39:17,653 - INFO - [GA] Geração 7/50 — best_fitness=19.291123
2025-12-01 09:39:18,769 - INFO - [GA] Geração 8/50 — best_fitness=19.291123
2025-12-01 09:39:19,893 - INFO - [GA] Geração 9/50 — best_fitness=19.291123
2025-12-01 09:39:21,007 - INFO - [GA] Geração 10/50 — best_fitness=19.361211
2025-12-01 09:39:22,034 - INFO - [GA] Geração 11/50 — best_fitness=19.564403
2025-12-01 09:39:23,081 - INFO - [GA] Geração 12/50 — best_fitness=19.564403
2025-12-01 09:39:24,152 - INFO - [GA] Geração 13/50 — best_fitness=19.564403
2025-12-01 09:39:25,208 - INFO - [GA] Geração 14/50 — best_fitness=19.731282
2025-12-01 09:39:26,280 - INFO - [GA] Geração 15/50 — best_fitness=19.731282
2025-12-01 09:39:27,572 - INFO - [GA] Geração 16/50 — best_fitness=19.768267
2025-12-01 09:39:28,770 - INFO - [GA] Geração 17/50 — best_fitness=19.852696
2025-12-01 09:39:29,875 - INFO - [GA] Geração 18/50 — best_fitness=19.891313
2025-12-01 09:39:30,914 - INFO - [GA] Geração 19/50 — best_fitness=20.073782
2025-12-01 09:39:31,954 - INFO - [GA] Geração 20/50 — best_fitness=20.136885
2025-12-01 09:39:33,046 - INFO - [GA] Geração 21/50 — best_fitness=20.136885
2025-12-01 09:39:34,175 - INFO - [GA] Geração 22/50 — best_fitness=20.136885
2025-12-01 09:39:35,311 - INFO - [GA] Geração 23/50 — best_fitness=20.136885
2025-12-01 09:39:36,456 - INFO - [GA] Geração 24/50 — best_fitness=20.136885
2025-12-01 09:39:37,499 - INFO - [GA] Geração 25/50 — best_fitness=20.136885
2025-12-01 09:39:38,566 - INFO - [GA] Geração 26/50 — best_fitness=20.136885
2025-12-01 09:39:39,702 - INFO - [GA] Geração 27/50 — best_fitness=20.136885
2025-12-01 09:39:40,800 - INFO - [GA] Geração 28/50 — best_fitness=20.136885
2025-12-01 09:39:41,893 - INFO - [GA] Geração 29/50 — best_fitness=20.157016
2025-12-01 09:39:42,984 - INFO - [GA] Geração 30/50 — best_fitness=20.157016
2025-12-01 09:39:44,117 - INFO - [GA] Geração 31/50 — best_fitness=20.157016
2025-12-01 09:39:45,167 - INFO - [GA] Geração 32/50 — best_fitness=20.244692
2025-12-01 09:39:46,271 - INFO - [GA] Geração 33/50 — best_fitness=20.244692
2025-12-01 09:39:47,330 - INFO - [GA] Geração 34/50 — best_fitness=20.244692
2025-12-01 09:39:47,836 - INFO - [GA] Geração 35/50 — best_fitness=20.244692
2025-12-01 09:39:48,892 - INFO - [GA] Geração 36/50 — best_fitness=20.244692
2025-12-01 09:39:49,995 - INFO - [GA] Geração 37/50 — best_fitness=20.244692
2025-12-01 09:39:51,169 - INFO - [GA] Geração 38/50 — best_fitness=20.244692
2025-12-01 09:39:52,222 - INFO - [GA] Geração 39/50 — best_fitness=20.244692
2025-12-01 09:39:53,280 - INFO - [GA] Geração 40/50 — best_fitness=20.244692
2025-12-01 09:39:54,409 - INFO - [GA] Geração 41/50 — best_fitness=20.244692
2025-12-01 09:39:55,449 - INFO - [GA] Geração 42/50 — best_fitness=20.244692
2025-12-01 09:39:56,579 - INFO - [GA] Geração 43/50 — best_fitness=20.244692
2025-12-01 09:39:57,796 - INFO - [GA] Geração 44/50 — best_fitness=20.244692
2025-12-01 09:39:58,871 - INFO - [GA] Geração 45/50 — best_fitness=20.244692
2025-12-01 09:40:00,018 - INFO - [GA] Geração 46/50 — best_fitness=20.244692
2025-12-01 09:40:01,115 - INFO - [GA] Geração 47/50 — best_fitness=20.244692
2025-12-01 09:40:02,328 - INFO - [GA] Geração 48/50 — best_fitness=20.244692
2025-12-01 09:40:03,686 - INFO - [GA] Geração 49/50 — best_fitness=20.244692
2025-12-01 09:40:04,786 - INFO - [GA] Geração 50/50 — best_fitness=20.244692
2025-12-01 09:40:04,787 - INFO - GA concluído em 54.68s
2025-12-01 09:40:05.660 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:05.725 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:05,746 - INFO - Vetorizando 3 nomes de colunas com BERT (PT)...
2025-12-01 09:40:05,902 - INFO - --- Iniciando GA para Seleção de Features (3 colunas) ---
2025-12-01 09:40:07,417 - INFO - GA de Features concluído em 1.51s
2025-12-01 09:40:07.445 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:07,448 - INFO - Vetorizando 3 nomes de colunas com BERT (PT)...
2025-12-01 09:40:07,573 - INFO - --- Iniciando GA para Seleção de Features (3 colunas) ---
2025-12-01 09:40:08,923 - INFO - GA de Features concluído em 1.35s
2025-12-01 09:40:08.943 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:08,947 - INFO - Vetorizando 2 nomes de colunas com BERT (PT)...
2025-12-01 09:40:09,037 - INFO - --- Iniciando GA para Seleção de Features (2 colunas) ---
2025-12-01 09:40:09,851 - INFO - GA de Features concluído em 0.81s
2025-12-01 09:40:09.867 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:09,870 - INFO - Vetorizando 3 nomes de colunas com BERT (PT)...
2025-12-01 09:40:09,998 - INFO - --- Iniciando GA para Seleção de Features (3 colunas) ---
2025-12-01 09:40:11,613 - INFO - GA de Features concluído em 1.61s
2025-12-01 09:40:11.648 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:11,652 - INFO - Vetorizando 2 nomes de colunas com BERT (PT)...
2025-12-01 09:40:11,754 - INFO - --- Iniciando GA para Seleção de Features (2 colunas) ---
2025-12-01 09:40:12,509 - INFO - GA de Features concluído em 0.75s
2025-12-01 09:40:12.524 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:12,526 - INFO - Vetorizando 2 nomes de colunas com BERT (PT)...
2025-12-01 09:40:12,619 - INFO - --- Iniciando GA para Seleção de Features (2 colunas) ---
2025-12-01 09:40:13,369 - INFO - GA de Features concluído em 0.75s
2025-12-01 09:40:13.384 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:13,386 - INFO - Vetorizando 2 nomes de colunas com BERT (PT)...
2025-12-01 09:40:13,473 - INFO - --- Iniciando GA para Seleção de Features (2 colunas) ---
2025-12-01 09:40:14,227 - INFO - GA de Features concluído em 0.75s
2025-12-01 09:40:14.244 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:14,246 - INFO - Vetorizando 2 nomes de colunas com BERT (PT)...
2025-12-01 09:40:14,333 - INFO - --- Iniciando GA para Seleção de Features (2 colunas) ---
2025-12-01 09:40:15,070 - INFO - GA de Features concluído em 0.73s
2025-12-01 09:40:15.084 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:15,086 - INFO - Vetorizando 2 nomes de colunas com BERT (PT)...
2025-12-01 09:40:15,174 - INFO - --- Iniciando GA para Seleção de Features (2 colunas) ---
2025-12-01 09:40:15,923 - INFO - GA de Features concluído em 0.75s
2025-12-01 09:40:15.934 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:15,937 - INFO - Vetorizando 3 nomes de colunas com BERT (PT)...
2025-12-01 09:40:16,073 - INFO - --- Iniciando GA para Seleção de Features (3 colunas) ---
2025-12-01 09:40:17,249 - INFO - GA de Features concluído em 1.17s
2025-12-01 09:40:17.265 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:17,268 - INFO - Vetorizando 3 nomes de colunas com BERT (PT)...
2025-12-01 09:40:17,359 - INFO - --- Iniciando GA para Seleção de Features (3 colunas) ---
2025-12-01 09:40:18,908 - INFO - GA de Features concluído em 1.55s
2025-12-01 09:40:18.940 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:18,942 - INFO - Vetorizando 3 nomes de colunas com BERT (PT)...
2025-12-01 09:40:18,517 - INFO - --- Iniciando GA para Seleção de Features (3 colunas) ---
2025-12-01 09:40:19,732 - INFO - GA de Features concluído em 1.21s
2025-12-01 09:40:19.754 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:19,756 - INFO - Vetorizando 2 nomes de colunas com BERT (PT)...
2025-12-01 09:40:19,853 - INFO - --- Iniciando GA para Seleção de Features (2 colunas) ---
2025-12-01 09:40:20,627 - INFO - GA de Features concluído em 0.77s
2025-12-01 09:40:20.641 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:20,643 - INFO - Vetorizando 5 nomes de colunas com BERT (PT)...
2025-12-01 09:40:20,737 - INFO - --- Iniciando GA para Seleção de Features (5 colunas) ---
2025-12-01 09:40:22,655 - INFO - GA de Features concluído em 1.91s
2025-12-01 09:40:22.695 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
2025-12-01 09:40:22,698 - INFO - Vetorizando 1 nomes de colunas com BERT (PT)...
2025-12-01 09:40:22,754 - INFO - --- Iniciando GA para Seleção de Features (1 colunas) ---
2025-12-01 09:40:23,008 - INFO - GA de Features concluído em 0.25s
2025-12-01 09:40:23.012 Please replace `use_container_width` with `width`.

`use_container_width` will be removed after 2025-12-31.

For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.

---

# SAÍDA NA INTERFACE DO STREAMLIT:
Motor de Busca Semântica Otimizada por AG 🧬
Carregando modelos (BERT, W2V) e vetores de tabela... (só na primeira vez)

Buscador carregado em 5.21s

Carregando metadados das tabelas...

Metadados carregados.

Faça sua consulta
Termo de busca (ex: pre-eclampsia):

pré-eclâmpsia
Nº de termos para expansão (N do AG):

3
20
Nº de tabelas para analisar features:

1
20

Resultados da Etapa 1: Ranking de Tabelas para 'pré-eclâmpsia'
1. Expandindo consulta 'pré-eclâmpsia' com W2V...

Termos Candidatos: ['pré-eclâmpsia', 'induzida pela gravidez', 'descolamento', 'gestose', 'aborto espontâneo', 'ajuda', 'macrossomia', 'não gravidez', 'pré-gestacional', 'placenta', 'pré-parto', 'trimestre', 'pih', 'iugr', 'corioamnionite', 'primigestas', 'natimorto', 'fetopatia', 'pprom', 'prematuridade']

2. Vetorizando 20 termos com BERT...

3. Executando Algoritmo Genético (AG) para otimizar pesos...

AG concluído! Pesos otimizados encontrados.

4. Criando V_QUERY_FINAL (Consulta Otimizada)...

5. Calculando ranking final...

Pesos da Consulta Otimizados pelo AG
O AG aprendeu a importância de cada termo para esta busca:

0	pré-eclâmpsia	36.30%
7	não gravidez	18.51%
5	ajuda	8.00%
13	iugr	7.42%
1	induzida pela gravidez	5.40%
16	natimorto	4.59%
18	pprom	3.42%
14	corioamnionite	2.93%
3	gestose	2.30%
12	pih	2.17%
15	primigestas	1.80%
6	macrossomia	1.40%
4	aborto espontâneo	1.23%
2	descolamento	1.19%
9	placenta	0.92%
8	pré-gestacional	0.79%
17	fetopatia	0.49%
19	prematuridade	0.48%
11	trimestre	0.46%
10	pré-parto	0.21%

Ranking Final das Tabelas
Tabelas mais relevantes, ordenadas pela consulta otimizada.

0	tb_processo	0.7802
1	tb_tipo_paridade	0.7701
2	tb_cronicidade	0.7621
3	tb_tipo_gravidez	0.7615
4	tb_gravidade	0.7588
5	tb_tipo_ciap	0.7491
6	tb_racionalidade_saude	0.7466
7	tb_tipo_situacao_moradia	0.7454
8	tb_tipo_dado_transp	0.7454
9	tb_complexidade	0.7449
10	tb_tema_saude	0.7424
11	tb_etnia	0.7391
12	tb_situacao_lote_transp_nodo	0.7384
13	tl_tipo_gravidez	0.7356
14	tb_ator	0.7345
15	tb_tipo_consulta_odonto	0.7300
16	tb_tipo_encam_odonto	0.7285
17	tb_situacao_raiz	0.7281
18	rl_proced_cds_proced	0.7275
19	tb_situacao_dado_recebido	0.7271
20	tb_pratica_saude	0.7256
21	tb_situacao_problema	0.7252
22	tb_tipo_parto	0.7239
23	tb_estado_civil	0.7237
24	tb_parte_bucal_proced	0.7235
25	tb_inep	0.7221
26	tb_sexo	0.7203
27	tb_tipo_opcao	0.7193
28	tl_tipo_encam_odonto	0.7185
29	tb_proced_filtro	0.7184
30	tb_tema_reuniao	0.7180
31	tl_parte_bucal_proced	0.7178
32	tb_classificacao_prioridade_cc	0.7176
33	tb_tipo_edema	0.7173
34	tb_tipo_localizacao	0.7173
35	tl_parte_bucal	0.7170
36	rl_unidade_saude_horus	0.7159
37	tb_status_revisao_atend	0.7147
38	tb_evolucao_plano	0.7136
39	tb_motivo_reserva	0.7133
40	tb_escolaridade	0.7125
41	tb_situacao_face	0.7123
42	tb_cds_tipo_vig_saude_bucal	0.7120
43	tl_unidade_saude_horus	0.7114
44	tb_situacao_coroa	0.7114
45	tb_tipo_agravo	0.7112
46	tb_grau_parentesco	0.7108
47	tb_grupo_condicao_saude	0.7108
48	tb_atributo_complem	0.7108
49	tb_tipo_glicemia	0.7104

Análise de Features (Etapa 2)
Analisando as colunas das 15 tabelas mais relevantes...

Tabela: tb_processo (Similaridade da Tabela: 0.7802)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_processo

Schema: public

Row Count: 49

Score de Relevância das Features: 0.6152

Chave Primária: co_seq_processo

Justificativa: Score 0.6152 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas dt_inicio contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tb_tipo_paridade (Similaridade da Tabela: 0.7701)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_tipo_paridade

Schema: public

Row Count: 5

Score de Relevância das Features: 0.5472

Chave Primária: co_tipo_paridade

Justificativa: Score 0.5472 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas no_tipo_paridade contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tb_cronicidade (Similaridade da Tabela: 0.7621)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_cronicidade

Schema: public

Row Count: 3

Score de Relevância das Features: 0.5985

Chave Primária: co_cronicidade

Justificativa: Score 0.5985 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas co_cronicidade contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tb_tipo_gravidez (Similaridade da Tabela: 0.7615)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_tipo_gravidez

Schema: public

Row Count: 4

Score de Relevância das Features: 0.5498

Chave Primária: co_tipo_gravidez

Justificativa: Score 0.5498 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas no_tipo_gravidez, no_identificador contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tb_gravidade (Similaridade da Tabela: 0.7588)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_gravidade

Schema: public

Row Count: 3

Score de Relevância das Features: 0.5471

Chave Primária: co_gravidade

Justificativa: Score 0.5471 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas co_gravidade contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tb_tipo_ciap (Similaridade da Tabela: 0.7491)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_tipo_ciap

Schema: public

Row Count: 7

Score de Relevância das Features: 0.5454

Chave Primária: co_tipo_ciap

Justificativa: Score 0.5454 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas no_tipo_ciap contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tb_racionalidade_saude (Similaridade da Tabela: 0.7466)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_racionalidade_saude

Schema: public

Row Count: 6

Score de Relevância das Features: 0.5995

Chave Primária: co_racionalidade_saude

Justificativa: Score 0.5995 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas co_racionalidade_saude contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tb_tipo_situacao_moradia (Similaridade da Tabela: 0.7454)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_tipo_situacao_moradia

Schema: public

Row Count: 9

Score de Relevância das Features: 0.5707

Chave Primária: co_tipo_situacao_moradia

Justificativa: Score 0.5707 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas co_tipo_situacao_moradia contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tb_tipo_dado_transp (Similaridade da Tabela: 0.7454)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_tipo_dado_transp

Schema: public

Row Count: 16

Score de Relevância das Features: 0.5421

Chave Primária: co_tipo_dado_transp

Justificativa: Score 0.5421 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas no_tipo_dado_transp contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tb_complexidade (Similaridade da Tabela: 0.7449)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_complexidade

Schema: public

Row Count: 7

Score de Relevância das Features: 0.6073

Chave Primária: co_complexidade

Justificativa: Score 0.6073 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas sg_complexidade contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tb_tema_saude (Similaridade da Tabela: 0.7424)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_tema_saude

Schema: public

Row Count: 18

Score de Relevância das Features: 0.5934

Chave Primária: co_tema_saude

Justificativa: Score 0.5934 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas co_tema_saude, no_identificador contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tb_etnia (Similaridade da Tabela: 0.7391)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_etnia

Schema: public

Row Count: 406

Score de Relevância das Features: 0.5760

Chave Primária: co_etnia

Justificativa: Score 0.5760 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas co_etnia_cadsus contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tb_situacao_lote_transp_nodo (Similaridade da Tabela: 0.7384)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_situacao_lote_transp_nodo

Schema: public

Row Count: 4

Score de Relevância das Features: 0.5496

Chave Primária: co_situacao_lote_transp_nodo

Justificativa: Score 0.5496 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas co_situacao_lote_transp_nodo contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tl_tipo_gravidez (Similaridade da Tabela: 0.7356)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tl_tipo_gravidez

Schema: public

Row Count: 0

Score de Relevância das Features: 0.4412

Chave Primária: co_revisao, co_tipo_gravidez

Justificativa: Score 0.4412 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas co_revisao, co_tipo_gravidez contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

Tabela: tb_ator (Similaridade da Tabela: 0.7345)

Analisando colunas...

Executando Algoritmo Genético para selecionar as melhores colunas...

Tabela: tb_ator

Schema: public

Row Count: 4,151

Score de Relevância das Features: 0.5712

Chave Primária: co_seq_ator

Justificativa: Score 0.5712 devido à alta similaridade semântica com a consulta e boa qualidade das colunas selecionadas. As colunas co_seq_ator contribuem significativamente para a relevância.

Colunas Contribuintes para o Score:

---

# QUESTÃO DE QUALIDADE

Toda a seleção de colunas e features é baseado em "semelhança semântica" dos termos pesquisados. Porém os resultados não estão tão bons quanto poderiam.

Por exemplo, para o termo pesquisado "pré-eclâmpsia", era possivel trazer resultados como "hipertensão", "colesterol", etc - isso são apenas exemplos. Porém como um dos retornos tive algo tabelas como:
tb_processo	0.7802
tb_racionalidade_saude	0.7466
tb_tipo_situacao_moradia	0.7454
tb_tipo_dado_transp	0.7454
...

Sinto que semanticamente não é o melhor. Outro ponto também a se trabalhar é em relação a qualidade do Algoritmo Genetico - que tem uma importância altissima para o desenvolvimento desse projeto. Essa saída de Melhor Fitness ao final das gerações é boa mesmo(?):
2025-12-01 09:40:04,786 - INFO - [GA] Geração 50/50 — best_fitness=20.244692

Se não for boa, trabalhe para melhorar.

Outro ponto também é que rodei um sistema semelhante de busca semântica buscando nesse mesmo metadado de bases de saúde, tabelas e features relevantes para a busca "pré-eclâmpsia". Esse outro sistema foi construído e rodado usando APIs de LLMs; as principais tabelas e features retornadas, usando a API da OpenAI, com a busca "pré-eclâmpsia", foram:
---
# SAÍDA DE UM SISTEMA SEMELHANTE QUE USA LLM PARA BUSCA

Detalhamento das Tabelas de Alta Relevância
1. ta_exame_colesterol_hdl
Schema: public
Row Count: 1,382
Score de Relevância: 85
Chave Primária: co_seq_taexamecolesterolhdl
Justificativa: Score 85 devido à presença de dados sobre colesterol HDL, um fator crítico na detecção de doenças cardiovasculares. A tabela possui alta completude e dados numéricos relevantes.

Colunas Contribuintes para o Score:

vl_colesterol_hdl
2. ta_exame_colesterol_ldl
Schema: public
Row Count: 1,358
Score de Relevância: 85
Chave Primária: co_seq_taexamecolesterolldl
Justificativa: Score 85 devido à presença da coluna 'vl_colesterol_ldl', que é crucial para a detecção de doenças cardiovasculares. A tabela possui alta completude e dados relevantes para análise.

Colunas Contribuintes para o Score:

vl_colesterol_ldl
3. ta_exame_colesterol_total
Schema: public
Row Count: 2,530
Score de Relevância: 85
Chave Primária: co_seq_taexamecolesteroltotal
Justificativa: Score 85 devido à presença da coluna 'vl_colesterol_total', que é crucial para a detecção de doenças cardiovasculares. A tabela possui alta completude e dados relevantes para análise de risco.

Colunas Contribuintes para o Score:

vl_colesterol_total
4. tb_exame_colesterol_hdl
Schema: public
Row Count: 1,886
Score de Relevância: 85
Chave Primária: co_seq_exame_colesterol_hdl
Justificativa: Score 85 devido à presença de dados sobre colesterol HDL, um fator de risco importante para doenças cardiovasculares. A tabela possui alta completude e dados relevantes para a análise de saúde cardiovascular.

Colunas Contribuintes para o Score:

vl_colesterol_hdl
5. tb_exame_colesterol_ldl
Schema: public
Row Count: 1,840
Score de Relevância: 85
Chave Primária: co_seq_exame_colesterol_ldl
Justificativa: Score 85 devido à presença de dados sobre colesterol LDL, um fator de risco importante para doenças cardiovasculares. A tabela possui alta completude e dados numéricos relevantes.

Colunas Contribuintes para o Score:

vl_colesterol_ldl
6. tb_exame_colesterol_total
Schema: public
Row Count: 3,660
Score de Relevância: 85
Chave Primária: co_seq_exame_colesterol_total
Justificativa: Score 85 devido à presença de dados sobre colesterol total, um fator de risco importante para doenças cardiovasculares. A tabela possui alta completude e dados relevantes para a análise de saúde cardiovascular.

Colunas Contribuintes para o Score:

vl_colesterol_total
7. tb_fat_cad_individual
Schema: public
Row Count: 235,405
Score de Relevância: 85
Chave Primária: co_seq_fat_cad_individual
Justificativa: Score 85 devido à presença de colunas que indicam fatores de risco cardiovascular, como hipertensão e diabetes. A tabela possui alta completude e diversidade de dados relevantes.

Colunas Contribuintes para o Score:

st_hipertensao_arterial
st_diabete
st_doenca_cardiaca
st_fumante
st_alcool
8. tb_fat_rel_op_risco_cardio
Schema: public
Row Count: 27,691
Score de Relevância: 85
Chave Primária: co_seq_fat_rel_op_risco_cardio
Justificativa: Score 85 devido à presença de colunas relevantes como hipertensão, diabetes e obesidade, que são fatores de risco para doenças cardiovasculares. A tabela possui uma boa completude em dados relevantes, apesar de algumas colunas com alta taxa de nulos.

Colunas Contribuintes para o Score:

dt_hipertensao_arterial_fai
dt_diabetes_fai
dt_obesidade_fai
st_risco_cardio
9. tb_fat_visita_domiciliar
Schema: public
Row Count: 975,591
Score de Relevância: 85
Chave Primária: co_seq_fat_visita_domiciliar
Justificativa: Score 85 devido à presença de colunas que indicam hipertensão e diabetes, fatores de risco para doenças cardiovasculares. A tabela possui alta completude e um grande número de registros.

Colunas Contribuintes para o Score:

st_acomp_pessoa_hipertensao
st_acomp_pessoa_diabetes
nu_medicao_pressao_arterial
10. tl_exame_colesterol_ldl
Schema: public
Row Count: 465
Score de Relevância: 85
Chave Primária: NÃO ENCONTRADA
Justificativa: Score 85 devido à presença de dados sobre colesterol LDL, um fator de risco importante para doenças cardiovasculares. A tabela possui alta completude e dados relevantes para a análise de saúde cardiovascular.

Colunas Contribuintes para o Score:

vl_colesterol_ldl
co_exame_requisitado
11. tl_exame_colesterol_total
Schema: public
Row Count: 1,100
Score de Relevância: 85
Chave Primária: NÃO ENCONTRADA
Justificativa: Score 85 devido à presença do campo 'vl_colesterol_total', que é crucial para a detecção de doenças cardiovasculares. A tabela possui alta completude e dados relevantes para a análise de risco cardiovascular.

Colunas Contribuintes para o Score:

vl_colesterol_total

---

Estou enviando isso não porque estou dizendo que o resultado desse sistema precisa ser igual, não precisa necessariamente (a não ser que de fato esses sejam os melhores resultados), mas estou falando sobre a relação entre as tabelas e colunas retornadas que são semelhantes a pesquisa feita.

---

# O que preciso:
Preciso que você analise todo o projeto e todos esses resultados que enviei; analise se a saída do Algoritmo Genetico foi boa ou se é possivel melhorá-la; analise se a busca de tabela é baseada somente no nome das tabelas ou no nome das tabelas e das colunas associadas a tabela, e só depois as "principais colunas que contribuiram para a seleção da tabela são destacadas" - pois, por exemplo, a tabela "tb_fat_visita_domiciliar" aparenta não ser relevante, porém suas principais colunas que contribuiram para deixá-la num ranking elevado foram:
Colunas Contribuintes para o Score:
st_acomp_pessoa_hipertensao
st_acomp_pessoa_diabetes
nu_medicao_pressao_arterial

A Arquitetura já está montada, o foco agora é melhorar a qualdidade dos resultados.

O ALGORITMO GENETICO precisa ter uma participação central no sistema e estar rodando na mais alta qualidade possível.

Quero que monte uma analise completa da qualidade do sistema e de seus resultados em um arquivo .md e depois monte um plano completo, robusto e detalhadado com passo a passo em todas as etapas, para melhorar a qualidade dos resultados desse sitema.
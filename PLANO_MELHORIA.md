# Plano de Melhoria para o Sistema de Busca Semântica

## Objetivo Principal
Reestruturar o núcleo do sistema de busca para que a relevância seja determinada pelo **conteúdo semântico das colunas**, e não apenas pelos metadados (nomes de tabelas/colunas). Isso irá melhorar drasticamente a qualidade dos resultados, mantendo e potencializando o uso dos Algoritmos Genéticos.

---

## Fase 1: Enriquecimento Semântico dos Vetores de Busca (Ação Mais Crítica)
Esta fase corrige a falha central do sistema. A ideia é mudar a unidade de busca: em vez de procurar "tabelas", vamos procurar "colunas" e, em seguida, agregar a relevância de volta para a tabela.

### **Passo 1.1: Mudar a Estratégia de Vetorização (De Tabela para Coluna)**
- **O quê:** Modificar o script `etapa1/etapa1_script1_vetorizar_tabelas.py`.
- **Como:**
    1. O script não irá mais gerar um vetor por tabela. Ele deverá iterar sobre **cada coluna de cada tabela** presente no `asset/metadata_advanced_consolidated.json`.
    2. Para cada coluna, ele criará um "documento de coluna" representativo. Este documento deve ser uma string contendo:
        - Nome da tabela.
        - Nome da coluna.
        - Descrição da coluna (se disponível no metadado).
        - **Amostra de dados distintos da coluna:** Para colunas do tipo string, pegar os 5-10 valores únicos mais frequentes. Para colunas numéricas, pode-se converter em texto termos como "valores numéricos de colesterol". Isso é crucial para capturar a semântica do conteúdo.
    3. Use o modelo BERT (`pucpr/biobertpt-all`) para gerar um vetor de embedding para **cada um desses "documentos de coluna"**.

### **Passo 1.2: Gerar Novos Artefatos de Índice**
- **O quê:** Ao final do script modificado, salve dois novos arquivos:
    1. `v_colunas.npy`: Um array NumPy onde cada linha é o vetor de uma coluna específica.
    2. `colunas_index.json`: Um arquivo JSON que mapeia cada índice (linha) de `v_colunas.npy` para sua origem. Exemplo:
       ```json
       [
         { "index": 0, "table_name": "tb_fat_cad_individual", "column_name": "st_hipertensao_arterial" },
         { "index": 1, "table_name": "tb_fat_cad_individual", "column_name": "st_diabete" },
         ...
       ]
       ```

### **Passo 1.3: Adaptar o Mecanismo de Busca (`etapa2/busca_semantica.py`)**
- **O quê:** Modificar a classe `BuscadorSemantico` para usar os novos artefatos.
- **Como:**
    1. Na inicialização (`__init__`), carregue `v_colunas.npy` e `colunas_index.json` em vez de `v_tabelas.npy` e `tabelas_index.json`.
    2. Na função `ranking_por_similaridade`, a lógica de similaridade de cosseno agora será entre o `V_QUERY_FINAL` e todos os vetores de **colunas**. O resultado será um ranking de colunas.
    3. **Agregar Scores por Tabela:** Após obter os scores de todas as colunas, crie o ranking final de tabelas. A melhor abordagem para começar é usar a agregação por `max`: o score de uma tabela é o score da sua coluna mais relevante.
        - Crie um dicionário `{nome_tabela: 0.0}`.
        - Itere sobre os scores das colunas. Para cada coluna, atualize o score da sua tabela-mãe no dicionário: `scores_tabela[tabela] = max(scores_tabela[tabela], score_coluna)`.
        - O ranking final será este dicionário ordenado por valor.

**Resultado Esperado da Fase 1:** Uma busca por "pré-eclâmpsia" agora encontrará a coluna `st_hipertensao_arterial` com alta similaridade. O score dessa coluna será atribuído à tabela `tb_fat_cad_individual`, que, por sua vez, aparecerá no topo do ranking, como esperado.

---

## Fase 2: Otimização e Refinamento do Algoritmo Genético
Com a Fase 1 completa, os AGs se tornarão muito mais úteis.

### **Passo 2.1: Revisar a Função de Fitness (`otimizador_ga.py`)**
- **O quê:** A função de fitness atual provavelmente maximiza a similaridade do melhor resultado. Podemos torná-la mais robusta.
- **Como:** Experimente uma função que recompense a **separação** entre os melhores e os piores resultados. Por exemplo, a fitness pode ser `(média dos scores dos top 5 resultados) - (média dos scores dos resultados entre as posições 100-105)`. Isso incentivará o AG a encontrar pesos que criem um ranking mais "decidido".

### **Passo 2.2: Ajustar Hiperparâmetros do AG**
- **O quê:** Experimentar com os parâmetros em `otimizador_ga.py` e `otimizador_ga_features.py`.
- **Como:** Aumentar o `population_size` (ex: de 50 para 100) e o `n_generations` pode ajudar a evitar mínimos locais e encontrar soluções melhores, agora que o "espaço de busca" (os vetores-alvo) é de maior qualidade. Ajustar a `mutation_rate` também pode ser benéfico.

---

## Fase 3: Melhoria na Expansão da Consulta
### **Passo 3.1: Filtrar Termos Genéricos**
- **O quê:** A expansão de "pré-eclâmpsia" gerou o termo "ajuda", que é semanticamente pobre e genérico.
- **Como:** Em `busca_semantica.py`, após a expansão com W2V, adicione uma etapa de filtragem para remover stopwords ou termos excessivamente comuns de um vocabulário geral (não apenas do domínio médico).

---

## Fase 4: Validação e Métricas de Qualidade
Para medir objetivamente o sucesso das melhorias, é crucial formalizar a avaliação.

### **Passo 4.1: Criar um Conjunto de Validação ("Golden Set")**
- **O quê:** Crie um arquivo (ex: `evaluation_set.json`) com um conjunto de consultas e os resultados esperados.
- **Como:** Liste de 5 a 10 termos de busca relevantes para o domínio (ex: "diabetes", "infarto", "saúde bucal") e, para cada um, liste manualmente as 5-10 tabelas que um especialista consideraria mais importantes.

### **Passo 4.2: Implementar Métricas de Avaliação**
- **O quê:** Crie um script de avaliação (`test_quality.py`) que rode as buscas do "Golden Set" e calcule métricas de qualidade.
- **Como:** Boas métricas para sistemas de ranking são:
    - **Mean Reciprocal Rank (MRR):** Quão perto do topo do ranking está o *primeiro* resultado relevante?
    - **Precision@K (ex: P@5):** Dos 5 primeiros resultados, quantos são relevantes?
    - **NDCG@K:** Considera a posição dos resultados relevantes, dando mais peso aos que aparecem no topo.

Rode este script após a Fase 1 e depois novamente após as outras fases para quantificar o ganho de qualidade.

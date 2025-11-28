# app.py (O Ponto de Entrada)

import streamlit as st
import pandas as pd
import numpy as np
import logging
import time

# Importa os módulos das etapas 2 e 3
from etapa2.busca_semantica import BuscadorSemantico
from etapa3.otimizador_ga import rodar_otimizacao_ga
from etapa2.metadata_loader import MetadataLoader
from etapa3.otimizador_ga_features import rodar_ga_feature_selection

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Funções de Cache ---

@st.cache_resource
def carregar_buscador():
    """
    Carrega o BuscadorSemantico uma única vez e o mantém em cache.
    Isso evita recarregar os modelos BERT e W2V a cada interação.
    """
    st.write("Carregando modelos (BERT, W2V) e vetores de tabela... (só na primeira vez)")
    start_time = time.time()
    try:
        buscador = BuscadorSemantico()
        end_time = time.time()
        st.write(f"Buscador carregado em {end_time - start_time:.2f}s")
        return buscador
    except Exception as e:
        st.error(f"Falha ao carregar os modelos: {e}")
        st.stop()

@st.cache_resource
def carregar_metadata_loader():
    """Carrega o MetadataLoader uma vez e o mantém em cache."""
    st.write("Carregando metadados das tabelas...")
    try:
        loader = MetadataLoader()
        st.write("Metadados carregados.")
        return loader
    except Exception as e:
        st.error(f"Falha ao carregar os metadados: {e}")
        st.stop()

def rodar_pipeline_features(buscador, loader, query, top_n_tables):
    """
    Executa o pipeline da Etapa 2: Seleção de Features para as tabelas do Top N.
    """
    st.header("Análise de Features (Etapa 2)")
    st.write(f"Analisando as colunas das {len(top_n_tables)} tabelas mais relevantes...")

    # Vetoriza a consulta original do usuário uma única vez
    user_query_vector = buscador._get_bert_embedding(query, lang='pt')
    
    results = []
    for table_name, table_score in top_n_tables:
        with st.expander(f"Tabela: **{table_name}** (Similaridade da Tabela: {table_score:.4f})"):
            st.write("Analisando colunas...")
            
            # 1. Obter metadados e nomes das colunas
            table_metadata = loader.get_table_metadata(table_name)
            if not table_metadata or 'columns' not in table_metadata:
                st.warning("Metadados da tabela não encontrados ou tabela sem colunas.")
                continue
            
            columns_metadata = table_metadata['columns']
            column_names = [col['name'] for col in columns_metadata]

            # 2. Vetorizar colunas
            with st.spinner(f"Vetorizando {len(column_names)} colunas..."):
                columns_vectors = buscador.vetorizar_colunas(column_names)

            # 3. Executar GA para seleção de features
            st.write("Executando Algoritmo Genético para selecionar as melhores colunas...")
            with st.spinner("O AG está aprendendo as melhores features..."):
                best_solution, best_fitness = rodar_ga_feature_selection(
                    user_query_vector=user_query_vector,
                    table_columns_metadata=columns_metadata,
                    table_columns_vectors=columns_vectors
                )

            # 4. Interpretar e exibir resultados
            if best_solution is None:
                st.warning("O GA não retornou uma solução.")
                continue

            selected_indices = np.where(best_solution == 1)[0]

            if len(selected_indices) == 0:
                st.warning("O GA não selecionou nenhuma coluna como relevante.")
                continue
            selected_columns = [column_names[i] for i in selected_indices]
            
            # --- Exibição detalhada conforme SEGUNDA-ETAPA.md ---
            st.markdown(f"**Tabela:** `{table_name}`")
            st.markdown(f"**Schema:** `{table_metadata.get('schema', 'N/A')}`")
            st.markdown(f"**Row Count:** `{table_metadata.get('row_count', 'N/A'):,}`")
            st.markdown(f"**Score de Relevância das Features:** `{best_fitness:.4f}`")
            primary_key = table_metadata.get('primary_key', ['NÃO ENCONTRADA'])
            st.markdown(f"**Chave Primária:** `{', '.join(primary_key)}`")

            # Gerar Justificativa (exemplo simplificado)
            justificativa = (
                f"Score {best_fitness:.4f} devido à alta similaridade semântica com a consulta "
                f"e boa qualidade das colunas selecionadas. "
                f"As colunas {', '.join(selected_columns[:3])}{'...' if len(selected_columns) > 3 else ''} "
                f"contribuem significativamente para a relevância."
            )
            st.markdown(f"**Justificativa:** {justificativa}")
            
            if selected_columns:
                st.markdown("**Colunas Contribuintes para o Score:**")
                # Usar um DataFrame para melhor visualização
                df_cols = pd.DataFrame({'Coluna Selecionada': selected_columns})
                st.dataframe(df_cols, use_container_width=True)
            else:
                st.warning("Nenhuma coluna foi considerada relevante o suficiente pelo AG.")

            # Guarda os resultados para possível uso futuro
            results.append({
                "table_name": table_name,
                "feature_relevance_score": best_fitness,
                "selected_columns": selected_columns,
                "schema": table_metadata.get('schema', 'N/A'),
                "row_count": table_metadata.get('row_count', 'N/A'),
                "primary_key": primary_key,
                "justificativa": justificativa
            })
    return results


def rodar_pipeline_busca(buscador, query_usuario, n_termos=10):
    """
    Executa o pipeline completo de busca (Etapas 1, 2 e 3 do contexto).
    """
    st.write(f"**1. Expandindo consulta '{query_usuario}' com W2V...**")
    termos_candidatos_tuplas = buscador.expandir_consulta(query_usuario, n=n_termos)
    termos_candidatos = [termo for termo, score in termos_candidatos_tuplas]
    st.write(f"Termos Candidatos: `{termos_candidatos}`")

    st.write(f"**2. Vetorizando {len(termos_candidatos)} termos com BERT...**")
    v_candidatos = buscador.vetorizar_termos_candidatos(termos_candidatos)

    st.write(f"**3. Executando Algoritmo Genético (AG) para otimizar pesos...**")
    
    # O AG otimiza os pesos para os V_CANDIDATOS
    with st.spinner('O AG está aprendendo a melhor consulta... (Isso pode levar um minuto)'):
        w_otimizado = rodar_otimizacao_ga(
            buscador,
            v_candidatos,
            buscador.vetores_tabelas,
            buscador.nomes_tabelas
        )

    st.write("AG concluído! Pesos otimizados encontrados.")

    st.write(f"**4. Criando V_QUERY_FINAL (Consulta Otimizada)...**")
    v_query_final = buscador.criar_vetor_consulta_ponderado(v_candidatos, w_otimizado)

    st.write(f"**5. Calculando ranking final...**")
    ranking_final = buscador.ranking_por_similaridade(v_query_final)
    
    return termos_candidatos, w_otimizado, ranking_final

# --- Interface Principal do Streamlit ---

st.set_page_config(layout="wide")
st.title("Motor de Busca Semântica Otimizada por AG 🧬")

# Carrega o buscador e o loader de metadados
buscador = carregar_buscador()
metadata_loader = carregar_metadata_loader()

st.header("Faça sua consulta")

# Entrada do usuário
query_usuario = st.text_input("Termo de busca (ex: pre-eclampsia):", "pre-eclampsia")
n_termos = st.slider("Nº de termos para expansão (N do AG):", min_value=3, max_value=20, value=10)
top_n_para_analise = st.slider("Nº de tabelas para analisar features:", min_value=1, max_value=20, value=5)


if st.button("Buscar Tabelas Relevantes"):
    if not query_usuario:
        st.error("Por favor, digite um termo de busca.")
    else:
        st.divider()
        st.header(f"Resultados da Etapa 1: Ranking de Tabelas para '{query_usuario}'")
        
        # Executa todo o pipeline
        termos, pesos, ranking = rodar_pipeline_busca(buscador, query_usuario, n_termos)
        
        st.divider()

        # --- Exibe os resultados em duas colunas ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pesos da Consulta Otimizados pelo AG")
            st.write("O AG aprendeu a importância de cada termo para esta busca:")
            
            # Prepara os dados dos pesos para exibição
            df_pesos_data = {
                "Termo Candidato": termos,
                "Peso (Importância)": pesos
            }
            df_pesos = pd.DataFrame(df_pesos_data).sort_values(by="Peso (Importância)", ascending=False)
            df_pesos["Peso (Importância)"] = df_pesos["Peso (Importância)"].map(lambda x: f"{x*100:.2f}%")
            
            st.dataframe(df_pesos, use_container_width=True)

        with col2:
            st.subheader("Ranking Final das Tabelas")
            st.write("Tabelas mais relevantes, ordenadas pela consulta otimizada.")
            
            # Prepara os dados do ranking para exibição
            df_ranking_data = {
                "Tabela": [t for t, s in ranking],
                "Similaridade (Pontuação)": [s for t, s in ranking]
            }
            df_ranking = pd.DataFrame(df_ranking_data)
            
            # Formata a similaridade
            df_ranking["Similaridade (Pontuação)"] = df_ranking["Similaridade (Pontuação)"].map(lambda x: f"{x:.4f}")
            
            st.dataframe(df_ranking.head(50), use_container_width=True) # Mostra as Top 50
        
        st.divider()

        # --- ETAPA 2: ANÁLISE DE FEATURES ---
        # Pega as N tabelas mais relevantes para analisar as colunas
        rodar_pipeline_features(buscador, metadata_loader, query_usuario, ranking[:top_n_para_analise])
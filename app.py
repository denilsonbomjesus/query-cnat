# app.py (O Ponto de Entrada Final)

import streamlit as st
import pandas as pd
import numpy as np
import logging
import time

# Importa os módulos das etapas 2 e 3
from etapa2.busca_semantica import BuscadorSemantico
from etapa3.otimizador_ga import rodar_otimizacao_ga

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

# Carrega o buscador
buscador = carregar_buscador()

st.header("Faça sua consulta")

# Entrada do usuário
query_usuario = st.text_input("Termo de busca (ex: pre-eclampsia):", "pre-eclampsia")
n_termos = st.slider("Nº de termos para expansão (N do AG):", min_value=3, max_value=20, value=10)

if st.button("Buscar Tabelas Relevantes"):
    if not query_usuario:
        st.error("Por favor, digite um termo de busca.")
    else:
        st.divider()
        st.header(f"Resultados para: '{query_usuario}'")
        
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
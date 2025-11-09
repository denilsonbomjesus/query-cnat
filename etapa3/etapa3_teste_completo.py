# etapa3_teste_completo.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etapa2.busca_semantica import BuscadorSemantico
from otimizador_ga import rodar_otimizacao_ga
import logging
import time
import numpy as np

# Configura o logging para ver os detalhes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Executa o pipeline completo:
    1. Carrega os modelos (Etapa 0)
    2. Prepara a consulta (Etapa 1 Contexto)
    3. Otimiza com AG (Etapa 2 Contexto)
    4. Gera o Ranking Final (Etapa 3 Contexto)
    """
    logging.info("--- INICIANDO PIPELINE COMPLETO (ETAPA 1, 2 e 3) ---")
    
    # 0. Carregar o buscador (Etapa 0 do Contexto)
    # Isso inicializa BERT, W2V e os vetores das tabelas
    try:
        start_load = time.time()
        buscador = BuscadorSemantico()
        logging.info(f"Buscador (Etapa 0) carregado em {time.time() - start_load:.2f}s")
    except Exception as e:
        logging.error(f"Falha ao carregar o BuscadorSemantico. Verifique os arquivos da Etapa 1. Erro: {e}")
        return

    # --- Início do processo em tempo real ---
    query_usuario = "pré-eclâmpsia"
    N_TERMOS = 10 # Número de termos para expansão (define o N do AG)
    
    logging.info(f"CONSULTA DO USUÁRIO: '{query_usuario}'")
    
    

    # 1. Preparação da Consulta (Etapa 1 do Contexto)
    start_search = time.time()
    
    # (Processo 1.1) Expandir a consulta com W2V
    termos_candidatos_tuplas = buscador.expandir_consulta(query_usuario, n=N_TERMOS)
    termos_candidatos = [termo for termo, score in termos_candidatos_tuplas]
    
    logging.info(f"(Etapa 1.1) Termos Candidatos: {termos_candidatos}")

    # (Processo 1.2) Vetorizar os termos candidatos com BERT
    v_candidatos = buscador.vetorizar_termos_candidatos(termos_candidatos)
    logging.info(f"(Etapa 1.2) V_CANDIDATOS (shape: {np.array(v_candidatos).shape}) gerados.")

    # 2. O Algoritmo Genético (Etapa 2 do Contexto)
    # Esta é a etapa central de CE obrigatória
    w_otimizado = rodar_otimizacao_ga(buscador, v_candidatos)

    # 3. Busca e Rankeamento (Etapa 3 do Contexto)
    logging.info("--- Iniciando Etapa 3 (Busca e Rankeamento Final) ---")
    
    # (Processo 3.1) Criar o vetor de consulta final com os pesos otimizados
    v_query_final = buscador.criar_vetor_consulta_ponderado(v_candidatos, w_otimizado)
    logging.info("(Processo 3.1) V_QUERY_FINAL (otimizado) criado.")

    # (Processo 3.2) Executar o ranking final
    ranking_final_otimizado = buscador.ranking_por_similaridade(v_query_final)
    logging.info("(Processo 3.2) Ranking final gerado.")
    
    end_search = time.time()
    
    # --- Exibir resultados ---
    logging.info(f"BUSCA COMPLETA (AG + Ranking) concluída em {end_search - start_search:.4f}s")
    
    print("\n" + "="*80)
    print(f"RESULTADOS DA BUSCA OTIMIZADA POR AG PARA: '{query_usuario}'")
    print("="*80)
    
    print("\nPESOS OTINIZADOS (W_OTIMIZADO) APRENDIDOS PELO AG:")
    for i, termo in enumerate(termos_candidatos):
        print(f"  - {w_otimizado[i]:.4f} * '{termo}'")
        
    print("\n--- Top 15 Tabelas (Ranking Otimizado) ---")
    for i, (tabela, score) in enumerate(ranking_final_otimizado[:15]):
        print(f"  {i+1:2}. {tabela:50} (Similaridade: {score:.4f})")
    print("="*80)
    
    logging.info("--- PIPELINE COMPLETO CONCLUÍDO ---")

if __name__ == "__main__":
    main()
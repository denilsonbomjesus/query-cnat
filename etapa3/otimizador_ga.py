# etapa3/otimizador_ga.py

import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import logging
import time
import sys
import os

# --- Mágica para importar módulos da raiz e de 'etapa2' ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir) # Adiciona a raiz ao path

# AGORA a importação funciona, pois 'etapa2' está visível
from etapa2.busca_semantica import BuscadorSemantico 
import config
# ---------------------------------------------------------

class FitnessFunctionWrapper:
    """ Wrapper para a função de fitness... (igual antes) """
    def __init__(self, buscador, v_candidatos, k_top=20):
        self.buscador = buscador
        self.v_candidatos = v_candidatos
        self.k_top = k_top
        self.dimensao = len(v_candidatos)
        if self.dimensao == 0:
            raise ValueError("v_candidatos não pode estar vazio.")

    def __call__(self, pesos_individuo):
        """ Esta é a Função de Aptidão (Fitness Function)... (igual antes) """
        v_query_evoluido = self.buscador.criar_vetor_consulta_ponderado(
            self.v_candidatos,
            pesos_individuo
        )
        ranking = self.buscador.ranking_por_similaridade(v_query_evoluido)
        top_k_scores = [score for tabela, score in ranking[:self.k_top]]
        soma_similaridade_top_k = sum(top_k_scores)
        
        if soma_similaridade_top_k == 0:
            return 0.0
            
        return -soma_similaridade_top_k

def rodar_otimizacao_ga(buscador, v_candidatos):
    """ Configura e executa o Algoritmo Genético. """
    logging.info("--- Iniciando Etapa 2 (Otimização com AG) ---")
    
    n_termos = len(v_candidatos)
    if n_termos == 0:
        logging.warning("Não há termos candidatos para otimizar. Pulando AG.")
        return np.array([1.0])

    logging.info(f"O AG irá otimizar um vetor de {n_termos} pesos.")

    # Usa os parâmetros do config.py
    funcao_fitness = FitnessFunctionWrapper(
        buscador, v_candidatos, k_top=config.K_TOP_FITNESS
    )

    varbound = np.array([[0.0, 1.0]] * n_termos)
    
    # Usa os parâmetros do config.py
    algorithm_params = config.GA_PARAMS
    
    logging.info(f"Executando AG por {algorithm_params['max_num_iteration']} gerações...")
    start_time = time.time()
    
    model_ga = ga(
        function=funcao_fitness,
        dimension=n_termos,
        variable_type='real',
        variable_boundaries=varbound,
        algorithm_parameters=algorithm_params,
        function_timeout=600
    )
    
    model_ga.run()
    
    end_time = time.time()
    logging.info(f"Otimização do AG concluída em {end_time - start_time:.2f}s")
    
    solucao = model_ga.output_dict
    w_otimizado = solucao['variable']
    fitness_otimizado = solucao['function']
    
    logging.info(f"Melhor fitness (Soma Top {config.K_TOP_FITNESS} * -1): {fitness_otimizado:.4f}")
    
    # Normaliza os pesos otimizados
    w_sum = np.sum(w_otimizado)
    if w_sum > 0:
        w_otimizado_normalizado = w_otimizado / w_sum
    else:
        w_otimizado_normalizado = w_otimizado # Evita divisão por zero
    
    logging.info(f"Pesos Otimizados (W_OTIMIZADO) normalizados: \n{w_otimizado_normalizado}")
    
    logging.info("--- Otimização com AG Concluída ---")
    
    return w_otimizado_normalizado
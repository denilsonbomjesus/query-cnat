# otimizador_ga.py

import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import logging
import time

# (Assume que 'busca_semantica.py' está na mesma pasta)
# Não podemos importar o 'BuscadorSemantico' aqui no topo
# porque ele seria inicializado. Em vez disso,
# o buscador será passado como argumento.

class FitnessFunctionWrapper:
    """
    Esta classe é um 'wrapper' (empacotador) para a função de fitness.
    Isso é necessário porque a biblioteca do AG espera uma função
    simples f(X), mas nossa função precisa de acesso ao Objeto 'BuscadorSemantico' e aos 'V_CANDIDATOS'.
    """
    def __init__(self, buscador, v_candidatos, k_top=20):
        self.buscador = buscador
        self.v_candidatos = v_candidatos
        self.k_top = k_top
        self.dimensao = len(v_candidatos)
        
        if self.dimensao == 0:
            raise ValueError("v_candidatos não pode estar vazio.")

    def __call__(self, pesos_individuo):
        """
        Esta é a Função de Aptidão (Fitness Function) - Processo 2.2 do contexto.
        Ela será chamada pelo AG para CADA indivíduo em CADA geração.
        
        'pesos_individuo' é um array numpy com N pesos (ex: [0.9, 0.7, ...])
        """
        
        # 1. (Processo 2.2a) Criar o V_QUERY_EVOLUIDO
        # O 'buscador' já lida com a normalização
        v_query_evoluido = self.buscador.criar_vetor_consulta_ponderado(
            self.v_candidatos,
            pesos_individuo
        )
        
        # 2. (Processo 2.2b/c) Fazer a busca e obter o ranking
        ranking = self.buscador.ranking_por_similaridade(v_query_evoluido)
        
        # 3. (Processo 2.2d) Calcular o Fitness
        # Métrica: Soma da Similaridade das Top K tabelas.
        # Pegamos as K primeiras pontuações do ranking.
        top_k_scores = [score for tabela, score in ranking[:self.k_top]]
        
        # Se o ranking tiver menos de K itens (improvável), apenas some o que tem
        soma_similaridade_top_k = sum(top_k_scores)

        # 4. Retornar o Fitness
        # A biblioteca 'geneticalgorithm' por padrão MINIMIZA a função.
        # Nós queremos MAXIMIZAR a 'soma_similaridade'.
        # Portanto, retornamos o NEGATIVO da soma.
        # Minimizar (-soma) é o mesmo que maximizar (soma).
        
        if soma_similaridade_top_k == 0:
            # Evita fitness 0, caso todos os scores sejam 0
            return 0.0
            
        return -soma_similaridade_top_k

def rodar_otimizacao_ga(buscador, v_candidatos):
    """
    (Processo 2.1 e 2.3) Configura e executa o Algoritmo Genético.
    
    Retorna: W_OTIMIZADO (o melhor vetor de pesos encontrado)
    """
    logging.info("--- Iniciando Etapa 2 (Otimização com AG) ---")
    
    n_termos = len(v_candidatos)
    if n_termos == 0:
        logging.warning("Não há termos candidatos para otimizar. Pulando AG.")
        return np.array([1.0]) # Retorna um peso padrão

    logging.info(f"O AG irá otimizar um vetor de {n_termos} pesos.")

    # 1. Instanciar o wrapper da Função de Fitness
    K_FITNESS = 20 # Usar a soma das Top 20 tabelas para o fitness
    funcao_fitness = FitnessFunctionWrapper(buscador, v_candidatos, k_top=K_FITNESS)

    # 2. Definir os limites das variáveis (pesos)
    # Cada peso (gene) pode variar de 0.0 a 1.0
    varbound = np.array([[0.0, 1.0]] * n_termos)
    
    # 3. Configurar os parâmetros do AG (Processo 2.1)
    # Estes são os hiperparâmetros (você pode ajustá-los)
    algorithm_params = {
        'max_num_iteration': 50,      # (Número de Gerações)
        'population_size': 100,       # (Tamanho da População)
        'mutation_probability': 0.1,  # (Taxa de Mutação)
        'elit_ratio': 0.01,           # (Taxa de Elitismo)
        'crossover_probability': 0.5, # (Taxa de Crossover)
        'parents_portion': 0.3,       # (Seleção por Torneio)
        'crossover_type': 'uniform',  # (Tipo de Crossover)
        'max_iteration_without_improv': 5 # (Parada antecipada)
    }

    # 4. Instanciar e executar o modelo do AG
    logging.info(f"Executando AG por {algorithm_params['max_num_iteration']} gerações...")
    start_time = time.time()
    
    model_ga = ga(
        function=funcao_fitness,
        dimension=n_termos,
        variable_type='real',
        variable_boundaries=varbound,
        algorithm_parameters=algorithm_params,
        function_timeout=600 # Timeout de 10 min por função
    )
    
    model_ga.run()
    
    end_time = time.time()
    logging.info(f"Otimização do AG concluída em {end_time - start_time:.2f}s")
    
    # 5. Obter os resultados
    solucao = model_ga.output_dict
    w_otimizado = solucao['variable']
    fitness_otimizado = solucao['function']
    
    logging.info(f"Melhor fitness (Soma Top {K_FITNESS} * -1): {fitness_otimizado:.4f}")
    
    # Normaliza os pesos otimizados para que sua soma seja 1 (opcional, mas bom)
    # Isso torna os pesos mais fáceis de interpretar (ex: 60% termo A, 40% termo B)
    w_otimizado_normalizado = w_otimizado / np.sum(w_otimizado)
    
    logging.info(f"Pesos Otimizados (W_OTIMIZADO) normalizados: \n{w_otimizado_normalizado}")
    
    logging.info("--- Otimização com AG Concluída ---")
    
    return w_otimizado_normalizado
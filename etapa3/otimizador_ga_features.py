# etapa3/otimizador_ga_features.py

import os
import sys
import logging
import numpy as np
import pygad
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- Mágica para importar o config.py da raiz ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GAFeatureSelector:
    """
    Usa um Algoritmo Genético para selecionar o melhor subconjunto de colunas
    de uma tabela, com base na relevância para uma consulta do usuário e na
    qualidade dos dados das colunas.
    """

    def __init__(
        self,
        user_query_vector: np.ndarray,
        table_columns_metadata: List[Dict[str, Any]],
        table_columns_vectors: Dict[str, np.ndarray],
    ):
        """
        Inicializa o seletor de features com GA.

        Args:
            user_query_vector (np.ndarray): O vetor da consulta do usuário.
            table_columns_metadata (List[Dict[str, Any]]): Uma lista de dicionários,
                onde cada dicionário contém os metadados de uma coluna.
            table_columns_vectors (Dict[str, np.ndarray]): Um dicionário mapeando
                o nome de cada coluna ao seu vetor.
        """
        if not table_columns_metadata or not table_columns_vectors:
            raise ValueError("Metadados e vetores das colunas não podem ser vazios.")

        self.user_query_vector = user_query_vector.reshape(1, -1)
        self.columns_metadata = table_columns_metadata
        self.columns_vectors = table_columns_vectors
        
        # Garante que a ordem das colunas seja consistente
        self.ordered_columns = [col['name'] for col in self.columns_metadata]
        self.num_genes = len(self.ordered_columns)

        # Parâmetros do GA a partir do config
        ga_conf = getattr(config, "GA_PARAMS", {})
        self.num_generations = ga_conf.get("max_num_iteration", 20)
        self.sol_per_pop = ga_conf.get("population_size", 50)
        self.mutation_probability = ga_conf.get("mutation_probability", 0.2)
        self.crossover_probability = ga_conf.get("crossover_probability", 0.7)
        self.elitism_ratio = ga_conf.get("elit_ratio", 0.05)
        self.parents_portion = ga_conf.get("parents_portion", 0.3)
        self.crossover_type = ga_conf.get("crossover_type", "single_point")

        # Pesos da fitness function
        self.w_semantic = 0.7
        self.w_quality = 0.2
        self.w_complexity = 0.1

        # --- Watchdog de timeout (Bug A/B) ---
        # Tempo máximo total de execução (s). Define em run() para permitir
        # chamadas diretas de fitness_function em testes/unittest.
        self._start_time = time.time()
        self._max_time_seconds = None
        self._generation_count = 0
        self._timeout_atingido = False

        # Melhor solução vista até o momento (usada se o timeout interromper)
        self.best_solution = None
        self.best_fitness = None

        # Fitness degradado devolvido quando o tempo estoura: o GA de features
        # tem fitness >= 0, então -1e9 nunca será escolhido como melhor solução.
        self._fitness_degradado = -1e9

    def _timeout_excedido(self):
        return (
            self._max_time_seconds is not None
            and (time.time() - self._start_time) > self._max_time_seconds
        )

    def fitness_function(self, ga_instance, solution, solution_idx):
        """
        Calcula o fitness de uma solução (um subconjunto de colunas).
        A solução é um vetor binário onde 1 indica uma coluna selecionada.
        """
        # Watchdog intra-geração: devolve fitness degradado assim que o tempo
        # estourar (a geração termina rápido e o on_generation para o GA),
        # reduzindo a janela de stall para o custo de UMA avaliação de fitness.
        if self._timeout_excedido():
            self._timeout_atingido = True
            return self._fitness_degradado

        selected_indices = np.where(solution == 1)[0]
        
        if len(selected_indices) == 0:
            return 0.0  # Penaliza soluções vazias

        # --- 1. Similaridade Semântica ---
        selected_vectors = [
            self.columns_vectors[self.ordered_columns[i]] for i in selected_indices
        ]
        
        # Cria um vetor agregado para as colunas selecionadas (média)
        aggregated_vector = np.mean(selected_vectors, axis=0).reshape(1, -1)
        
        # Calcula a similaridade de cosseno com o vetor da query
        semantic_score = cosine_similarity(self.user_query_vector, aggregated_vector)[0][0]

        # --- 2. Qualidade dos Dados ---
        total_quality_score = 0
        for i in selected_indices:
            col_meta = self.columns_metadata[i]
            stats = col_meta.get('stats', {})
            
            # Ex: (1 - perc_nulos) * (1 - (1 / num_valores_distintos))
            null_perc = stats.get('null_percentage', 0.0)
            distinct_count = stats.get('distinct_count', 1)
            
            # Evita divisão por zero se distinct_count for 0
            if distinct_count == 0: distinct_count = 1

            # A pontuação de qualidade favorece poucos nulos e alta cardinalidade
            quality = (1.0 - null_perc) * (1.0 - (1.0 / distinct_count))
            total_quality_score += quality
        
        avg_quality_score = total_quality_score / len(selected_indices)

        # --- 3. Penalidade por Complexidade ---
        # Penaliza soluções com muitas colunas para favorecer a simplicidade
        complexity_penalty = 1.0 / (1.0 + len(selected_indices))

        # --- Fitness Final Ponderado ---
        fitness = (
            (self.w_semantic * semantic_score) +
            (self.w_quality * avg_quality_score) +
            (self.w_complexity * complexity_penalty)
        )

        # Registra a melhor solução (fallback seguro caso o timeout interrompa)
        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = solution.copy()

        return float(fitness)

    def on_generation_timeout(self, ga_instance):
        """Callback entre gerações: interrompe o GA quando o tempo total estourar."""
        self._generation_count += 1
        elapsed = time.time() - self._start_time
        logging.info(
            f"[GA Features] Geração {self._generation_count}/{self.num_generations} — "
            f"elapsed={elapsed:.1f}s"
        )
        if self._timeout_excedido():
            logging.warning(
                f"[GA Features] TIMEOUT atingido ({elapsed:.1f}s > {self._max_time_seconds}s). Parando GA."
            )
            self._timeout_atingido = True
            return "stop"

    def run(self, max_time_seconds: int = 120):
        """
        Configura e executa a otimização com o Algoritmo Genético.

        Bug A/B (backport): adiciona timeout de segurança (default 120 s) e
        desliga o paralelismo (parallel_processing=1), espelhando o fix do GA
        contínuo — evita stall de ~30 min+ observado na Fase 2 (seed 789).

        Args:
            max_time_seconds: Tempo máximo total de execução em segundos.
                              Se excedido, retorna a melhor solução encontrada.
        """
        if self.num_genes == 0:
            logging.warning("Nenhum gene (coluna) para otimizar. Retornando resultado vazio.")
            return np.array([]), 0.0

        self._start_time = time.time()
        self._max_time_seconds = max_time_seconds
        self._generation_count = 0
        self._timeout_atingido = False
        self.best_solution = None
        self.best_fitness = None

        logging.info(
            f"--- Iniciando GA para Seleção de Features ({self.num_genes} colunas, "
            f"timeout={max_time_seconds}s) ---"
        )

        ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=max(2, int(self.sol_per_pop * self.parents_portion)),
            fitness_func=self.fitness_function,
            sol_per_pop=self.sol_per_pop,
            num_genes=self.num_genes,
            gene_space=[0, 1],  # Cromossomo binário
            gene_type=int,
            keep_elitism=int(max(1, np.round(self.elitism_ratio * self.sol_per_pop))),
            crossover_type=self.crossover_type,
            crossover_probability=self.crossover_probability,
            mutation_type="random",
            mutation_probability=self.mutation_probability,
            on_generation=self.on_generation_timeout,
            parallel_processing=1  # Desabilitado para evitar stall/contenda de threads
        )

        start = time.time()
        ga_instance.run()
        logging.info(f"GA de Features concluído em {time.time()-start:.2f}s")

        if self._timeout_atingido:
            # Timeout interrompeu no meio: usa a melhor solução rastreada pela
            # fitness function (ga.best_solution() pode estar desatualizado ou
            # apontar para a população parcialmente avaliada com fitness degradado).
            if self.best_solution is not None:
                return self.best_solution, self.best_fitness
            # Timeout antes de qualquer avaliação concluída (caso extremo):
            # devolve solução vazia em vez de propagar o erro do PyGAD.
            logging.warning("[GA Features] Timeout antes da 1ª geração. Retornando solução vazia.")
            return np.zeros(self.num_genes, dtype=int), 0.0

        solution, solution_fitness, _ = ga_instance.best_solution()
        return solution, solution_fitness


def rodar_ga_feature_selection(
    user_query_vector: np.ndarray,
    table_columns_metadata: List[Dict[str, Any]],
    table_columns_vectors: Dict[str, np.ndarray],
    max_time_seconds: int = 120
) -> (np.ndarray, float):
    """
    Função de orquestração para executar o GA de seleção de features.

    Args:
        max_time_seconds: Tempo máximo de execução (default 120 s). Propagado a
                          partir de app.py / utilitários de validação.
    """
    optimizer = GAFeatureSelector(
        user_query_vector=user_query_vector,
        table_columns_metadata=table_columns_metadata,
        table_columns_vectors=table_columns_vectors
    )
    best_solution, best_fitness = optimizer.run(max_time_seconds=max_time_seconds)
    return best_solution, best_fitness

# etapa3/otimizador_ga.py
"""
GA otimizado para busca semântica médica/hospitalar
Fluxo:
- Mantém expansão semântica existente
- Reforça peso da palavra pesquisada
- Penaliza tabelas irrelevantes lexicalmente
- Alinha melhor vetor de consulta com tabelas reais
"""

import os
import sys
import time
import logging
import numpy as np
import config
import pygad

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# from etapa2.busca_semantica import BuscadorSemantico
from etapa2.busca_semantica import BuscadorSemantico, normalize_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GAOptimizer:
    def __init__(self, buscador: BuscadorSemantico, v_candidatos, tabela_embeddings, tabela_nomes, k_top=20):
        self.buscador = buscador
        self.v_candidatos = v_candidatos
        self.v_tabelas = tabela_embeddings
        self.tabela_nomes = tabela_nomes
        self.k_top = k_top
        self.n_dim = len(v_candidatos)

        if self.n_dim == 0:
            raise ValueError("v_candidatos não pode estar vazio.")

        self._prepare_candidate_matrix()

        # parâmetros GA
        ga_conf = getattr(config, "GA_PARAMS", {}) or {}
        self.num_generations = ga_conf.get("max_num_iteration", 50)
        self.sol_per_pop = ga_conf.get("population_size", 100)
        self.mutation_prob = ga_conf.get("mutation_probability", 0.1)
        self.mutation_percent_genes = max(1, int(self.mutation_prob * 100))
        self.crossover_probability = ga_conf.get("crossover_probability", 0.5)
        self.elitism_ratio = ga_conf.get("elit_ratio", 0.01)
        self.parents_portion = ga_conf.get("parents_portion", 0.3)
        self.crossover_type = ga_conf.get("crossover_type", "single_point")

        # pesos do fitness
        self.alpha_topk = 1.0
        self.beta_diversity = 0.4
        self.gamma_bert = 0.6
        self.delta_table = 0.8  # novo peso para alinhamento com tabelas

        self.best_solution = None
        self.best_fitness = None
        self.generation_count = 0

        # Watchdog de timeout (Bug B): inicializados para permitir chamadas
        # diretas de fitness_function em testes; run() define os valores reais.
        self._start_time = time.time()
        self._max_time = 300
        self._timeout_atingido = False

        # Fitness degradado devolvido quando o tempo estoura (abaixo de qualquer
        # fitness legítimo, para nunca ser escolhido como melhor solução).
        self._fitness_degradado = -1e9

        self.persistence_dir = os.path.join("modelos", "ga_pesos")

        logging.info(f"GAOptimizer inicializado com {self.n_dim} candidatos e {len(tabela_nomes)} tabelas")

    def _prepare_candidate_matrix(self):
        X = np.vstack(self.v_candidatos)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._cand_emb = X / norms
        self._pairwise_cos = np.dot(self._cand_emb, self._cand_emb.T)

    def _diversity_penalty(self, weights):
        w = np.asarray(weights).reshape(-1, 1)
        redundancy = np.sum((np.dot(w, w.T) * self._pairwise_cos) * np.triu(np.ones_like(self._pairwise_cos), k=1))
        return float(redundancy)

    def _compute_sum_topk(self, weights):
        w = np.array(weights, dtype=float)
        w_sum = w.sum() or 1.0
        w_norm = w / w_sum
        v_query = self.buscador.criar_vetor_consulta_ponderado(self.v_candidatos, w_norm)
        ranking = self.buscador.ranking_por_similaridade(v_query)
        top_k_scores = [score for _, score in ranking[: self.k_top]]
        return float(np.sum(top_k_scores)), v_query

    def _compute_table_alignment(self, v_query):
        """Similaridade média com as tabelas reais."""
        vq_norm = v_query / (np.linalg.norm(v_query) + 1e-12)
        vt_norm = self.v_tabelas / (np.linalg.norm(self.v_tabelas, axis=1, keepdims=True) + 1e-12)
        cos = np.dot(vt_norm, vq_norm)
        return float(np.mean(sorted(cos, reverse=True)[:20]))  # top 20 médias

    def _compute_bert_context_metric(self, weights):
        if not hasattr(self.buscador, "last_query") or not self.buscador.last_query:
            return 0.0
        try:
            vq_text = self.buscador._get_bert_embedding(self.buscador.last_query, lang='pt')
            vq_text = vq_text / (np.linalg.norm(vq_text) + 1e-12)
            cosines = np.dot(self._cand_emb, vq_text)
            w = np.array(weights, dtype=float)
            w_sum = w.sum() or 1.0
            w_norm = w / w_sum
            bert_score = float(np.dot(w_norm, cosines))
            return (bert_score + 1.0) / 2.0
        except Exception:
            return 0.0

    def fitness_function(self, ga_instance, solution, solution_idx):
        # Watchdog intra-geração (Bug B): se o tempo total já estourou, devolve
        # imediatamente um fitness degradado — reduz a janela de stall (ex.: 2,8 h
        # em uma única geração no B3 N=20) para o custo de UMA avaliação; a
        # geração termina rápido e o on_generation para o GA em seguida.
        if (time.time() - self._start_time) > self._max_time:
            self._timeout_atingido = True
            return self._fitness_degradado

        # Solução (pesos) vinda do GA
        sol = np.clip(np.array(solution, dtype=float), 0.0, 1.0)

        # Calcule todas as métricas com a solução original
        sum_topk, v_query = self._compute_sum_topk(sol)
        penalty = self._diversity_penalty(sol)
        bert_metric = self._compute_bert_context_metric(sol)
        table_align = self._compute_table_alignment(v_query)

        # --- LÓGICA DE BÔNUS DA QUERY CORRIGIDA ---
        # Não modificamos 'sol'. Em vez disso, damos um bônus de fitness
        # se o GA der um peso alto ao termo da query original.
        query_gene_bonus = 0.0
        if hasattr(self.buscador, "last_query") and self.buscador.last_query:

            # Usa a mesma normalização da Etapa 1 e 2
            qterm = normalize_text(self.buscador.last_query) 

            found_index = -1
            for i, cand in enumerate(self.buscador.termos_expandidos):
                # Compara termos normalizados
                if qterm == normalize_text(cand):
                    found_index = i
                    break

            if found_index != -1:
                # RECOMPENSA: Bônus proporcional ao peso que o GA deu
                # ao gene da query original (sol[found_index]).
                # O fator 5.0 torna este bônus significativo.
                query_gene_bonus = 5.0 * sol[found_index]

        # Calcula o fitness final somando todos os componentes
        fitness = (
            self.alpha_topk * sum_topk
            + self.gamma_bert * bert_metric
            + self.delta_table * table_align
            + query_gene_bonus # <--- BÔNUS
            - self.beta_diversity * penalty
        )

        # Atualiza a melhor solução encontrada
        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = sol.copy()

        return float(fitness)

    def run(self, max_time_seconds: int = 300, save_weights: bool = False):
        """
        Executa o GA com timeout de segurança.

        Bug B (backport): além do timeout entre gerações, a fitness function
        também verifica o tempo (watchdog intra-geração), evitando que uma
        geração única trave o processo por horas.

        Bug J (backport): persistência de pesos agora é opt-in (save_weights),
        evitando acúmulo de pesos_*.npy a cada execução da UI/teste.

        Args:
            max_time_seconds: Tempo máximo de execução em segundos (default: 300 = 5 min).
                             Se o GA não convergir neste tempo, retorna a melhor solução encontrada.
            save_weights: Se True, persiste a melhor solução em modelos/ga_pesos/.
        """
        logging.info(f"--- Iniciando GA (timeout={max_time_seconds}s) ---")
        gene_space = [{'low': 0.0, 'high': 1.0}] * self.n_dim

        # Variável para controlar timeout
        self._start_time = time.time()
        self._max_time = max_time_seconds
        self.best_solution = None
        self.best_fitness = None
        self.generation_count = 0
        self._timeout_atingido = False

        # Callback para verificar timeout a cada geração
        def on_generation_timeout(ga_instance):
            self.generation_count += 1
            gen = self.generation_count
            elapsed = time.time() - self._start_time
            
            # Decay de mutação
            decay = np.exp(-3 * (gen / max(1, self.num_generations)))
            ga_instance.mutation_percent_genes = max(1, int(self.mutation_percent_genes * decay))
            
            best_fit_str = f"{self.best_fitness:.6f}" if self.best_fitness is not None else "N/A"
            logging.info(f"[GA] Geração {gen}/{self.num_generations} — best_fitness={best_fit_str} — elapsed={elapsed:.1f}s")
            
            # Verificar timeout (entre gerações)
            if elapsed > self._max_time:
                logging.warning(f"[GA] TIMEOUT atingido ({elapsed:.1f}s > {self._max_time}s). Parando GA.")
                self._timeout_atingido = True
                return "stop"
        
        ga = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=max(2, int(self.sol_per_pop * self.parents_portion)),
            fitness_func=self.fitness_function,
            sol_per_pop=self.sol_per_pop,
            num_genes=self.n_dim,
            gene_space=gene_space,
            keep_elitism=int(max(1, np.round(self.elitism_ratio * self.sol_per_pop))),
            crossover_type=self.crossover_type,
            crossover_probability=self.crossover_probability,
            mutation_type="random",
            mutation_percent_genes=self.mutation_percent_genes,
            on_generation=on_generation_timeout,
            parallel_processing=1  # Desabilitado para evitar issues com BioBERT em threads
        )

        start = time.time()
        ga.run()
        elapsed_total = time.time() - start
        logging.info(f"GA concluído em {elapsed_total:.2f}s ({self.generation_count} gerações)")

        if self._timeout_atingido:
            # Timeout interrompeu o GA: usa a melhor solução rastreada pela
            # fitness function (ga.best_solution() pode estar desatualizado
            # ou apontar para a população parcialmente avaliada).
            if self.best_solution is not None:
                best_sol = np.clip(self.best_solution, 0.0, 1.0)
            else:
                logging.warning("[GA] Timeout antes da 1ª geração. Retornando pesos uniformes.")
                best_sol = np.ones(self.n_dim) / self.n_dim
        else:
            best_sol, best_fit, _ = ga.best_solution()
            best_sol = np.clip(best_sol, 0.0, 1.0)
        best_sol /= best_sol.sum() + 1e-12
        if save_weights:
            os.makedirs(self.persistence_dir, exist_ok=True)
            np.save(os.path.join(self.persistence_dir, f"pesos_{int(time.time())}.npy"), best_sol)
        return best_sol


def rodar_otimizacao_ga(buscador, v_candidatos, tabela_embeddings, tabela_nomes, max_time_seconds: int = 300, save_weights: bool = False):
    """
    Executa a otimização de pesos via GA contínuo.

    Args:
        max_time_seconds: Tempo máximo de execução em segundos (default: 300 = 5 min).
        save_weights: Se True, persiste os pesos otimizados em modelos/ga_pesos/.
    """
    optimizer = GAOptimizer(buscador, v_candidatos, tabela_embeddings, tabela_nomes, k_top=config.K_TOP_FITNESS)
    return optimizer.run(max_time_seconds=max_time_seconds, save_weights=save_weights)

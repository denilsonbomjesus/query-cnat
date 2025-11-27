import sys
import os
import unittest
import numpy as np

# Adicionar o diretório raiz do projeto ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etapa3.otimizador_ga_features import GAFeatureSelector, rodar_ga_feature_selection

class TestGAFeatureSelector(unittest.TestCase):

    def setUp(self):
        """Configura os dados de teste para o GA de features."""
        self.embedding_dim = 768  # Dimensão dos vetores (igual ao BERT)

        # 1. Vetor de consulta do usuário (simulado)
        self.user_query_vector = np.random.rand(self.embedding_dim)

        # 2. Metadados das colunas (simulado)
        self.columns_metadata = [
            {'name': 'col_a', 'stats': {'null_percentage': 0.1, 'distinct_count': 100}},
            {'name': 'col_b', 'stats': {'null_percentage': 0.9, 'distinct_count': 2}}, # Baixa qualidade
            {'name': 'col_c', 'stats': {'null_percentage': 0.0, 'distinct_count': 500}}, # Alta qualidade
            {'name': 'col_d', 'stats': {'null_percentage': 0.2, 'distinct_count': 20}},
        ]
        self.column_names = [col['name'] for col in self.columns_metadata]

        # 3. Vetores das colunas (simulado)
        self.columns_vectors = {name: np.random.rand(self.embedding_dim) for name in self.column_names}
        
        # Simular que 'col_c' é muito similar à query
        self.columns_vectors['col_c'] = self.user_query_vector + np.random.normal(0, 0.1, self.embedding_dim)


    def test_fitness_function(self):
        """Testa a função de fitness diretamente com uma solução de exemplo."""
        selector = GAFeatureSelector(
            user_query_vector=self.user_query_vector,
            table_columns_metadata=self.columns_metadata,
            table_columns_vectors=self.columns_vectors
        )
        
        # Solução de exemplo: selecionar col_a e col_c
        solution = np.array([1, 0, 1, 0])
        
        fitness = selector.fitness_function(None, solution, 0)
        
        self.assertIsInstance(fitness, float)
        self.assertGreater(fitness, 0.0)

        # Solução com coluna de baixa qualidade
        solution_bad = np.array([0, 1, 0, 0])
        fitness_bad = selector.fitness_function(None, solution_bad, 0)
        self.assertLess(fitness_bad, fitness, "A fitness da solução ruim deveria ser menor.")

    def test_run_ga_feature_selection(self):
        """Testa a execução completa do GA de seleção de features."""
        
        solution, fitness = rodar_ga_feature_selection(
            user_query_vector=self.user_query_vector,
            table_columns_metadata=self.columns_metadata,
            table_columns_vectors=self.columns_vectors
        )

        self.assertIsInstance(solution, np.ndarray)
        self.assertEqual(solution.shape[0], len(self.column_names))
        self.assertTrue(np.all(np.isin(solution, [0, 1])))
        
        self.assertIsInstance(fitness, float)
        self.assertGreaterEqual(fitness, 0.0)

        # Dado que col_c foi simulada para ser a melhor, esperamos que ela seja selecionada
        # O índice da col_c é 2
        self.assertEqual(solution[2], 1, "A melhor coluna (col_c) deveria ter sido selecionada.")
        
        # Dado que col_b foi simulada para ser a pior, esperamos que não seja selecionada
        # O índice da col_b é 1
        self.assertEqual(solution[1], 0, "A pior coluna (col_b) não deveria ter sido selecionada.")


if __name__ == '__main__':
    unittest.main()

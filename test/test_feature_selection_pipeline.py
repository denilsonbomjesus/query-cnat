import sys
import os
import unittest
import numpy as np

# Adicionar o diretório raiz do projeto ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar todos os componentes necessários
from etapa2.busca_semantica import BuscadorSemantico
from etapa2.metadata_loader import MetadataLoader
from etapa3.otimizador_ga_features import rodar_ga_feature_selection

class TestFeatureSelectionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Configura os componentes pesados (modelos e dados) uma vez para todos os testes.
        """
        print("\nCarregando BuscadorSemantico e MetadataLoader para teste de pipeline...")
        try:
            cls.buscador = BuscadorSemantico()
            cls.loader = MetadataLoader()
            print("Componentes carregados com sucesso.")
        except Exception as e:
            cls.buscador = None
            cls.loader = None
            print(f"Erro ao carregar componentes: {e}. Testes de pipeline serão pulados.")

    def setUp(self):
        "Pula o teste se os componentes não puderam ser carregados."
        if self.buscador is None or self.loader is None:
            self.skipTest("Componentes (Buscador/Loader) não foram carregados, pulando teste.")

    def test_full_feature_selection_pipeline(self):
        """
        Testa o fluxo completo de seleção de features para uma tabela de exemplo.
        Isso simula a lógica executada dentro do app.py para uma tabela.
        """
        # 1. Definir uma tabela de exemplo e uma consulta
        table_name = "tb_fat_cad_individual" # Uma tabela com colunas relevantes
        query = "paciente com diabete"

        # 2. Obter metadados e nomes de colunas
        table_metadata = self.loader.get_table_metadata(table_name)
        self.assertIsNotNone(table_metadata, "Metadados da tabela não encontrados.")
        self.assertIn('columns', table_metadata)
        
        columns_metadata = table_metadata['columns']
        column_names = [col['name'] for col in columns_metadata]
        self.assertGreater(len(column_names), 0, "A tabela de teste não tem colunas.")

        # 3. Vetorizar colunas
        columns_vectors = self.buscador.vetorizar_colunas(column_names)
        self.assertEqual(len(columns_vectors), len(column_names))

        # 4. Vetorizar a consulta do usuário
        user_query_vector = self.buscador._get_bert_embedding(query, lang='pt')
        self.assertEqual(user_query_vector.shape, (768,))

        # 5. Executar o GA de seleção de features
        solution, fitness = rodar_ga_feature_selection(
            user_query_vector=user_query_vector,
            table_columns_metadata=columns_metadata,
            table_columns_vectors=columns_vectors
        )

        # 6. Validar os resultados
        self.assertIsNotNone(solution, "A solução do GA não pode ser nula.")
        self.assertIsInstance(solution, np.ndarray)
        self.assertEqual(solution.shape[0], len(column_names))
        
        self.assertIsNotNone(fitness, "O fitness do GA não pode ser nulo.")
        self.assertIsInstance(fitness, float)
        self.assertGreaterEqual(fitness, 0.0)

        # Opcional: Verificar se alguma coluna foi selecionada
        selected_indices = np.where(solution == 1)[0]
        print(f"\nTeste de Pipeline para '{table_name}' com query '{query}':")
        print(f"Fitness: {fitness:.4f}")
        if len(selected_indices) > 0:
            selected_columns = [column_names[i] for i in selected_indices]
            print(f"Colunas Selecionadas: {selected_columns}")
            self.assertIn("st_diabete", selected_columns, "A coluna 'st_diabete' deveria ter sido selecionada para a query 'paciente com diabete'.")
        else:
            print("Nenhuma coluna selecionada pelo GA.")


if __name__ == '__main__':
    unittest.main()

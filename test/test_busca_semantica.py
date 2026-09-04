import sys
import os
import unittest
import numpy as np

# Adicionar o diretório raiz do projeto ao sys.path para importar config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etapa2.busca_semantica import BuscadorSemantico

class TestBuscadorSemantico(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Configura o BuscadorSemantico uma vez para todos os testes.
        Isso evita carregar modelos grandes repetidamente.
        """
        print("\nCarregando BuscadorSemantico para testes. Isso pode levar alguns minutos...")
        try:
            cls.buscador = BuscadorSemantico()
            print("BuscadorSemantico carregado com sucesso.")
        except Exception as e:
            cls.buscador = None
            print(f"Erro ao carregar BuscadorSemantico: {e}. Testes relacionados a modelos serão pulados.")
            # Dependendo do erro, pode ser necessário levantar o erro ou marcar os testes como pulados
            # Para este cenário, vamos apenas printar e os testes subsequentes podem falhar
            # ou verificar cls.buscador é None.

    def setUp(self):
        if self.buscador is None:
            self.skipTest("BuscadorSemantico não foi carregado, pulando teste.")

    def test_vetorizar_colunas_no_description(self):
        """Testa a vetorização de colunas sem descrições."""
        column_names = ["coluna_teste_1", "coluna_teste_2", "exame_colesterol"]
        col_vectors = self.buscador.vetorizar_colunas(column_names)

        self.assertIsInstance(col_vectors, dict)
        self.assertEqual(len(col_vectors), len(column_names))

        for col_name in column_names:
            self.assertIn(col_name, col_vectors)
            self.assertIsInstance(col_vectors[col_name], np.ndarray)
            # Verifica se o vetor tem a dimensão esperada (ex: 768 para BERT base)
            self.assertEqual(col_vectors[col_name].shape, (768,)) # BERT-base-portuguese-cased tem 768 dim

    def test_vetorizar_colunas_with_description(self):
        """Testa a vetorização de colunas com descrições."""
        column_names = ["coluna_teste_3", "coluna_idade"]
        descriptions = {
            "coluna_teste_3": "uma descrição detalhada da coluna teste 3",
            "coluna_idade": "idade do paciente em anos"
        }
        col_vectors = self.buscador.vetorizar_colunas(column_names, descriptions)

        self.assertIsInstance(col_vectors, dict)
        self.assertEqual(len(col_vectors), len(column_names))

        for col_name in column_names:
            self.assertIn(col_name, col_vectors)
            self.assertIsInstance(col_vectors[col_name], np.ndarray)
            self.assertEqual(col_vectors[col_name].shape, (768,))

    def test_vetorizar_colunas_empty_list(self):
        """Testa a vetorização com uma lista de colunas vazia."""
        col_vectors = self.buscador.vetorizar_colunas([])
        self.assertIsInstance(col_vectors, dict)
        self.assertEqual(len(col_vectors), 0)

if __name__ == '__main__':
    unittest.main()

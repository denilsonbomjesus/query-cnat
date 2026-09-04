import sys
import os
import unittest
import json

# Adicionar o diretório raiz do projeto ao sys.path para importar config e etapa2.metadata_loader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etapa2.metadata_loader import MetadataLoader
from config import METADATA_ADVANCED_FILE_PATH

class TestMetadataLoader(unittest.TestCase):

    def setUp(self):
        """Configura o MetadataLoader antes de cada teste."""
        # Garantir que o arquivo de metadados existe para os testes
        if not os.path.exists(METADATA_ADVANCED_FILE_PATH):
            self.fail(f"Arquivo de metadados não encontrado em: {METADATA_ADVANCED_FILE_PATH}")
        self.loader = MetadataLoader(METADATA_ADVANCED_FILE_PATH)

    def test_load_metadata(self):
        """Verifica se os metadados são carregados corretamente."""
        self.assertIsNotNone(self.loader._metadata)
        self.assertIsInstance(self.loader._metadata, dict)
        self.assertGreater(len(self.loader._metadata), 0, "Nenhuma tabela carregada.")
        self.assertIn("TB_MIGRACAO_DADOS", self.loader._metadata)

    def test_get_table_metadata_existing(self):
        """Testa a recuperação de metadados para uma tabela existente."""
        table_name = "TB_MIGRACAO_DADOS"
        metadata = self.loader.get_table_metadata(table_name)
        self.assertIsNotNone(metadata)
        self.assertIn("table_name", metadata)
        self.assertEqual(metadata["table_name"], table_name)
        self.assertIn("columns", metadata)
        self.assertIsInstance(metadata["columns"], list)
        self.assertGreater(len(metadata["columns"]), 0)

    def test_get_table_metadata_non_existing(self):
        """Testa a recuperação de metadados para uma tabela não existente."""
        table_name = "NON_EXISTENT_TABLE"
        metadata = self.loader.get_table_metadata(table_name)
        self.assertEqual(metadata, {})

    def test_get_column_metadata_existing(self):
        """Testa a recuperação de metadados para uma coluna existente."""
        table_name = "TB_MIGRACAO_DADOS"
        column_name = "installed_rank"
        metadata = self.loader.get_column_metadata(table_name, column_name)
        self.assertIsNotNone(metadata)
        self.assertIn("name", metadata)
        self.assertEqual(metadata["name"], column_name)
        self.assertIn("type", metadata)

    def test_get_column_metadata_non_existing_column(self):
        """Testa a recuperação de metadados para uma coluna não existente."""
        table_name = "TB_MIGRACAO_DADOS"
        column_name = "NON_EXISTENT_COLUMN"
        metadata = self.loader.get_column_metadata(table_name, column_name)
        self.assertEqual(metadata, {})

    def test_get_column_metadata_non_existing_table(self):
        """Testa a recuperação de metadados para uma coluna em uma tabela não existente."""
        table_name = "NON_EXISTENT_TABLE"
        column_name = "some_column"
        metadata = self.loader.get_column_metadata(table_name, column_name)
        self.assertEqual(metadata, {})

if __name__ == '__main__':
    unittest.main()

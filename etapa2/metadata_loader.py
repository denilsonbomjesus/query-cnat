import json
import os
from collections import defaultdict
from typing import Dict, Any

from config import METADATA_ADVANCED_FILE_PATH # Assumindo que config.py está no PYTHONPATH ou acessível

class MetadataLoader:
    def __init__(self, metadata_file_path: str = METADATA_ADVANCED_FILE_PATH):
        """
        Inicializa o carregador de metadados, carregando o arquivo JSON especificado.
        Os metadados são armazenados em um dicionário para acesso rápido por nome de tabela.
        """
        self.metadata_file_path = metadata_file_path
        self._metadata: Dict[str, Dict[str, Any]] = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Carrega o arquivo JSON de metadados e o organiza em um dicionário
        onde a chave principal é o nome da tabela.
        """
        if not os.path.exists(self.metadata_file_path):
            raise FileNotFoundError(f"Arquivo de metadados não encontrado em: {self.metadata_file_path}")

        with open(self.metadata_file_path, 'r', encoding='utf-8') as f:
            full_metadata = json.load(f)

        # Organiza os metadados por nome de tabela para acesso rápido
        organized_metadata = {}
        for item in full_metadata:
            table_name = item.get("table_name")
            if table_name:
                organized_metadata[table_name] = item
        return organized_metadata

    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """
        Retorna todos os metadados para uma tabela específica.

        Args:
            table_name (str): O nome da tabela.

        Returns:
            Dict[str, Any]: Um dicionário contendo todos os metadados da tabela,
                            ou um dicionário vazio se a tabela não for encontrada.
        """
        return self._metadata.get(table_name, {})

    def get_column_metadata(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """
        Retorna os metadados para uma coluna específica dentro de uma tabela.

        Args:
            table_name (str): O nome da tabela.
            column_name (str): O nome da coluna.

        Returns:
            Dict[str, Any]: Um dicionário contendo os metadados da coluna,
                            ou um dicionário vazio se a tabela ou coluna não for encontrada.
        """
        table_data = self.get_table_metadata(table_name)
        if not table_data:
            return {}

        for col in table_data.get("columns", []):
            if col.get("name") == column_name:
                return col
        return {}

import os
import sys

# Adiciona o diretório raiz do projeto ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etapa2.busca_semantica import BuscadorSemantico

try:
    buscador = BuscadorSemantico()
    query = "dislipidemia"
    print(f"\n--- Testando Expansão para: '{query}' ---")
    termos = buscador.expandir_consulta(query, n=10)
    for t, s in termos:
        print(f"  - {t} ({s:.4f})")
except Exception as e:
    print(f"Erro: {e}")

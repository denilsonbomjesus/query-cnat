# etapa1/etapa1_script1_vetorizar_tabelas.py

import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os
import logging
import sys
import unicodedata

# --- Mágica para importar o config.py da raiz ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config  # Agora podemos importar o config.py da raiz
# ------------------------------------------------

# Configura o logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def check_file_exists(filepath):
    """Verifica se o arquivo JSON de entrada existe."""
    if not os.path.exists(filepath):
        logging.error(f"Erro: Arquivo não encontrado em '{filepath}'")
        logging.error("Por favor, verifique o caminho em config.py (JSON_FILE_PATH).")
        return False
    return True

def normalize_text(text):
    """Remove acentos, converte para minúsculas e remove espaços extras."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    return " ".join(text.lower().split())

def create_table_document(table_metadata):
    """
    Cria texto estruturado indicando tabela e suas colunas,
    normalizando acentos e letras, sem incluir tipos nem exemplos.
    """
    table_name = table_metadata.get("table_name", "")
    if not table_name:
        return ""

    parts = [f"tabela {table_name}"]

    if "columns" in table_metadata:
        column_names = [c.get("name", "") for c in table_metadata["columns"] if c.get("name")]
        if column_names:
            parts.append("colunas " + " ".join(column_names))

    document = " ".join(parts)
    document = normalize_text(document)
    return document.strip()

def get_bert_embedding(text, model, tokenizer, device):
    """Gera o embedding do texto usando o modelo BERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
    return cls_embedding

def main():
    """Função principal para executar o pré-processamento."""
    if not check_file_exists(config.JSON_FILE_PATH):
        return

    logging.info(f"Carregando metadados de {config.JSON_FILE_PATH}...")
    try:
        with open(config.JSON_FILE_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        logging.error(f"Falha ao carregar o JSON: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")

    logging.info(f"Carregando modelo BERT: {config.BERT_MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = BertModel.from_pretrained(config.BERT_MODEL_NAME).to(device)
    model.eval()

    all_vectors = []
    table_index_map = {}

    logging.info("Iniciando vetorização das tabelas...")
    for i, table in enumerate(tqdm(metadata, desc="Vetorizando Tabelas")):
        table_name = table.get("table_name")
        if not table_name:
            continue

        document = create_table_document(table)
        if not document:
            continue

        vector = get_bert_embedding(document, model, tokenizer, device)
        all_vectors.append(vector)
        table_index_map[table_name] = i

    v_tabelas = np.array(all_vectors)
    logging.info(f"Vetorização concluída. Shape: {v_tabelas.shape}")

    try:
        logging.info(f"Salvando vetores em {config.V_TABELAS_PATH}...")
        np.save(config.V_TABELAS_PATH, v_tabelas)

        logging.info(f"Salvando índice de tabelas em {config.INDEX_PATH}...")
        with open(config.INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(table_index_map, f, indent=2)

        logging.info("--- ETAPA 1 (SCRIPT 1) CONCLUÍDA ---")

    except Exception as e:
        logging.error(f"Falha ao salvar os arquivos de saída: {e}")

if __name__ == "__main__":
    main()

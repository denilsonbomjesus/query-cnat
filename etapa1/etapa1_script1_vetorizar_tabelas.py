# etapa1/etapa1_script1_vetorizar_tabelas.py

import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os
import logging
import sys

# --- Mágica para importar o config.py da raiz ---
# 1. Pega o diretório deste script (etapa1/)
# 2. Pega o diretório pai (a raiz do projeto)
# 3. Adiciona o pai ao sys.path para encontrar o 'config'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config # Agora podemos importar o config.py da raiz
# ------------------------------------------------

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_file_exists(filepath):
    """Verifica se o arquivo JSON de entrada existe."""
    if not os.path.exists(filepath):
        logging.error(f"Erro: Arquivo não encontrado em '{filepath}'")
        logging.error("Por favor, verifique o caminho em config.py (JSON_FILE_PATH).")
        return False
    return True

# (O resto das funções create_table_document e get_bert_embedding permanecem iguais...)
# ... (Cole as funções create_table_document e get_bert_embedding aqui) ...
def create_table_document(table_metadata):
    table_name = table_metadata.get('table_name', '')
    column_names = []
    if 'columns' in table_metadata:
        for column in table_metadata['columns']:
            column_names.append(column.get('name', ''))
    document = f"{table_name} {' '.join(column_names)}"
    return document.strip()

def get_bert_embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
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
        with open(config.JSON_FILE_PATH, 'r', encoding='utf-8') as f:
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
        table_name = table.get('table_name')
        if not table_name:
            continue
            
        document = create_table_document(table)
        vector = get_bert_embedding(document, model, tokenizer, device)
        all_vectors.append(vector)
        table_index_map[table_name] = i

    v_tabelas = np.array(all_vectors)
    logging.info(f"Vetorização concluída. Shape: {v_tabelas.shape}")

    try:
        logging.info(f"Salvando vetores em {config.V_TABELAS_PATH}...")
        np.save(config.V_TABELAS_PATH, v_tabelas)
        
        logging.info(f"Salvando índice de tabelas em {config.INDEX_PATH}...")
        with open(config.INDEX_PATH, 'w', encoding='utf-8') as f:
            json.dump(table_index_map, f, indent=2)
            
        logging.info("--- ETAPA 1 (SCRIPT 1) CONCLUÍDA ---")

    except Exception as e:
        logging.error(f"Falha ao salvar os arquivos de saída: {e}")

if __name__ == "__main__":
    main()
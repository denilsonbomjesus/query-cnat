# etapa1/etapa1_script1_vetorizar_tabelas.py
# NOVA VERSÃO: Vetorização por COLUNA

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
# Usaremos o alias 'cfg' para evitar conflito com 'config' de outros módulos
import config as cfg
# ------------------------------------------------

# Configura o logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_bert_mean_pooling_embedding(model_output, attention_mask):
    """Aplica Mean Pooling para obter um embedding de nível de sentença."""
    last_hidden_state = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embedding = sum_embeddings / sum_mask
    return mean_embedding.cpu().numpy()[0]

def check_file_exists(filepath):
    """Verifica se o arquivo JSON de entrada existe."""
    if not os.path.exists(filepath):
        logging.error(f"Erro: Arquivo não encontrado em '{filepath}'")
        return False
    return True

def normalize_text(text):
    """Remove acentos, converte para minúsculas e remove espaços extras."""
    if not text: return ""
    try:
        # Tenta a normalização padrão
        text = str(text)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        return " ".join(text.lower().split())
    except Exception:
        # Se falhar, retorna o texto original normalizado de forma simples
        return " ".join(str(text).lower().split())


def create_column_document(table_name, column_metadata):
    """
    Cria um "documento" textual para uma única coluna, incluindo o nome da tabela,
    nome da coluna, descrição e uma amostra de seus valores.
    """
    column_name = column_metadata.get("name", "")
    if not column_name:
        return ""

    # 1. Nomes da tabela e coluna
    doc_parts = [normalize_text(table_name), normalize_text(column_name)]

    # 2. Descrição (se houver, mas por enquanto não temos no JSON)
    # description = column_metadata.get("description", "")
    # if description:
    #     doc_parts.append(normalize_text(description))

    # 3. Amostra de dados (o mais importante para a semântica)
    stats = column_metadata.get("stats", {})
    sample_values = []
    if "frequent_values" in stats and stats["frequent_values"]:
        # Prioriza valores frequentes, que são mais representativos
        for item in stats["frequent_values"][:10]: # Limita a 10 para não poluir
            sample_values.append(str(item.get("value", "")))
    elif "sample_values" in stats and stats["sample_values"]:
        # Usa valores de amostra como fallback
        sample_values = [str(v) for v in stats["sample_values"][:10]] # Limita a 10

    if sample_values:
        # Adiciona um prefixo para dar contexto ao modelo de que são valores da coluna
        doc_parts.append("valores da coluna:")
        normalized_samples = [normalize_text(val) for val in sample_values if val]
        doc_parts.extend(list(set(normalized_samples))) # Usa set para evitar repetições

    # Junta tudo em um único documento
    document = " ".join(doc_parts)
    return document.strip()


def get_bert_embedding(text, model, tokenizer, device):
    """Gera o embedding do texto usando o modelo BERT (mean pooling)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    mean_embedding = get_bert_mean_pooling_embedding(outputs, inputs['attention_mask'])
    return mean_embedding

def main():
    """Função principal para vetorizar COLUNAS."""
    if not check_file_exists(cfg.METADATA_ADVANCED_FILE_PATH):
        return

    logging.info(f"Carregando metadados de {cfg.METADATA_ADVANCED_FILE_PATH}...")
    try:
        with open(cfg.METADATA_ADVANCED_FILE_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        logging.error(f"Falha ao carregar o JSON: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")

    logging.info(f"Carregando modelo BERT: {cfg.BERT_MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
    model = BertModel.from_pretrained(cfg.BERT_MODEL_NAME).to(device)
    model.eval()

    all_column_vectors = []
    column_index = []
    idx_counter = 0

    logging.info("Iniciando vetorização por COLUNA...")
    for table in tqdm(metadata, desc="Processando Tabelas"):
        table_name = table.get("table_name")
        if not table_name or "columns" not in table:
            continue

        for column in table["columns"]:
            column_name = column.get("name")
            if not column_name:
                continue

            # Cria o documento para a coluna atual
            document = create_column_document(table_name, column)

            if not document:
                logging.warning(f"Nenhum documento gerado para {table_name}.{column_name}, pulando.")
                continue

            # Gera o vetor
            vector = get_bert_embedding(document, model, tokenizer, device)
            all_column_vectors.append(vector)

            # Adiciona a entrada no índice
            column_index.append({
                "index": idx_counter,
                "table_name": table_name,
                "column_name": column_name
            })
            idx_counter += 1

    v_colunas = np.array(all_column_vectors)
    logging.info(f"Vetorização de colunas concluída. Shape: {v_colunas.shape}")

    try:
        logging.info(f"Salvando vetores de colunas em {cfg.V_COLUNAS_PATH}...")
        np.save(cfg.V_COLUNAS_PATH, v_colunas)

        logging.info(f"Salvando índice de colunas em {cfg.COLUNAS_INDEX_PATH}...")
        with open(cfg.COLUNAS_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(column_index, f, indent=2)

        logging.info("--- ETAPA 1 (SCRIPT 1 - VETORIZAÇÃO POR COLUNA) CONCLUÍDA ---")

    except Exception as e:
        logging.error(f"Falha ao salvar os arquivos de saída: {e}")

if __name__ == "__main__":
    main()

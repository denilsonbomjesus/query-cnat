# etapa1_script1_vetorizar_tabelas.py

import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os
import logging

# --- Configuração ---
# !! IMPORTANTE: Coloque o caminho para o seu arquivo JSON aqui !!
JSON_FILE_PATH = "asset/metadata_advanced_consolidated.json" 

# Modelo BERT em Português
BERT_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'

# Arquivos de Saída (Ativos que serão gerados)
V_TABELAS_PATH = "v_tabelas.npy"
INDEX_PATH = "tabelas_index.json"

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_file_exists(filepath):
    """Verifica se o arquivo JSON de entrada existe."""
    if not os.path.exists(filepath):
        logging.error(f"Erro: Arquivo não encontrado em '{filepath}'")
        logging.error("Por favor, baixe o arquivo ou atualize o caminho na variável JSON_FILE_PATH.")
        return False
    return True

def load_metadata(filepath):
    """Carrega os metadados do arquivo JSON."""
    logging.info(f"Carregando metadados de {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Metadados carregados: {len(data)} tabelas encontradas.")
        return data
    except Exception as e:
        logging.error(f"Falha ao carregar ou processar o JSON: {e}")
        return None

def create_table_document(table_metadata):
    """
    Cria um "documento" de texto para uma tabela, conforme descrito no contexto.
    Isso concatena metadados relevantes para representar a tabela.
    """
    # Pega o nome da tabela
    table_name = table_metadata.get('table_name', '')
    
    # Pega os nomes de todas as colunas
    column_names = []
    if 'columns' in table_metadata:
        for column in table_metadata['columns']:
            column_names.append(column.get('name', ''))
            
    # Concatena tudo em uma única string
    # Ex: "TB_PACIENTE NM_PACIENTE DT_NASCIMENTO CD_GENERO"
    document = f"{table_name} {' '.join(column_names)}"
    return document.strip()

def get_bert_embedding(text, model, tokenizer, device):
    """
    Gera o vetor (embedding) para um dado texto usando o modelo BERT.
    Usamos o vetor do token [CLS], que representa a frase inteira.
    """
    # Tokeniza o texto e move para o dispositivo (GPU ou CPU)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Executa o modelo sem calcular gradientes (mais rápido)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Pega o embedding do token [CLS] (o primeiro token)
    # .last_hidden_state tem shape [batch_size, sequence_length, hidden_size]
    cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
    return cls_embedding

def main():
    """Função principal para executar o pré-processamento."""
    
    if not check_file_exists(JSON_FILE_PATH):
        return

    metadata = load_metadata(JSON_FILE_PATH)
    if metadata is None:
        return

    # Detecta se há GPU (CUDA) disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")

    # Carrega o tokenizador e o modelo BERT
    logging.info(f"Carregando modelo BERT: {BERT_MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
    model.eval()  # Coloca o modelo em modo de avaliação

    all_vectors = []
    table_index_map = {}

    logging.info("Iniciando vetorização das tabelas...")
    # Itera por todas as tabelas com uma barra de progresso
    for i, table in enumerate(tqdm(metadata, desc="Vetorizando Tabelas")):
        table_name = table.get('table_name')
        if not table_name:
            logging.warning(f"Tabela no índice {i} não possui 'table_name'. Pulando.")
            continue
            
        # 1. Criar o documento
        document = create_table_document(table)
        
        # 2. Gerar o vetor
        vector = get_bert_embedding(document, model, tokenizer, device)
        
        # 3. Armazenar o vetor e o índice
        all_vectors.append(vector)
        table_index_map[table_name] = i

    if not all_vectors:
        logging.error("Nenhum vetor foi gerado. Verifique o seu arquivo JSON.")
        return

    # Converte a lista de vetores em um array NumPy
    v_tabelas = np.array(all_vectors)
    logging.info(f"Vetorização concluída. Shape do array de vetores: {v_tabelas.shape}")

    # 4. Salvar os "pesos" (ativos) no disco
    try:
        logging.info(f"Salvando vetores em {V_TABELAS_PATH}...")
        np.save(V_TABELAS_PATH, v_tabelas)
        
        logging.info(f"Salvando índice de tabelas em {INDEX_PATH}...")
        with open(INDEX_PATH, 'w', encoding='utf-8') as f:
            json.dump(table_index_map, f, indent=2)
            
        logging.info("--- ETAPA 1 (SCRIPT 1) CONCLUÍDA ---")
        logging.info(f"Ativos gerados: {V_TABELAS_PATH} e {INDEX_PATH}")

    except Exception as e:
        logging.error(f"Falha ao salvar os arquivos de saída: {e}")

if __name__ == "__main__":
    main()
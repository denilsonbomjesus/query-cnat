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

def get_bert_mean_pooling_embedding(model_output, attention_mask):
    """Aplica Mean Pooling para obter um embedding de nível de sentença."""
    # model_output[0] é o last_hidden_state
    last_hidden_state = model_output.last_hidden_state
    
    # Expande a máscara de atenção para as dimensões do embedding
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    # Soma os embeddings (anulando os de padding)
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    
    # Soma a máscara (para obter o número de tokens reais)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Retorna a média
    mean_embedding = sum_embeddings / sum_mask
    return mean_embedding.cpu().numpy()[0] # [0] para extrair do batch
# -------------------------------------------

def check_file_exists(filepath):
    """Verifica se o arquivo JSON de entrada existe."""
    if not os.path.exists(filepath):
        logging.error(f"Erro: Arquivo não encontrado em '{filepath}'")
        logging.error("Por favor, verifique o caminho em config.py (JSON_FILE_PATH).")
        return False
    return True

def normalize_text(text):
    """Remove acentos, converte para minúsculas e remove espaços extras."""
    if not text: return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    return " ".join(text.lower().split())

# --- FUNÇÃO MODIFICADA ---
def expandir_nome_tabela(nome):
    """
    Expande nomes técnicos de tabelas (ex: 'ta_exame_colesterol_total')
    em palavras-chave (ex: 'associacao exame colesterol total').
    """
    nome = nome.lower().strip().replace("_", " ")
    
    # Apenas troca prefixos por palavras-chave, sem adicionar "tabela de"
    if nome.startswith("tb "):
        nome = nome.replace("tb ", "tabela ", 1)
    elif nome.startswith("ta "):
        nome = nome.replace("ta ", "associacao ", 1)
    elif nome.startswith("tl "):
        nome = nome.replace("tl ", "lista ", 1)
    elif nome.startswith("rl "):
        nome = nome.replace("rl ", "relacao ", 1)
    elif nome.startswith("dim "):
        nome = nome.replace("dim ", "dimensao ", 1)
    
    # REMOVEMOS AS HEURÍSTICAS QUE ADICIONAVAM POLUIÇÃO
    # (ex: "laboratoriais e biomédicos", "relacionados a pacientes...")
    
    return nome

# --- FUNÇÃO MODIFICADA ---
def create_table_document(table_metadata):
    """
    Cria um "documento" de palavras-chave puras para a tabela,
    removendo stop-words e prefixos de coluna.
    """
    table_name = table_metadata.get("table_name", "")
    if not table_name:
        return ""
    
    table_name_keywords = set(expandir_nome_tabela(table_name).split())
    
    column_keywords = set()
    if "columns" in table_metadata:
        column_names_raw = [c.get("name", "") for c in table_metadata["columns"] if c.get("name")]
        for name in column_names_raw:
            name = name.lower().strip().replace("_", " ")
            
            # Remove prefixos comuns de colunas que não têm semântica
            if name.startswith("co "): name = name[3:]
            if name.startswith("id "): name = name[3:]
            if name.startswith("cd "): name = name[3:]
            if name.startswith("dt "): name = name[3:]
            if name.startswith("st "): name = name[3:]
            if name.startswith("tp "): name = name[3:]
            if name.startswith("ds "): name = name[3:]
            if name.startswith("nr "): name = name[3:]
            
            # Adiciona as palavras-chave da coluna
            column_keywords.update(name.split())

    # Une todas as palavras-chave (nome da tabela + colunas)
    all_keywords = table_name_keywords.union(column_keywords)
    
    # Filtra stop words em português para não poluir o vetor mean-pooling
    stop_words_pt = set([
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "nao", "uma", 
        "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi", "ao", 
        "ele", "das", "tem", "sem", "nos", "ja", "eu", "tambem", "so", "pelo", "pela", 
        "ate", "isso", "ela", "entre", "era", "depois", "nem", "mesmo", "outro", "ha", 
        "sua", "ou", "ser", "quando", "muito", "qm", "voce", "ainda", "sao", "quem",
        "contem", "colunas", "laboratoriais", "biomedicos", "classificacao", 
        "internacional", "doencas", "relacionados", "pacientes", "cidadaos", "tabela"
    ])

    final_keywords = [k for k in all_keywords if k not in stop_words_pt and len(k) > 2]
    
    document = " ".join(final_keywords)
    
    # Chama a normalização aqui para garantir
    document = normalize_text(document)
    return document.strip()

def get_bert_embedding(text, model, tokenizer, device):
    """Gera o embedding do texto usando o modelo BERT (mean pooling)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Usa a nova função de mean pooling
    mean_embedding = get_bert_mean_pooling_embedding(outputs, inputs['attention_mask'])
    return mean_embedding

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

        # Chama a nova função create_table_document "limpa"
        document = create_table_document(table)
        
        if not document:
            # Pula tabelas que não geraram palavras-chave (ex: tabelas só de ID)
            logging.warning(f"Nenhum documento de palavra-chave gerado para {table_name}, pulando.")
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
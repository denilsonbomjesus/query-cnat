# config.py
import os

# Raiz do projeto
# __file__ se refere a este arquivo (config.py)
# os.path.dirname(...) pega o diretório dele (a raiz do projeto)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Caminhos dos Ativos (Entrada) ---

# JSON original
METADATA_ADVANCED_FILE_PATH = os.path.join(BASE_DIR, "asset", "metadata_advanced_consolidated.json")

# Modelo W2V
# W2V_MODEL_PATH = os.path.join(BASE_DIR, "nilc_model", "cbow_s300.txt")

# Novo modelo BioWordVec (inglês biomédico)
BIOWORDVEC_MODEL_PATH = os.path.join(BASE_DIR, "modelos", "biowordvec_500k.kv")

# Escolha qual modelo usar (opções: 'nilc' ou 'biowordvec')
ACTIVE_W2V_MODEL = 'biowordvec'

# --- Caminhos dos Artefatos Gerados (Saída da Etapa 1) ---
V_TABELAS_PATH = os.path.join(BASE_DIR, "v_tabelas.npy")
INDEX_PATH = os.path.join(BASE_DIR, "tabelas_index.json")

# --- Configurações dos Modelos ---
# BERT_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
BERT_MODEL_NAME = 'pucpr/biobertpt-all'
BERT_EN_MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'

# --- Configurações do AG (Etapa 3) ---
GA_PARAMS = {
    'max_num_iteration': 50,      # (Número de Gerações)
    'population_size': 100,       # (Tamanho da População)
    'mutation_probability': 0.1,  # (Taxa de Mutação)
    'elit_ratio': 0.01,           # (Taxa de Elitismo)
    'crossover_probability': 0.5, # (Taxa de Crossover)
    'parents_portion': 0.3,       # (Seleção por Torneio)
    'crossover_type': 'uniform',  # (Tipo de Crossover)
    'max_iteration_without_improv': 5 # (Parada antecipada)
}
K_TOP_FITNESS = 20 # K tabelas usadas no cálculo do Fitness

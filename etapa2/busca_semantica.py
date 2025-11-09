# etapa2/busca_semantica.py

import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
import logging
import os
import time
import sys

# --- Mágica para importar o config.py da raiz ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config # Importa o config.py da raiz
# ------------------------------------------------

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BuscadorSemantico:
    """ Encapsula toda a lógica de busca semântica. """
    
    def __init__(self):
        logging.info("Inicializando o BuscadorSemantico...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Usando dispositivo: {self.device}")
        
        self.tokenizer = None
        self.bert_model = None
        self.w2v_model = None
        self.v_tabelas = None
        self.tabelas_index_map = None # Mapa: nome -> id
        self.index_tabelas_map = None # Mapa: id -> nome

        self._load_bert()
        self._load_w2v()
        self._load_table_vectors()
        
        logging.info("BuscadorSemantico pronto para uso.")

    def _load_bert(self):
        """Carrega o tokenizador e o modelo BERT do config."""
        try:
            logging.info(f"Carregando modelo BERT: {config.BERT_MODEL_NAME}...")
            self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
            self.bert_model = BertModel.from_pretrained(config.BERT_MODEL_NAME).to(self.device)
            self.bert_model.eval()
            logging.info("Modelo BERT carregado com sucesso.")
        except Exception as e:
            logging.error(f"Falha ao carregar o modelo BERT: {e}")
            raise

    def _load_w2v(self):
        """Carrega o modelo Word2Vec do config."""
        try:
            logging.info(f"Carregando modelo W2V de {config.W2V_MODEL_PATH} (pode demorar)...")
            start_time = time.time()
            self.w2v_model = KeyedVectors.load_word2vec_format(config.W2V_MODEL_PATH, binary=False)
            logging.info(f"Modelo W2V carregado em {time.time() - start_time:.2f}s")
        except Exception as e:
            logging.error(f"Falha ao carregar o modelo W2V. O arquivo existe? {e}")
            raise

    def _load_table_vectors(self):
        """Carrega os vetores da tabela e o índice do config."""
        try:
            logging.info(f"Carregando vetores das tabelas de {config.V_TABELAS_PATH}...")
            self.v_tabelas = np.load(config.V_TABELAS_PATH)
            self.v_tabelas = normalize(self.v_tabelas, norm='l2', axis=1)
            
            logging.info(f"Carregando índice de {config.INDEX_PATH}...")
            with open(config.INDEX_PATH, 'r', encoding='utf-8') as f:
                self.tabelas_index_map = json.load(f)
                
            self.index_tabelas_map = {v: k for k, v in self.tabelas_index_map.items()}
            logging.info(f"Vetores das tabelas (shape: {self.v_tabelas.shape}) e índices carregados.")
        except Exception as e:
            logging.error(f"Falha ao carregar os arquivos das tabelas: {e}")
            raise

    # (O resto das funções _get_bert_embedding, expandir_consulta, 
    #  vetorizar_termos_candidatos, criar_vetor_consulta_ponderado, 
    #  e ranking_por_similaridade permanecem EXATAMENTE IGUAIS)
    # ... (Cole o resto das funções da Etapa 2 aqui) ...
    def _get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        return cls_embedding

    def expandir_consulta(self, query, n=10):
        query = query.lower()
        termos_candidatos = []
        if query in self.w2v_model:
            termos_candidatos.append((query, 1.0))
        else:
            logging.warning(f"Termo da consulta '{query}' não encontrado no W2V. Usando apenas ele.")
            return [(query, 1.0)]
        try:
            similar_words = self.w2v_model.most_similar(query, topn=n-1)
            termos_candidatos.extend(similar_words)
        except Exception as e:
            logging.warning(f"Erro ao buscar similares para '{query}': {e}")
        return termos_candidatos

    def vetorizar_termos_candidatos(self, termos):
        logging.info(f"Vetorizando {len(termos)} termos candidatos com BERT...")
        v_candidatos = [self._get_bert_embedding(termo) for termo in termos]
        return v_candidatos

    def criar_vetor_consulta_ponderado(self, v_candidatos, pesos):
        if len(v_candidatos) != len(pesos):
            raise ValueError("Número de vetores candidatos e pesos não bate.")
        v_candidatos_np = np.array(v_candidatos)
        pesos_np = np.array(pesos).reshape(-1, 1)
        v_query_ponderado = np.sum(v_candidatos_np * pesos_np, axis=0)
        v_query_final = normalize(v_query_ponderado.reshape(1, -1), norm='l2', axis=1)
        return v_query_final.flatten()

    def ranking_por_similaridade(self, v_query_final):
        if self.v_tabelas is None:
            raise RuntimeError("Vetores das tabelas não foram carregados.")
        scores = np.dot(self.v_tabelas, v_query_final)
        indices_ordenados = np.argsort(scores)[::-1]
        ranking = []
        for idx in indices_ordenados:
            table_name = self.index_tabelas_map.get(idx, f"ID_{idx}_DESCONHECIDO")
            score = scores[idx]
            ranking.append((table_name, float(score)))
        return ranking
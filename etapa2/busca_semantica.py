# busca_semantica.py

import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
import logging
import os
import time

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes de Caminho (Configure com os caminhos da Etapa 1) ---
V_TABELAS_PATH = "v_tabelas.npy"
INDEX_PATH = "tabelas_index.json"
W2V_MODEL_PATH = os.path.join("nilc_model", "cbow_s300.txt")
BERT_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
# -------------------------------------------------------------

class BuscadorSemantico:
    """
    Encapsula toda a lógica de busca semântica, desde o carregamento
    de modelos até a execução do ranking.
    """
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
        """Carrega o tokenizador e o modelo BERT."""
        try:
            logging.info(f"Carregando modelo BERT: {BERT_MODEL_NAME}...")
            self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
            self.bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(self.device)
            self.bert_model.eval()
            logging.info("Modelo BERT carregado com sucesso.")
        except Exception as e:
            logging.error(f"Falha ao carregar o modelo BERT: {e}")
            raise

    def _load_w2v(self):
        """Carrega o modelo Word2Vec do NILC."""
        try:
            logging.info(f"Carregando modelo W2V de {W2V_MODEL_PATH} (pode demorar)...")
            start_time = time.time()
            self.w2v_model = KeyedVectors.load_word2vec_format(W2V_MODEL_PATH, binary=False)
            logging.info(f"Modelo W2V carregado em {time.time() - start_time:.2f}s")
        except Exception as e:
            logging.error(f"Falha ao carregar o modelo W2V. O arquivo existe? {e}")
            raise

    def _load_table_vectors(self):
        """Carrega os vetores da tabela e o índice. Pré-normaliza os vetores."""
        try:
            logging.info(f"Carregando vetores das tabelas de {V_TABELAS_PATH}...")
            # Carrega os vetores
            self.v_tabelas = np.load(V_TABELAS_PATH)
            
            # **Otimização Importante**: Pré-normaliza os vetores das tabelas (Norma L2)
            # Isso acelera o cálculo da similaridade de cosseno,
            # transformando-o em um simples produto escalar.
            self.v_tabelas = normalize(self.v_tabelas, norm='l2', axis=1)
            
            logging.info(f"Carregando índice de {INDEX_PATH}...")
            # Carrega o mapa de nome -> id
            with open(INDEX_PATH, 'r', encoding='utf-8') as f:
                self.tabelas_index_map = json.load(f)
                
            # Cria o mapa reverso de id -> nome
            self.index_tabelas_map = {v: k for k, v in self.tabelas_index_map.items()}
            
            logging.info(f"Vetores das tabelas (shape: {self.v_tabelas.shape}) e índices carregados.")
        except Exception as e:
            logging.error(f"Falha ao carregar os arquivos das tabelas: {e}")
            raise

    def _get_bert_embedding(self, text):
        """Gera o vetor (embedding) [CLS] para um dado texto usando BERT."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
        cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        return cls_embedding

    def expandir_consulta(self, query, n=10):
        """
        (Processo 1.1) Expande a consulta usando W2V.
        Retorna (n) termos mais similares.
        """
        query = query.lower()
        termos_candidatos = []
        
        # 1. Adiciona a própria consulta como o termo mais importante
        if query in self.w2v_model:
            termos_candidatos.append((query, 1.0))
        else:
            logging.warning(f"Termo da consulta '{query}' não encontrado no W2V. Usando apenas ele.")
            return [(query, 1.0)] # Retorna apenas a si mesmo

        # 2. Busca os (n-1) termos mais similares
        try:
            similar_words = self.w2v_model.most_similar(query, topn=n-1)
            termos_candidatos.extend(similar_words)
        except Exception as e:
            logging.warning(f"Erro ao buscar similares para '{query}': {e}")

        return termos_candidatos

    def vetorizar_termos_candidatos(self, termos):
        """
        (Processo 1.2) Pega uma lista de strings (termos) e retorna
        seus vetores BERT (V_CANDIDATOS).
        """
        logging.info(f"Vetorizando {len(termos)} termos candidatos com BERT...")
        v_candidatos = [self._get_bert_embedding(termo) for termo in termos]
        return v_candidatos

    def criar_vetor_consulta_ponderado(self, v_candidatos, pesos):
        """
        Cria o V_QUERY_FINAL (ou V_QUERY_EVOLUIDO).
        Recebe os vetores candidatos e os pesos (do AG ou manual).
        """
        if len(v_candidatos) != len(pesos):
            raise ValueError("Número de vetores candidatos e pesos não bate.")
            
        # 1. Calcula a soma ponderada: V_QUERY = (W[0]*V[0]) + (W[1]*V[1]) + ...
        # Converte para numpy arrays para multiplicação broadcast
        v_candidatos_np = np.array(v_candidatos)
        pesos_np = np.array(pesos).reshape(-1, 1) # Shape [N, 1]
        
        v_query_ponderado = np.sum(v_candidatos_np * pesos_np, axis=0)
        
        # 2. **Obrigatório**: Normaliza o vetor de consulta final (Norma L2)
        # Isso é crucial para que o produto escalar seja = similaridade de cosseno
        v_query_final = normalize(v_query_ponderado.reshape(1, -1), norm='l2', axis=1)
        
        return v_query_final.flatten() # Retorna como vetor 1D

    def ranking_por_similaridade(self, v_query_final):
        """
        (Processo 3.2) Calcula a similaridade do vetor de consulta final
        contra TODOS os vetores de tabela e retorna o ranking.
        """
        if self.v_tabelas is None:
            raise RuntimeError("Vetores das tabelas não foram carregados.")

        # 1. Calcula o Produto Escalar (Dot Product)
        # Como v_query_final e v_tabelas estão L2-normalizados,
        # o produto escalar é EXATAMENTE a similaridade de cosseno.
        # Shape (1000+, 768) . (768,) -> (1000+,)
        scores = np.dot(self.v_tabelas, v_query_final)
        
        # 2. Cria os pares (índice, pontuação)
        # argsort() retorna os *índices* que ordenariam o array
        # [::-1] inverte a ordem (do maior para o menor)
        indices_ordenados = np.argsort(scores)[::-1]
        
        # 3. Monta o ranking final
        ranking = []
        for idx in indices_ordenados:
            table_name = self.index_tabelas_map.get(idx, f"ID_{idx}_DESCONHECIDO")
            score = scores[idx]
            ranking.append((table_name, float(score))) # float() para ser serializável
            
        return ranking
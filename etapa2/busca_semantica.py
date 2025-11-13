# etapa2/busca_semantica.py

import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import logging
import os
import time
import sys
from functools import lru_cache
from tradutor import TradutorPTEN

# --- Mágica para importar o config.py da raiz ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config  # Importa o config.py da raiz
# ------------------------------------------------

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BuscadorSemantico:
    """ Encapsula toda a lógica de busca semântica. """

    def __init__(self):
        logging.info("Inicializando o BuscadorSemantico...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Usando dispositivo: {self.device}")

        # BERTs
        self.tokenizer_pt = None
        self.bert_pt = None
        self.tokenizer_en = None
        self.bert_en = None

        # W2V (BioWordVec reduzido já convertido .kv)
        self.w2v_model = None

        # Tabelas / índices
        self.v_tabelas = None
        self.tabelas_index_map = None  # Mapa: nome -> id
        self.index_tabelas_map = None  # Mapa: id -> nome

        # Cache local de embeddings BERT para acelerar múltiplas chamadas
        self._bert_embedding_cache = {}

        # Templates para avaliação contextual (em inglês)
        self._context_templates = [
            "The condition is related to {}.",
            "Clinical features associated with {} include {}.",
            "{} is associated with",
            "{} is a risk factor for"
        ]

        # Carregamentos
        self._load_bert()
        self._load_w2v()
        self._load_table_vectors()

        self.tradutor = TradutorPTEN()

        logging.info("BuscadorSemantico pronto para uso.")

    # ---------------- BERT Loading & Embeddings ----------------
    def _load_bert(self):
        """Carrega tokenizadores e modelos BERT (pt + opcional inglês)."""
        try:
            logging.info(f"Carregando modelo BERT principal (pt): {config.BERT_MODEL_NAME} …")
            self.tokenizer_pt = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
            self.bert_pt = BertModel.from_pretrained(config.BERT_MODEL_NAME).to(self.device)
            self.bert_pt.eval()
            logging.info("✅ Modelo BERT principal carregado com sucesso.")

            # Se houver um modelo BERT inglês configurado, carrega também
            if hasattr(config, "BERT_EN_MODEL_NAME") and config.BERT_EN_MODEL_NAME:
                logging.info(f"Carregando modelo BERT inglês: {config.BERT_EN_MODEL_NAME} …")
                self.tokenizer_en = BertTokenizer.from_pretrained(config.BERT_EN_MODEL_NAME)
                self.bert_en = BertModel.from_pretrained(config.BERT_EN_MODEL_NAME).to(self.device)
                self.bert_en.eval()
                logging.info("✅ Modelo BERT inglês carregado com sucesso.")
            else:
                self.tokenizer_en = None
                self.bert_en = None
                logging.info("⚠️ Nenhum modelo BERT inglês configurado.")
        except Exception as e:
            logging.error(f"❌ Falha ao carregar modelos BERT: {e}")
            raise

    def _get_bert_embedding(self, text, lang='auto'):
        """
        Gera o embedding CLS de um texto.
        lang='auto' escolhe automaticamente o modelo com base no W2V ativo.
        Usa cache para acelerar múltiplas consultas.
        """
        # chave para cache
        cache_key = f"{lang}||{text}"
        if cache_key in self._bert_embedding_cache:
            return self._bert_embedding_cache[cache_key]

        # Define o tokenizer/model a usar
        if lang == 'auto':
            if config.ACTIVE_W2V_MODEL.lower() == 'biowordvec' and self.bert_en:
                tokenizer, model = self.tokenizer_en, self.bert_en
            else:
                tokenizer, model = self.tokenizer_pt, self.bert_pt
        elif lang == 'en':
            if not self.bert_en:
                # fallback para pt se en não disponível
                tokenizer, model = self.tokenizer_pt, self.bert_pt
            else:
                tokenizer, model = self.tokenizer_en, self.bert_en
        else:
            tokenizer, model = self.tokenizer_pt, self.bert_pt

        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()

        # salva no cache
        self._bert_embedding_cache[cache_key] = cls_embedding
        return cls_embedding

    # ---------------- W2V Loading ----------------
    def _load_w2v(self):
        """Carrega o modelo Word2Vec (BioWordVec .kv já gerado)."""
        try:
            start_time = time.time()
            # Espera-se que config.BIOWORDVEC_MODEL_PATH aponte para o .kv salvo
            if not hasattr(config, "BIOWORDVEC_MODEL_PATH"):
                raise ValueError("config.BIOWORDVEC_MODEL_PATH não definido")

            model_path = config.BIOWORDVEC_MODEL_PATH
            logging.info(f"🧬 Carregando modelo W2V (BioWordVec reduzido) de {model_path} …")
            self.w2v_model = KeyedVectors.load(model_path, mmap='r')
            logging.info(f"✅ W2V carregado em {time.time() - start_time:.2f}s (vocab: {len(self.w2v_model.key_to_index):,})")
        except Exception as e:
            logging.error(f"❌ Falha ao carregar o modelo W2V: {e}")
            raise

    # ---------------- Table vectors ----------------
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

    # ---------------- Helpers para expansão ----------------
    def _w2v_cosine(self, a_word, b_word):
        """Retorna coseno entre vetores W2V (robusto). Retorna None se não houver vetores."""
        try:
            va = self.w2v_model.get_vector(a_word)
            vb = self.w2v_model.get_vector(b_word)
            # normaliza e calcula dot
            va = va / np.linalg.norm(va)
            vb = vb / np.linalg.norm(vb)
            return float(np.dot(va, vb))
        except Exception:
            return None

    def _cooccurrence_score(self, query, term):
        """
        Uma heurística simples de 'co-ocorrência' baseada em produto escalar bruto
        (pode ser ajustada). Aqui usamos dot entre vetores W2V não-normalizados como proxy.
        """
        try:
            va = self.w2v_model.get_vector(query)
            vb = self.w2v_model.get_vector(term)
            return float(np.dot(va, vb))
        except Exception:
            return 0.0

    def _contextual_similarity(self, query, term):
        """
        Calcula similaridade contextual usando templates (BERT).
        Retorna média das similaridades entre frases geradas.
        """
        sims = []
        # Force english BERT if W2V is English
        lang = 'en' if config.ACTIVE_W2V_MODEL.lower() == 'biowordvec' else 'pt'
        for tpl in self._context_templates:
            # some templates expect two slots; if so, use only first in a safe manner
            try:
                if "{} {}" in tpl or tpl.count("{}") > 1:
                    # if template has multiple placeholders, place term in second or adjust
                    phrase_q = tpl.format(query, "")
                    phrase_t = tpl.format(term, "")
                else:
                    phrase_q = tpl.format(query)
                    phrase_t = tpl.format(term)
            except Exception:
                phrase_q = tpl.format(query)
                phrase_t = tpl.format(term)

            vq = self._get_bert_embedding(phrase_q, lang=lang).reshape(1, -1)
            vt = self._get_bert_embedding(phrase_t, lang=lang).reshape(1, -1)
            sim = cosine_similarity(vq, vt)[0][0]
            sims.append(sim)
        # média das similaridades contextuais
        return float(np.mean(sims)) if sims else 0.0

    # ---------------- Expansão principal (pipeline híbrido) ----------------
    def expandir_consulta(self, query, n=10,
                         topn_w2v=300,
                         difflib_threshold=0.5,
                         w_bert=0.6,
                         w_w2v=0.3,
                         w_cooc=0.1,
                         filtro_bert=True):
        """
        Expande uma consulta médica com pipeline híbrido:
         - traduz automaticamente PT→EN se necessário
         - busca larga no W2V (topn_w2v)
         - filtra variações lexicais / substrings
         - calcula scores: W2V-cosine, cooccurrence, contextual-BERT
         - combina scores (ponderado)
         - traduz termos expandidos EN→PT se necessário
        """
        if not query:
            return []

        # ---------------- Tradução PT → EN (somente se BioWordVec ativo) ----------------
        query_original = query.strip().lower()
        traduzido_para_en = False
        if config.ACTIVE_W2V_MODEL.lower() == 'biowordvec':
            query_en = self.tradutor.pt_para_en(query_original)
            if query_en and query_en != query_original:
                logging.info(f"🌐 Traduzindo consulta (PT→EN): '{query_original}' → '{query_en}'")
                query = query_en.lower()
                traduzido_para_en = True
            else:
                query = query_original
        else:
            query = query_original

        # ---------------- Busca no W2V ----------------
        if query not in self.w2v_model:
            logging.warning(f"Termo da consulta '{query}' não encontrado no W2V. Retornando apenas o próprio termo.")
            return [(query_original, 1.0)]

        logging.info(f"Gerando candidatos W2V para '{query}' (topn={topn_w2v})...")
        candidatos_raw = self.w2v_model.most_similar(query, topn=topn_w2v)

        # ---------------- Filtragem lexical e diversidade ----------------
        candidatos = []
        seen = set()
        for termo, raw_score in candidatos_raw:
            termo_limpo = termo.lower().strip()

            if termo_limpo == query:
                continue
            lex_sim = difflib.SequenceMatcher(None, query, termo_limpo).ratio()
            if lex_sim >= difflib_threshold:
                continue
            if query.replace("-", "") in termo_limpo.replace("-", ""):
                continue
            if any(ch.isdigit() for ch in termo_limpo):
                continue
            if termo_limpo in seen:
                continue

            seen.add(termo_limpo)
            candidatos.append((termo_limpo, float(raw_score)))

        logging.info(f"{len(candidatos)} candidatos após filtragem lexical/diversidade.")

        if len(candidatos) < n:
            logging.info("Poucos candidatos após filtro — relaxando critérios (topn menor e lex threshold).")
            for termo, raw_score in candidatos_raw[:max(50, n * 3)]:
                termo_limpo = termo.lower().strip()
                if termo_limpo == query or termo_limpo in seen:
                    continue
                if query.replace("-", "") in termo_limpo.replace("-", ""):
                    continue
                seen.add(termo_limpo)
                candidatos.append((termo_limpo, float(raw_score)))
                if len(candidatos) >= n * 3:
                    break

        # ---------------- Cálculo dos scores híbridos ----------------
        resultados = []
        logging.info("Calculando scores híbridos (W2V-cosine, co-occurrence, contextual BERT)...")

        try:
            vq = self.w2v_model.get_vector(query)
            vq_norm = vq / np.linalg.norm(vq)
        except Exception:
            vq_norm = None

        for termo, raw_score in candidatos:
            w2v_cos = self._w2v_cosine(query, termo) if vq_norm is not None else 0.0
            cooc = self._cooccurrence_score(query, termo)

            bert_ctx = 0.0
            if filtro_bert:
                try:
                    bert_ctx = self._contextual_similarity(query, termo)
                except Exception as e:
                    logging.debug(f"Erro contextual BERT para {termo}: {e}")
                    bert_ctx = 0.0

            if w2v_cos is None:
                w2v_cos = 0.0
            w2v_cos_norm = (w2v_cos + 1.0) / 2.0
            cooc_norm = float(np.tanh(cooc / 1e4))
            bert_norm = float(np.clip(bert_ctx, -1.0, 1.0))
            bert_norm = (bert_norm + 1.0) / 2.0

            final_score = (w_bert * bert_norm) + (w_w2v * w2v_cos_norm) + (w_cooc * cooc_norm)

            resultados.append({
                "termo": termo,
                "raw_score": raw_score,
                "w2v_cos": w2v_cos_norm,
                "cooc": cooc_norm,
                "bert_ctx": bert_norm,
                "score": final_score
            })

        resultados.sort(key=lambda x: x["score"], reverse=True)

        finais = []
        seen_final = set()
        for item in resultados:
            termo = item["termo"]
            redundant = False
            for sel in seen_final:
                if difflib.SequenceMatcher(None, termo, sel).ratio() > 0.6:
                    redundant = True
                    break
            if redundant:
                continue
            seen_final.add(termo)
            finais.append((termo, float(item["score"])))
            if len(finais) >= n - 1:
                break

        # ---------------- Montagem do resultado final ----------------
        resultado_final = [(query, 1.0)]
        resultado_final.extend(finais)
        logging.info(f"Expandido '{query}' para {len(resultado_final)} termos (incluindo query).")

        # ---------------- Tradução EN → PT (somente se necessário) ----------------
        if traduzido_para_en:
            logging.info("🌐 Traduzindo termos expandidos EN→PT para exibição e uso no pipeline...")
            traduzidos = self.tradutor.traduz_lista([t for t, s in resultado_final], direcao="en2pt")
            resultado_final = list(zip(traduzidos, [s for t, s in resultado_final]))
            logging.info("✅ Tradução concluída.")

        return resultado_final

    # ---------------- Vetorização e composição da consulta ----------------
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

# ---------------- end of file ----------------

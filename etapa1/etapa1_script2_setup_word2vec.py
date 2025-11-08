# etapa1_script2_setup_word2vec.py

import os
import logging
from tqdm import tqdm

# a seguir libs opcionais para Hugging Face
try:
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file as safetensors_load
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from gensim.models import KeyedVectors

# --- Configuração ---
# Se preferir, você pode usar outro repo/versão de embeddings no Hugging Face
HF_REPO_ID = "nilc-nlp/word2vec-cbow-300d"

ZIP_PATH = "nilc_model.zip"  # mantido para compatibilidade, embora não use mais .zip deste repo
EXTRACT_DIR = "nilc_model"
MODEL_FILE_NAME = "cbow_s300.txt"
FINAL_MODEL_PATH = os.path.join(EXTRACT_DIR, MODEL_FILE_NAME)

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_from_hf(repo_id, extract_dir):
    """Baixa embeddings e vocab do HF e coloca em extract_dir, na forma que o seu script espera."""
    logging.info(f"Baixando modelo do Hugging Face: {repo_id} …")
    if not HF_AVAILABLE:
        logging.error("huggingface_hub ou safetensors não instalados. Instale com: pip install huggingface-hub safetensors")
        return False

    # Baixa os arquivos
    emb_file = hf_hub_download(repo_id=repo_id, filename="embeddings.safetensors")
    vocab_file = hf_hub_download(repo_id=repo_id, filename="vocab.txt")

    logging.info(f"Arquivos baixados em cache: {emb_file}, {vocab_file}")

    # Cria diretório target
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Carrega embeddings
    data = safetensors_load(emb_file)
    vectors = data["embeddings"]
    logging.info(f"Shape dos vetores: {vectors.shape}")

    # Carrega vocabulário
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = [w.strip() for w in f]

    # Converte para formato word2vec text para compatibilidade com Gensim
    out_txt = os.path.join(extract_dir, MODEL_FILE_NAME)
    logging.info(f"Salvando modelo em formato word2vec text em: {out_txt} …")

    with open(out_txt, "w", encoding="utf-8") as out_f:
        # escreve header: número de palavras + dimensões
        out_f.write(f"{len(vocab)} {vectors.shape[1]}\n")
        for idx, word in enumerate(vocab):
            vec = vectors[idx]
            vec_str = " ".join(str(x) for x in vec.tolist())
            out_f.write(f"{word} {vec_str}\n")

    logging.info("Conversão concluída.")
    return True

def test_model(model_path):
    """Testa o carregamento do modelo e busca palavras similares."""
    if not os.path.exists(model_path):
        logging.warning(f"Arquivo de modelo {model_path} não encontrado. Pulando teste.")
        return

    logging.info("Testando o carregamento do modelo Word2Vec (pode demorar)...")
    try:
        model = KeyedVectors.load_word2vec_format(model_path, binary=False)
        logging.info("Modelo carregado com sucesso.")

        test_word = "rei"
        if test_word in model:
            similar_words = model.most_similar(test_word, topn=5)
            logging.info(f"Teste de similaridade para '{test_word}':")
            for word, score in similar_words:
                logging.info(f"  {word}: {score:.4f}")
        else:
            logging.warning(f"Palavra de teste '{test_word}' não encontrada no vocabulário.")

    except Exception as e:
        logging.error(f"Falha ao carregar ou testar o modelo W2V: {e}")
        logging.error("Verifique se o formato está correto ou tente outro arquivo.")

def main():
    """Função principal para baixar/configurar o Word2Vec."""
    logging.info("=== INICIANDO ETAPA 1 (SCRIPT 2) ===")

    # Baixar + converter
    success = download_from_hf(HF_REPO_ID, EXTRACT_DIR)
    if not success:
        logging.error("O download ou conversão falhou — verifique as mensagens acima.")
        return

    # Testar o modelo
    test_model(FINAL_MODEL_PATH)

    logging.info("--- ETAPA 1 (SCRIPT 2) CONCLUÍDA ---")
    logging.info(f"O modelo Word2Vec está pronto para uso em: {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    main()

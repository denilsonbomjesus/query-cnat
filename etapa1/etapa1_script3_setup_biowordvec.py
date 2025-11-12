# etapa1_script3_setup_biowordvec.py

import logging
import os
import requests
from gensim.models import KeyedVectors

# ================================================================
# CONFIGURAÇÃO DO LOG
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ================================================================
# PARÂMETROS DO MODELO
# ================================================================
BIOWORDVEC_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin"
BIOWORDVEC_PATH = "biowordvec_model/BioWordVec_PubMed_MIMICIII_d200.vec.bin"
os.makedirs("biowordvec_model", exist_ok=True)

# ================================================================
# FUNÇÃO PARA BAIXAR O MODELO
# ================================================================
def baixar_modelo(url, destino):
    if os.path.exists(destino):
        logging.info("✅ Modelo já existe localmente — pulando download.")
        return

    logging.info(f"📥 Baixando modelo BioWordVec de {url} …")
    try:
        resposta = requests.get(url, stream=True)
        resposta.raise_for_status()

        total = int(resposta.headers.get('content-length', 0))
        comeco = 0
        bloco = 1024 * 1024  # 1 MB

        with open(destino, 'wb') as f:
            for dados in resposta.iter_content(block_size := bloco):
                comeco += len(dados)
                f.write(dados)
                porcentagem = (comeco / total) * 100 if total else 0
                print(f"\rProgresso: {porcentagem:.2f}%", end="")

        print()
        logging.info("✅ Download concluído com sucesso.")

    except Exception as e:
        logging.error(f"❌ Erro ao baixar o modelo: {e}")
        raise

# ================================================================
# FUNÇÃO PRINCIPAL
# ================================================================
# def carregar_biowordvec():
#     baixar_modelo(BIOWORDVEC_URL, BIOWORDVEC_PATH)
#     logging.info("📦 Carregando modelo BioWordVec … (isso pode levar alguns minutos)")
#     modelo = KeyedVectors.load_word2vec_format(BIOWORDVEC_PATH, binary=True)
#     logging.info("✅ Modelo BioWordVec carregado com sucesso!")
#     return modelo

# ================================================================
# EXECUÇÃO PRINCIPAL
# ================================================================
# if __name__ == "__main__":
#     logging.info("=== INICIANDO ETAPA 1 (SCRIPT 3) ===")
#     try:
#         modelo = carregar_biowordvec()

#         # Exemplo de uso:
#         termo = input("\nDigite um termo médico em inglês para ver palavras semelhantes: ").strip()
#         if termo in modelo.key_to_index:
#             similares = modelo.most_similar(termo, topn=10)
#             print(f"\n🔍 Termos semelhantes a '{termo}':")
#             for palavra, score in similares:
#                 print(f"  - {palavra} ({score:.4f})")
#         else:
#             print(f"⚠️ O termo '{termo}' não foi encontrado no vocabulário do modelo.")
#     except Exception as e:
#         logging.error(f"❌ Falha durante a execução: {e}")

# ================================================================
# SAÍDA DA EXECUÇÃO
# ================================================================

# 2025-11-12 18:35:01,469 - INFO - 📦 Carregando modelo BioWordVec … (isso pode levar alguns minutos)
# 2025-11-12 18:35:01,472 - INFO - loading projection weights from biowordvec_model/BioWordVec_PubMed_MIMICIII_d200.vec.bin
# 2025-11-12 18:35:01,782 - ERROR - ❌ Falha durante a execução: Unable to allocate 12.3 GiB for an array with shape (16545452, 200) and data type float32

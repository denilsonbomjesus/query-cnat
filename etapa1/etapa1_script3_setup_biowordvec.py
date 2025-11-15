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

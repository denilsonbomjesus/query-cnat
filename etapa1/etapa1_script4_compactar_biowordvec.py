# etapa1_script4_compactar_biowordvec.py

import logging
import os
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
BIOWORDVEC_PATH = "biowordvec_model/BioWordVec_PubMed_MIMICIII_d200.vec.bin"
OUTPUT_DIR = "modelos"
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_DIR, "biowordvec_500k.kv")
LIMIT = 500000  # Número máximo de palavras a carregar

# ================================================================
# FUNÇÃO PRINCIPAL
# ================================================================
def compactar_biowordvec(skip_test=False):
    """Compacta o modelo BioWordVec.

    Args:
        skip_test: se True, pula o teste interativo (útil para automação).
    """
    if not os.path.exists(BIOWORDVEC_PATH):
        logging.error(f"❌ Arquivo de modelo não encontrado: {BIOWORDVEC_PATH}")
        logging.error("Certifique-se de ter executado o script3 antes deste.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info(f"📦 Carregando parcialmente o modelo BioWordVec (limit={LIMIT:,} palavras)...")
    try:
        modelo = KeyedVectors.load_word2vec_format(
            BIOWORDVEC_PATH,
            binary=True,
            limit=LIMIT
        )
        logging.info("✅ Modelo carregado com sucesso!")

        logging.info(f"💾 Salvando modelo reduzido em {OUTPUT_MODEL_PATH} …")
        modelo.save(OUTPUT_MODEL_PATH)
        logging.info("✅ Modelo reduzido salvo com sucesso!")

        # Pequeno teste opcional (apenas interativo)
        if not skip_test:
            termo_teste = input("\nDigite um termo médico em inglês para testar: ").strip()
            if termo_teste in modelo.key_to_index:
                similares = modelo.most_similar(termo_teste, topn=10)
                print(f"\n🔍 Termos semelhantes a '{termo_teste}':")
                for palavra, score in similares:
                    print(f"  - {palavra} ({score:.4f})")
            else:
                print(f"⚠️ O termo '{termo_teste}' não foi encontrado no vocabulário.")

    except Exception as e:
        logging.error(f"❌ Erro ao carregar ou salvar modelo: {e}")

# ================================================================
# EXECUÇÃO PRINCIPAL
# ================================================================
if __name__ == "__main__":
    import sys
    skip_test = "--skip-test" in sys.argv
    logging.info("=== INICIANDO ETAPA 1 (SCRIPT 4): COMPACTAR BIOWORDVEC ===")
    compactar_biowordvec(skip_test=skip_test)
    logging.info("--- ETAPA 1 (SCRIPT 4) CONCLUÍDA ---")

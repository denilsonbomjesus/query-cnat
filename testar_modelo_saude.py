# ===============================================================
# TESTE DO MODELO WORD2VEC - DOMÍNIO SAÚDE (PUCRS)
# ===============================================================

import os
import requests
import tarfile
from gensim.models import KeyedVectors

MODEL_DIR = "modelos_saude"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URL = "https://www.inf.pucrs.br/linatural/wp-content/uploads/2021/07/health_word2vec_cbow_300.tar.gz"
MODEL_TAR_PATH = os.path.join(MODEL_DIR, "health_word2vec_cbow_300.tar.gz")

# ===============================================================
# DOWNLOAD DO MODELO (CORRIGIDO)
# ===============================================================
def download_model():
    if not os.path.exists(MODEL_TAR_PATH):
        print("⬇️  Baixando o modelo de Word2Vec da PUCRS (domínio saúde)...")
        response = requests.get(MODEL_URL, stream=True)
        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(MODEL_TAR_PATH, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        done = int(50 * downloaded / total)
                        print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB", end="")
                    else:
                        print(f"\rBaixados: {downloaded/1024/1024:.1f} MB", end="")
        print("\n✅ Download concluído!")
    else:
        print("📁 Modelo já baixado.")

# ===============================================================
# EXTRAÇÃO DO ARQUIVO
# ===============================================================
def extract_model():
    print("📦 Extraindo modelo...")
    with tarfile.open(MODEL_TAR_PATH, "r:gz") as tar:
        tar.extractall(MODEL_DIR)
    print("✅ Arquivo extraído com sucesso!")

# ===============================================================
# CARREGAMENTO
# ===============================================================
def load_model():
    print("🧠 Carregando o modelo Word2Vec em memória...")
    model_path = os.path.join(MODEL_DIR, "health_word2vec_cbow_300.model")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "health_word2vec_cbow_300.bin")
    model = KeyedVectors.load(model_path, mmap='r')
    print("✅ Modelo carregado com sucesso!")
    return model

# ===============================================================
# TESTE SEMÂNTICO
# ===============================================================
def test_model(model):
    palavras_teste = ["hospital", "doença", "médico", "paciente", "vacina"]
    for palavra in palavras_teste:
        if palavra in model.key_to_index:
            similares = model.most_similar(palavra, topn=5)
            print(f"\n🔹 Palavras mais semelhantes a '{palavra}':")
            for termo, score in similares:
                print(f"   {termo:<20} ({score:.4f})")
        else:
            print(f"\n⚠️ Palavra '{palavra}' não encontrada no vocabulário.")

# ===============================================================
# EXECUÇÃO
# ===============================================================
if __name__ == "__main__":
    download_model()
    extract_model()
    model = load_model()
    test_model(model)

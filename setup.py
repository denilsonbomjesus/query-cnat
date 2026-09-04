#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup.py — Preparação completa do ambiente do sistema Query-CNAT (Windows / Linux / macOS).

Executa, de forma idempotente (só faz o que ainda não existe):

  1. Verifica os pré-requisitos (Python, metadados de entrada, espaço em disco);
  2. Cria o virtualenv (venv/) e instala as dependências (requirements.txt);
  3. Garante o modelo BioWordVec compactado (modelos/biowordvec_500k.kv):
       - baixa o binário original da NCBI (~13 GB) se necessário (script3) e
       - gera a versão compactada usada em produção (script4);
  4. Garante os vetores das tabelas (asset/v_tabelas.npy + asset/tabelas_index.json):
       - se faltarem, roda a Etapa 1 (etapa1/etapa1_script1_vetorizar_tabelas.py);
  5. Pré-baixa os modelos BERT para o cache HuggingFace (1º uso do app fica offline);
  6. Imprime como ativar o ambiente e como iniciar o app (streamlit run app.py).

Exemplos de uso:
    python setup.py                     # prepara tudo (idempotente)
    python setup.py --dry-run           # mostra o que existe / o que falta, sem executar
    python setup.py --skip-bert         # pula o pré-download dos modelos BERT
    python setup.py --skip-install      # cria o venv, mas não instala o requirements.txt
    python setup.py --force-vectorize   # regenera os vetores das tabelas do zero
    python setup.py --run               # ao final, inicia o app (streamlit run app.py)
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

# ================================================================
# CONSTANTES / CAMINHOS
# ================================================================

ROOT = os.path.dirname(os.path.abspath(__file__))
SISTEMA = platform.system()  # 'Windows' | 'Linux' | 'Darwin'

VENV_DIR = os.path.join(ROOT, "venv")
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")
APP = os.path.join(ROOT, "app.py")

# Ativos da Etapa 1
METADATA_PATH = os.path.join(ROOT, "asset", "metadata_advanced_consolidated.json")
V_TABELAS_PATH = os.path.join(ROOT, "asset", "v_tabelas.npy")
INDEX_PATH = os.path.join(ROOT, "asset", "tabelas_index.json")

# Modelos W2V
BIOWORDVEC_BIN = os.path.join(ROOT, "biowordvec_model", "BioWordVec_PubMed_MIMICIII_d200.vec.bin")
W2V_KV_PATH = os.path.join(ROOT, "modelos", "biowordvec_500k.kv")

# Scripts orquestrados
SCRIPT1 = os.path.join(ROOT, "etapa1", "etapa1_script1_vetorizar_tabelas.py")
SCRIPT3 = os.path.join(ROOT, "etapa1", "etapa1_script3_setup_biowordvec.py")
SCRIPT4 = os.path.join(ROOT, "etapa1", "etapa1_script4_compactar_biowordvec.py")

# Modelos BERT pré-baixados no cache HuggingFace
MODELOS_BERT = ("pucpr/biobertpt-all", "dmis-lab/biobert-base-cased-v1.1")

PYTHON_MINIMO = (3, 9)

EMOJIS = {"ok": "✅", "falta": "⬜", "aviso": "⚠️", "info": "ℹ️", "erro": "❌"}


# ================================================================
# HELPERS
# ================================================================

def hr(titulo=""):
    largura = 62
    print("\n" + "=" * largura)
    if titulo:
        print(f" {titulo}")
        print("=" * largura)


def caminho_venv_python():
    """Retorna o interpretador Python dentro do venv (cross-platform)."""
    if SISTEMA == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def venv_existe():
    return os.path.exists(caminho_venv_python())


def requisitos_instalados():
    """Verifica se os pacotes principais importam dentro do venv."""
    if not venv_existe():
        return False
    modulos = "streamlit, torch, transformers, gensim, sklearn, numpy, pygad, deep_translator, requests, tqdm, joblib, matplotlib"
    cmd = [caminho_venv_python(), "-c", f"import {modulos}"]
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def cache_huggingface():
    """Diretório do cache HuggingFace (respeita HF_HOME se definido)."""
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    return os.path.join(hf_home, "hub")


def bert_em_cache():
    """Retorna True se os modelos BERT já estiverem no cache HF (heurística por nome de pasta)."""
    hub = cache_huggingface()
    faltando = []
    for modelo in MODELOS_BERT:
        pasta = "models--" + modelo.replace("/", "--")
        if not os.path.isdir(os.path.join(hub, pasta)):
            faltando.append(modelo)
    return faltando  # lista vazia = tudo em cache


def espaco_livre_gb():
    try:
        return shutil.disk_usage(ROOT).free / (1024 ** 3)
    except OSError:
        return None


def rodar(cmd, descricao=""):
    """Executa um comando e propaga erros com mensagem clara."""
    if descricao:
        print(f"\n{EMOJIS['info']} {descricao}")
        print(f"    Comando: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except KeyboardInterrupt:
        # Ctrl+C no console atinge o subprocesso E o setup.py (Windows e
        # Unix). Sem este tratamento, o usuário vê um traceback enorme.
        print(f"\n{EMOJIS['aviso']} Setup interrompido pelo usuário (Ctrl+C).")
        print("    Nenhum arquivo foi corrompido — basta rodar novamente: python setup.py")
        sys.exit(130)
    except subprocess.CalledProcessError as e:
        print(f"\n{EMOJIS['erro']} Falha ao executar: {' '.join(cmd)}")
        print(f"    Erro: {e}")
        sys.exit(1)


# ================================================================
# ETAPAS
# ================================================================

def verificar_pre_requisitos():
    """Python mínimo e metadados de entrada."""
    problemas = []
    if sys.version_info < PYTHON_MINIMO:
        problemas.append(f"Python {PYTHON_MINIMO[0]}.{PYTHON_MINIMO[1]}+ necessário (atual: {sys.version.split()[0]}).")
    if not os.path.exists(METADATA_PATH):
        problemas.append(f"Metadados não encontrados em: {METADATA_PATH}")
    if problemas:
        for p in problemas:
            print(f"{EMOJIS['erro']} {p}")
        print("\nCorrija os problemas acima e rode novamente: python setup.py")
        sys.exit(1)
    print(f"{EMOJIS['ok']} Pré-requisitos OK (Python {sys.version.split()[0]})")


def criar_venv():
    """Cria o virtualenv se não existir (cross-platform)."""
    if venv_existe():
        print(f"{EMOJIS['ok']} venv já existe em: {VENV_DIR}")
        return
    print(f"{EMOJIS['info']} Criando virtualenv em {VENV_DIR} ...")
    rodar([sys.executable, "-m", "venv", VENV_DIR], "Criando ambiente virtual")


def instalar_requisitos():
    """Instala (ou atualiza) o requirements.txt no venv."""
    if not os.path.exists(REQUIREMENTS):
        print(f"{EMOJIS['aviso']} requirements.txt não encontrado — pulando instalação.")
        return
    print(f"{EMOJIS['info']} Instalando dependências do {REQUIREMENTS} ...")
    # Sem passo de '--upgrade pip': o pip do venv recém-criado já é recente
    # o suficiente, e a atualização silenciosa (-q) parecia travada e era
    # interrompida (erro 'Operation cancelled by user' no Windows).
    rodar([caminho_venv_python(), "-m", "pip", "install", "-r", REQUIREMENTS,
           "--no-input", "--disable-pip-version-check"],
          "Instalando dependências (pode levar vários minutos)")


def garantir_biowordvec():
    """Garante modelos/biowordvec_500k.kv (baixa binário + compacta se preciso)."""
    if os.path.exists(W2V_KV_PATH):
        print(f"{EMOJIS['ok']} BioWordVec compactado já existe: {W2V_KV_PATH}")
        return

    if not os.path.exists(BIOWORDVEC_BIN):
        livre = espaco_livre_gb()
        if livre is not None and livre < 20:
            print(f"{EMOJIS['aviso']} Pouco espaço em disco ({livre:.1f} GB livres) para baixar ~13 GB.")
        rodar([caminho_venv_python(), SCRIPT3],
              "Baixando BioWordVec original da NCBI (~13 GB, pode demorar)")
    else:
        print(f"{EMOJIS['ok']} BioWordVec original já baixado: {BIOWORDVEC_BIN}")

    rodar([caminho_venv_python(), SCRIPT4, "--skip-test"],
          "Compactando BioWordVec (gera modelos/biowordvec_500k.kv)")

    if not os.path.exists(W2V_KV_PATH):
        print(f"{EMOJIS['erro']} Compactação não gerou {W2V_KV_PATH}. Verifique o script4.")
        sys.exit(1)


def garantir_vetores_tabelas(force=False):
    """Garante asset/v_tabelas.npy + asset/tabelas_index.json (Etapa 1 / script1)."""
    tem_vetores = os.path.exists(V_TABELAS_PATH)
    tem_indice = os.path.exists(INDEX_PATH)

    if tem_vetores and tem_indice and not force:
        print(f"{EMOJIS['ok']} Vetores das tabelas já existem: {V_TABELAS_PATH} + {INDEX_PATH}")
        return

    if force:
        print(f"{EMOJIS['aviso']} --force-vectorize: regenerando vetores das tabelas...")
    else:
        print(f"{EMOJIS['info']} Vetores das tabelas incompletos — rodando Etapa 1 (vetorização)...")

    rodar([caminho_venv_python(), SCRIPT1],
          "Vetorizando tabelas com BERT (Etapa 1)")

    if not (os.path.exists(V_TABELAS_PATH) and os.path.exists(INDEX_PATH)):
        print(f"{EMOJIS['erro']} A vetorização não gerou os arquivos esperados em asset/.")
        sys.exit(1)


def pre_baixar_bert():
    """Pré-baixa os modelos BERT no cache HuggingFace (1º uso do app fica offline)."""
    faltando = bert_em_cache()
    if not faltando:
        print(f"{EMOJIS['ok']} Modelos BERT já estão no cache HuggingFace.")
        return

    print(f"{EMOJIS['info']} Pré-baixando modelos BERT no cache HuggingFace: {', '.join(faltando)} ...")
    codigo = (
        "from transformers import BertModel, BertTokenizer\n"
        "import config\n"
        "for m in (config.BERT_MODEL_NAME, config.BERT_EN_MODEL_NAME):\n"
        "    print(f'Baixando {m} ...')\n"
        "    BertTokenizer.from_pretrained(m)\n"
        "    BertModel.from_pretrained(m)\n"
    )
    rodar([caminho_venv_python(), "-c", codigo],
          "Baixando modelos BERT (PT e EN)")


def verificar_consistencia():
    """Checagem leve: nº de tabelas nos metadados vs vetores gerados."""
    if not (os.path.exists(V_TABELAS_PATH) and os.path.exists(METADATA_PATH)):
        return
    codigo = (
        "import json, numpy as np, config\n"
        "meta = json.load(open(config.METADATA_ADVANCED_FILE_PATH, encoding='utf-8'))\n"
        "n_meta = sum(1 for t in meta if t.get('table_name'))\n"
        "v = np.load(config.V_TABELAS_PATH, mmap_mode='r')\n"
        "print(f'  Tabelas nos metadados: {n_meta} | Vetores gerados: {v.shape[0]}')\n"
        "if n_meta != v.shape[0]:\n"
        "    print('  ⚠️ Divergência entre metadados e vetores — considere --force-vectorize.')\n"
    )
    try:
        subprocess.run([caminho_venv_python(), "-c", codigo], cwd=ROOT, check=True)
    except subprocess.CalledProcessError:
        print(f"{EMOJIS['aviso']} Não foi possível checar consistência dos vetores.")


def instrucoes_finais(run_app=False):
    """Imprime como ativar o ambiente e iniciar o app."""
    hr(" PRÓXIMOS PASSOS ")

    if SISTEMA == "Windows":
        ativar = "venv\\Scripts\\activate"
        ativar_pwsh = "venv\\Scripts\\Activate.ps1"
        print(f"  Ativar o ambiente (cmd):       {ativar}")
        print(f"  Ativar o ambiente (PowerShell): {ativar_pwsh}")
    else:
        ativar = "source venv/bin/activate"
        print(f"  Ativar o ambiente:              {ativar}")

    print(f"  Iniciar a interface:            streamlit run app.py")
    print(f"  (ou rode:                       python setup.py --run)")

    if run_app:
        print(f"\n{EMOJIS['info']} Iniciando o app...")
        rodar([caminho_venv_python(), "-m", "streamlit", "run", APP],
              "Iniciando a interface (streamlit run app.py)")
    else:
        print(f"\n{EMOJIS['ok']} Setup concluído! Basta iniciar o app com o comando acima.")


# ================================================================
# STATUS (MODO DRY-RUN)
# ================================================================

def coletar_status():
    """Retorna dicionário com o estado de cada item do ambiente."""
    return {
        "Python (sistema)": (sys.version.split()[0], True),
        "virtualenv (venv/)": (VENV_DIR, venv_existe()),
        "requirements instalados": (REQUIREMENTS, requisitos_instalados()),
        "Metadados (asset/metadata_advanced_consolidated.json)": (METADATA_PATH, os.path.exists(METADATA_PATH)),
        "BioWordVec compactado (modelos/biowordvec_500k.kv)": (W2V_KV_PATH, os.path.exists(W2V_KV_PATH)),
        "BioWordVec original (biowordvec_model/*.bin)": (BIOWORDVEC_BIN, os.path.exists(BIOWORDVEC_BIN)),
        "Vetores das tabelas (asset/v_tabelas.npy)": (V_TABELAS_PATH, os.path.exists(V_TABELAS_PATH)),
        "Índice das tabelas (asset/tabelas_index.json)": (INDEX_PATH, os.path.exists(INDEX_PATH)),
        "Modelos BERT (cache HuggingFace)": (cache_huggingface(), not bert_em_cache()),
    }


def mostrar_status():
    """Imprime a tabela de status sem executar nada."""
    hr(" QUERY-CNAT — VERIFICAÇÃO DO AMBIENTE ")
    print(f"  Sistema: {SISTEMA} | Diretório: {ROOT}\n")
    for nome, (caminho, ok) in coletar_status().items():
        marcador = EMOJIS["ok"] if ok else EMOJIS["falta"]
        print(f"  {marcador} {nome}")
        if caminho:
            print(f"        → {caminho}")
    print()


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preparação offline completa do sistema Query-CNAT (Windows/Linux/macOS).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python setup.py                prepara tudo (idempotente)\n"
            "  python setup.py --dry-run      só mostra o que existe/falta\n"
            "  python setup.py --skip-bert    pula pré-download dos BERTs\n"
            "  python setup.py --run          inicia o app ao final\n"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="apenas verifica e mostra o status")
    parser.add_argument("--skip-bert", action="store_true", help="pula o pré-download dos modelos BERT")
    parser.add_argument("--skip-install", action="store_true", help="cria o venv mas não instala requirements")
    parser.add_argument("--force-vectorize", action="store_true", help="regenera os vetores das tabelas")
    parser.add_argument("--run", action="store_true", help="inicia o app (streamlit run app.py) ao final")
    args = parser.parse_args()

    if args.dry_run:
        mostrar_status()
        print("  (modo --dry-run: nada foi executado ou baixado)\n")
        return

    hr(" QUERY-CNAT — SETUP COMPLETO ")

    verificar_pre_requisitos()

    criar_venv()
    if not args.skip_install:
        instalar_requisitos()
    else:
        print(f"{EMOJIS['info']} --skip-install: dependências não instaladas.")

    garantir_biowordvec()
    garantir_vetores_tabelas(force=args.force_vectorize)

    if not args.skip_bert:
        pre_baixar_bert()
    else:
        print(f"{EMOJIS['info']} --skip-bert: pré-download dos BERTs pulado (serão baixados no 1º uso).")

    verificar_consistencia()
    instrucoes_finais(run_app=args.run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Rede de segurança para interrupções fora de rodar() (ex.: durante
        # a criação do venv): mensagem limpa em vez de traceback.
        print(f"\n{EMOJIS['aviso']} Setup interrompido pelo usuário (Ctrl+C).")
        print("    Nenhum arquivo foi corrompido — basta rodar novamente: python setup.py")
        sys.exit(130)
# Query CNAT

Motor de **busca semântica otimizada por Algoritmo Genético (GA)** para datasets de saúde (SUS/CADSUS), capaz de encontrar as tabelas mais relevantes para uma consulta em linguagem natural e selecionar as colunas (features) mais importantes de cada tabela.

O sistema combina:

- **Modelos de linguagem** — BERT (português biomédico `pucpr/biobertpt-all` e inglês biomédico `dmis-lab/biobert-base-cased-v1.1`) para vetorização semântica;
- **Modelo Word2Vec biomédico** — BioWordVec (PubMed + MIMIC-III) para expansão de consultas médicas em inglês;
- **Tradução automática PT→EN/EN→PT** — via Google Translate (`deep-translator`), com cache e retry;
- **Algoritmos Genéticos (PyGAD)** — um GA contínuo otimiza os pesos dos termos expandidos da consulta e um GA binário seleciona as colunas mais relevantes de cada tabela.

---

## Sumário

1. [Arquitetura do sistema](#arquitetura-do-sistema)
2. [Pré-requisitos](#pré-requisitos)
3. [Instalação (setup automático)](#instalação-setup-automático)
4. [Instalação manual (alternativa)](#instalação-manual-alternativa)
5. [Executando o aplicativo](#executando-o-aplicativo)
6. [Fase offline (Etapa 1) em detalhe](#fase-offline-etapa-1-em-detalhe)
7. [Executando os testes](#executando-os-testes)
8. [Estrutura do projeto](#estrutura-do-projeto)
9. [Configuração (config.py)](#configuração-configpy)
10. [Solução de problemas (FAQ)](#solução-de-problemas-faq)
11. [Notas de produção](#notas-de-produção)

---

## Arquitetura do sistema

O fluxo é dividido em **fases offline** (preparação dos modelos e artefatos) e o **pipeline online** (busca e otimização por consulta).

```
┌─────────────────────────── FASE OFFLINE (Etapa 1) ───────────────────────────┐
│                                                                              │
│  asset/metadata_advanced_consolidated.json  (metadados das ~1.100 tabelas)   │
│        │                                                                     │
│        ▼                                                                     │
│  etapa1/etapa1_script1_vetorizar_tabelas.py                                  │
│        │  (BERT pt: pucpr/biobertpt-all + mean pooling)                      │
│        ▼                                                                     │
│  asset/v_tabelas.npy            asset/tabelas_index.json                     │
│  (vetores 768-d de cada tabela)  (nome da tabela → índice)                   │
│                                                                              │
│  etapa1/etapa1_script3_setup_biowordvec.py  →  biowordvec_model/*.bin (~13GB)│
│        │  (download da NCBI)                                                 │
│        ▼                                                                     │
│  etapa1/etapa1_script4_compactar_biowordvec.py  →  modelos/biowordvec_500k.kv│
│              (compacta com limite de 500k palavras)                          │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── PIPELINE ONLINE (Etapas 2 e 3) ───────────────────┐
│                                                                              │
│  app.py (Streamlit)                                                          │
│    │                                                                         │
│    ▼                                                                         │
│  etapa2/busca_semantica.py  (BuscadorSemantico)                              │
│    │  1. Traduz a consulta PT → EN (se necessário)                           │
│    │  2. Expande a consulta com o BioWordVec (termos semelhantes)            │
│    │  3. Vetoriza os termos com BERT                                          │
│    ▼                                                                         │
│  etapa3/otimizador_ga.py  (GA contínuo — otimiza os pesos dos termos)        │
│    │  4. Ranking das tabelas por similaridade da consulta otimizada          │
│    ▼                                                                         │
│  etapa3/otimizador_ga_features.py  (GA binário — seleciona colunas)          │
│    │  5. Para cada tabela do Top-N, seleciona as features mais relevantes    │
│    ▼                                                                         │
│  etapa2/metadata_loader.py  (detalhes: schema, row_count, chave primária)    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Entradas e saídas

| Item | Caminho | Origem |
|---|---|---|
| Metadados das tabelas (entrada) | `asset/metadata_advanced_consolidated.json` | Versionado no repositório |
| Vetores das tabelas | `asset/v_tabelas.npy` | Gerado pela Etapa 1 (versionado) |
| Índice tabela → vetor | `asset/tabelas_index.json` | Gerado pela Etapa 1 (versionado) |
| BioWordVec original | `biowordvec_model/BioWordVec_PubMed_MIMICIII_d200.vec.bin` | Download (não versionado) |
| BioWordVec compactado | `modelos/biowordvec_500k.kv` | Gerado pela Etapa 1 (não versionado) |
| Modelos BERT | cache HuggingFace (`~/.cache/huggingface`) | Download automático via `transformers` |

> Os arquivos pesados (`biowordvec_model/`, `modelos/`, `venv/`) são ignorados pelo `.gitignore` e **não** devem ser commitados.

---

## Pré-requisitos

- **Python 3.9 ou superior** (testado com 3.12)
- **~20 GB de espaço em disco** (13 GB só para o binário do BioWordVec)
- **Conexão com a internet** no primeiro setup (downloads) e para a tradução PT→EN durante o uso
- Sistema operacional: Windows, Linux ou macOS

---

## Instalação (setup automático)

O `setup.py` faz **toda** a preparação do ambiente de uma vez, de forma **idempotente** (verifica o que já existe e só baixa/gera o que estiver faltando):

1. Cria o ambiente virtual `venv/`;
2. Instala as dependências do `requirements.txt`;
3. Baixa o modelo BioWordVec da NCBI (se ausente);
4. Compacta o BioWordVec para `modelos/biowordvec_500k.kv` (se ausente);
5. Gera os vetores das tabelas (Etapa 1) — somente se `asset/v_tabelas.npy`/`tabelas_index.json` estiverem ausentes;
6. Pré-baixa os modelos BERT para o cache HuggingFace;
7. Verifica a consistência (nº de tabelas nos metadados vs. vetores gerados) e mostra como iniciar o app.

```bash
git clone https://github.com/denilsonbomjesus/query-cnat.git
cd query-cnat

python setup.py            # Windows
python3 setup.py           # Linux/macOS
```

Ao final, o script imprime o comando para ativar o ambiente e iniciar o app.

### Flags do `setup.py`

| Flag | Descrição |
|---|---|
| `--dry-run` | Apenas verifica e mostra o status de cada item (não baixa nada) |
| `--skip-bert` | Pula o pré-download dos modelos BERT (serão baixados no 1º uso do app) |
| `--skip-install` | Cria o venv, mas não instala o `requirements.txt` |
| `--force-vectorize` | Regenera os vetores das tabelas do zero (Etapa 1) |
| `--run` | Após o setup, inicia o app (`streamlit run app.py`) |

Exemplos:

```bash
python setup.py --dry-run         # só mostra o que existe/falta
python setup.py --skip-bert       # setup sem pré-baixar os BERTs
python setup.py --force-vectorize # refaz a vetorização das tabelas
```

---

## Instalação manual (alternativa)

Se preferir fazer manualmente (equivale ao que o `setup.py` automatiza):

```bash
# 1. Ambiente virtual
python -m venv venv
venv\Scripts\activate            # Windows (cmd)
venv\Scripts\Activate.ps1        # Windows (PowerShell)
source venv/bin/activate         # Linux/macOS

# 2. Dependências
pip install -r requirements.txt

# 3. Fase offline — modelos Word2Vec
python etapa1/etapa1_script3_setup_biowordvec.py   # download (~13 GB)
python etapa1/etapa1_script4_compactar_biowordvec.py --skip-test  # compacta

# 4. Fase offline — vetores das tabelas (opcional se asset/ já existir)
python etapa1/etapa1_script1_vetorizar_tabelas.py
```

---

## Executando o aplicativo

Com o ambiente preparado, inicie a interface web (Streamlit):

```bash
streamlit run app.py
```

ou, usando o `setup.py`:

```bash
python setup.py --run
```

Acesse no navegador o endereço exibido no terminal (padrão: `http://localhost:8501`).

### Como usar a interface

1. Digite uma consulta em linguagem natural, ex.: `paciente com diabete` ou `pre-eclampsia`;
2. Ajuste o **Nº de termos para expansão** (N do AG) e o **Nº de tabelas para analisar features**;
3. Clique em **Buscar Tabelas Relevantes**;
4. O sistema mostra:
   - Os **termos candidatos** expandidos pelo W2V/BERT;
   - Os **pesos otimizados pelo GA** para cada termo;
   - O **ranking das tabelas** mais relevantes;
   - Para cada tabela do Top-N: as **colunas selecionadas pelo GA de features**, com score, schema, row count e chave primária.

---

## Fase offline (Etapa 1) em detalhe

A vetorização das tabelas (`etapa1_script1_vetorizar_tabelas.py`) transforma o nome e as colunas de cada tabela em um "documento" de palavras-chave (removendo stop-words e prefixos sem semântica como `co_`, `st_`, `dt_`), e gera um embedding **mean pooling** de 768 dimensões com o BERT português. O resultado é salvo em `asset/v_tabelas.npy` + `asset/tabelas_index.json`.

Os modelos Word2Vec:

- `etapa1_script3_setup_biowordvec.py` baixa o binário original da NCBI (`https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin`, ~13 GB). **Já baixado → pula automaticamente.**
- `etapa1_script4_compactar_biowordvec.py` carrega apenas as 500 mil palavras mais frequentes e salva em `modelos/biowordvec_500k.kv` (formato `.kv` do gensim, ~400 MB). Use `--skip-test` para pular o teste interativo em automação.

---

## Executando os testes

Os testes ficam no diretório `test/` e cobrem tradução, GA (timeout/watchdog), metadados, busca semântica, pipeline de features e o próprio `setup.py`.

```bash
# Suíte completa (da raiz do projeto)
python -m unittest discover -s . -p "test_*.py"

# Testes rápidos (sem carregar os modelos pesados)
cd test && python -m unittest test_tradutor test_ga_feature_selector test_metadata_loader test_setup
```

> Os testes de pipeline (`test_feature_selection_pipeline.py`, `test_busca_semantica.py`) carregam os modelos reais (BERT + BioWordVec) e podem levar alguns minutos.

---

## Estrutura do projeto

```
query-cnat/
├── app.py                              # Interface Streamlit (ponto de entrada)
├── config.py                           # Configurações centrais (caminhos, modelos, GA)
├── setup.py                            # Setup offline completo (venv, modelos, vetores)
├── tradutor.py                         # Tradução PT↔EN com cache, retry e fallback
├── requirements.txt                    # Dependências Python
├── README.md                           # Este documento
├── asset/                              # Ativos de entrada/saída (versionados)
│   ├── metadata_advanced_consolidated.json
│   ├── v_tabelas.npy
│   └── tabelas_index.json
├── etapa1/                             # FASE OFFLINE — preparação
│   ├── etapa1_script1_vetorizar_tabelas.py
│   ├── etapa1_script3_setup_biowordvec.py
│   ├── etapa1_script4_compactar_biowordvec.py
│   └── bert_model_compat.py
├── etapa2/                             # Busca semântica
│   ├── busca_semantica.py              #   BuscadorSemantico (expansão + vetorização)
│   └── metadata_loader.py              #   MetadataLoader (detalhes das tabelas)
├── etapa3/                             # Otimização por GA
│   ├── otimizador_ga.py                #   GA contínuo (pesos dos termos)
│   └── otimizador_ga_features.py       #   GA binário (seleção de colunas)
└── test/                               # Testes automatizados
    ├── test_tradutor.py
    ├── test_ga_feature_selector.py
    ├── test_feature_selection_pipeline.py
    ├── test_busca_semantica.py
    ├── test_metadata_loader.py
    └── test_setup.py
```

---

## Configuração (config.py)

| Chave | Valor padrão | Descrição |
|---|---|---|
| `METADATA_ADVANCED_FILE_PATH` | `asset/metadata_advanced_consolidated.json` | Metadados de entrada |
| `V_TABELAS_PATH` / `INDEX_PATH` | `asset/v_tabelas.npy` / `asset/tabelas_index.json` | Vetores e índice (saída da Etapa 1) |
| `BIOWORDVEC_MODEL_PATH` | `modelos/biowordvec_500k.kv` | Modelo W2V compactado |
| `ACTIVE_W2V_MODEL` | `biowordvec` | Modelo W2V ativo (`nilc` ou `biowordvec`) |
| `BERT_MODEL_NAME` | `pucpr/biobertpt-all` | BERT português biomédico |
| `BERT_EN_MODEL_NAME` | `dmis-lab/biobert-base-cased-v1.1` | BERT inglês biomédico |
| `GA_PARAMS` | gerações=50, população=100, mutação=0.1, elitismo=0.01, crossover=0.5, pais=0.3, `single_point` | Parâmetros dos GAs |
| `K_TOP_FITNESS` | 20 | K tabelas usadas no cálculo do fitness |

> Os GAs possuem **timeout de segurança** (300 s o GA de pesos, 120 s o GA de features) com watchdog intra-geração, evitando travamentos de horas. A persistência de pesos (`save_weights`) é opt-in.

---

## Solução de problemas (FAQ)

**"O setup.py diz que falta o venv e falha ao criar."**
Em Debian/Ubuntu, o Python do sistema pode não ter o módulo `venv`: instale com `sudo apt install python3-venv`.

**"O download do BioWordVec é muito grande."**
São ~13 GB da NCBI — é esperado. O `setup.py` pula se o arquivo já existir. Confira o espaço em disco antes.

**"Erro de tradução / rede no app."**
O `tradutor.py` tem retry (3 tentativas) e fallback: se a tradução falhar, a consulta original é usada. A expansão semântica funciona melhor com internet.

**"O app está lento na primeira consulta."**
A primeira execução carrega os modelos BERT e o BioWordVec na memória (pode levar alguns minutos). Depois, ficam em cache.

**"Quero refazer os vetores das tabelas."**
`python setup.py --force-vectorize` (ou rode `python etapa1/etapa1_script1_vetorizar_tabelas.py`).

**"Onde ficam os pesos otimizados pelo GA?"**
Por padrão **não são persistidos** (evita acúmulo de arquivos). Para persistir, use `save_weights=True` em `rodar_otimizacao_ga()` — os arquivos vão para `modelos/ga_pesos/`.

---

## Notas de produção

- **GPU opcional**: o código detecta CUDA automaticamente; o `pip install torch` padrão instala a versão CPU. Para GPU, instale a build CUDA do PyTorch.
- **Cache HuggingFace**: os modelos BERT ficam em `~/.cache/huggingface`. Em servidores, defina `HF_HOME` para um caminho persistente compartilhado.
- **Internet em runtime**: a tradução PT→EN usa Google Translate; sem internet, o sistema degrada com fallback para o texto original (a qualidade da expansão reduz).
- **Branch de desenvolvimento**: o trabalho de reprodutibilidade (timeouts, determinismo, testes) está na branch `reprodutibilidade`.
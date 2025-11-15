# Query CNAT

## Primeiros passos

### Pré-requisitos

Certifique-se de ter o Python 3.8 ou uma versão mais recente instalada.

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/denilsonbomjesus/query-cnat.git
    cd query-cnat
    ```

2.  **Criar um ambiente virtual (recomendado):**
    ```bash
    # Para Windows
    python -m venv venv
    venv\Scripts\activate

    # Para macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências necessárias:**
    ```bash
    pip install -r requirements.txt
    ```

### Executando o aplicativo

Para iniciar o sistema de busca semântica para datasets de saúde, execute o arquivo `app.py`:

```bash
streamlit run app.py
```
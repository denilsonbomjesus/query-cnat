# 🔍 ANÁLISE DE BUGS & BACKPORT — SISTEMA QUERY-CNAT vs. BATERIA DE VALIDAÇÃO SBTI

**Data:** 04/09/2026
**Escopo analisado:** `app.py`, `config.py`, `tradutor.py`, `etapa1/`, `etapa2/`, `etapa3/` (sistema principal) × `validation_sbti/` (B1–B8, utils_sbti, logs, results).
**Objetivo:** identificar tudo o que foi descoberto/corrigido **durante a execução dos testes de N=20** que ainda **não foi repassado ao sistema principal** (ou que foi corrigido apenas no harness de teste), além de inconsistências latentes entre código principal e código de validação.

> ✅ **STATUS (atualizado em 04/09/2026, tarde):** os itens **A, B (opção 2), C, E, I (limpeza de código) e J** foram **aplicados ao sistema principal** (`etapa3/otimizador_ga.py`, `etapa3/otimizador_ga_features.py`, `tradutor.py`, `config.py`) com testes de regressão adicionados (`test_tradutor.py`, `etapa3/test_ga_feature_selector.py`). Os itens **D, F e G** permanecem **restritos ao harness/docs de teste** (`validation_sbti/`) — **NÃO** devem ser aplicados ao sistema principal, conforme diretriz do autor. Ver seções individuais.

---

## 0. RESUMO EXECUTIVO

| # | Achado | Severidade | Onde existe hoje | Backport? | Status 04/09/2026 |
|---|---|---|---|---|---|
| A | **GA Binário (Fase 2) sem timeout e com `parallel_processing=n_cpus`** → stall de ~33 min na seed 789 (B1 N=20) | 🔴 CRÍTICA | `etapa3/otimizador_ga_features.py` (sistema principal, intacto) | ✅ **SIM — URGENTE** | ✅ **APLICADO** |
| B | Timeout do GA Contínuo só é checado **entre gerações**; geração única pode travar por horas (gen 33→34 = 10.041 s ≈ 2,8 h observado no B3 N=20) | 🟠 ALTA | `etapa3/otimizador_ga.py` (fix parcial aplicado) | ✅ **SIM** (endurecer watchdog) | ✅ **APLICADO (opção 2)** |
| C | `tradutor.py`: **classe duplicada** + decorator `lru_cache` num stub morto; a classe viva **não tem cache, timeout nem retry** → 56 erros de tradução no B3 N=20, não-determinismo entre seeds | 🟠 ALTA | `tradutor.py` (sistema principal) | ✅ **SIM** | ✅ **APLICADO** |
| D | **Regressão no harness B3**: troca de `import config` → `import config_sbti` **inutilizou a varredura N=20** (override nunca chega ao GA). B3 N=10 (código original) é válido; **B3 N=20 é inválido como análise de sensibilidade** | 🔴 CRÍTICA (resultados) | `validation_sbti/B3_sensibilidade/run_sensitivity_sweep.py` | ⚠️ Corrigir + **reexecutar B3 N=20** ou remover claims | ⚠️ **HARNESS — não aplicar ao principal** |
| E | `config.py` declara `crossover_type: 'uniform'` e `max_iteration_without_improv: 5`, mas o código usa **`single_point`** e **nunca implementa parada antecipada** | 🟡 MÉDIA | `config.py` × `etapa3/otimizador_ga.py`, `etapa3/otimizador_ga_features.py` | ✅ SIM (alinhar ou documentar) | ✅ **APLICADO** (docs do artigo pendente) |
| F | Ground truth "35 tabelas" **desatualizado** em `RESULTADOS_EXPANSAO_HARDWARE.md` (correto é **175**) | 🟡 MÉDIA (docs) | `validation_sbti/RESULTADOS_EXPANSAO_HARDWARE.md` | ⚠️ Docs | ⚠️ **HARNESS/docs — não aplicar ao principal** |
| G | Validação mede Fase 2 sobre **top-20** tabelas; a UI analisa apenas **top-5 (default do slider)** → métrica P@5_F2 não corresponde ao comportamento padrão da interface | 🟡 MÉDIA | `app.py` (slider `top_n_para_analise=5`) × `utils_sbti.py`/B2 (20) | ⚠️ Documentar/alinhar | ⚠️ **Não é bug de código** — decisão de relato no artigo |
| H | **Seeds não garantem reprodutibilidade total**: a tradução (rede) não é controlada por seed — mesma seed deu P@5 0,8 e 1,0 em execuções diferentes do B3 N=20 | 🟠 ALTA (artigo) | todo o pipeline (via `tradutor`) | ✅ SIM (cache de tradução) | ✅ **Parcialmente mitigado** (cache do item C) |
| I | `etapa3/otimizador_ga.py` já recebeu timeout + `parallel_processing=1`, **mas a alteração está sem commit** e o `n_cpus` ficou morto; `app.py` não passa `max_time_seconds` (usa default 300 s) | 🟡 MÉDIA | `etapa3/otimizador_ga.py`, `app.py` | ✅ SIM (commit + limpeza) | ✅ **Limpeza aplicada; commit pendente** |
| J | GA contínuo **persiste `pesos_*.npy` a cada execução** (inclusive na UI, a cada busca) → acúmulo de arquivos | 🟢 BAIXA | `etapa3/otimizador_ga.py` | Opcional | ✅ **APLICADO** (`save_weights=False` default) |

**Resposta direta à pergunta principal:** o problema de "N=10 fixo que ignora a seleção da interface" **NÃO existe no caminho principal** — o slider da UI (`app.py`) propaga `n_termos` corretamente até `expandir_consulta(query, n=n_termos)` e o GA redimensiona o cromossomo para `n`. O que **existe de fato** são os itens A–J acima, com destaque para **A** (bug real do sistema principal exposto pelos testes) e **D** (regressão que invalida o B3 N=20).

---

## 1. PERGUNTA CENTRAL: N=10 HARDCODADO QUE IGNORA A INTERFACE? — **NÃO EXISTE**

### 1.1 Fluxo da interface → pipeline (verificado em `app.py`)

```python
# app.py
n_termos = st.slider("Nº de termos para expansão (N do AG):", min_value=3, max_value=20, value=10)  # L182
...
termos, pesos, ranking = rodar_pipeline_busca(buscador, query_usuario, n_termos)                     # L194

# app.py → rodar_pipeline_busca
def rodar_pipeline_busca(buscador, query_usuario, n_termos=10):        # L136 (default é só fallback)
    termos_candidatos_tuplas = buscador.expandir_consulta(query_usuario, n=n_termos)   # L141  ✅ n propagado
```

- O valor do slider **é propagado** para `expandir_consulta(n=...)`, que retorna `n` termos (consulta + `n-1` expandidos).
- `vetorizar_termos_candidatos(termos)` e o GA contínuo operam com **dimensionalidade = n** (`GAOptimizer.n_dim = len(v_candidatos)`), ou seja, com 20 termos o GA otimiza 20 pesos. **Não há 10 fixo no ranking.**
- N=10 existe apenas como **default** (`n_termos=10` na assinatura, `value=10` no slider) e como default da assinatura `expandir_consulta(query, n=10)` em `etapa2/busca_semantica.py` — nunca como valor ignorante da UI.

### 1.2 Onde o "10" aparece de verdade (e por que não é bug do sistema)

| Arquivo | Ocorrência | Natureza |
|---|---|---|
| `etapa2/etapa2_teste_busca_simples.py` (L29) | `N_TERMOS = 10` | Script de demonstração, não usado pela UI |
| `etapa3/etapa3_teste_completo.py` (L38) | `N_TERMOS = 10` | Script de demonstração |
| `validation/*.py` (vários) | `expandir_consulta(query, n=10)` | Validação legada (pré-SBTI) |
| `config_sbti.py` | `N_EXPANSION_TERMS = 10` | Default de configuração, sobrescrito por env `N_EXPANSION` |
| `etapa2/busca_semantica.py` L257 | `def expandir_consulta(self, query, n=10, ...)` | Default da API, sempre sobrescrito pelos chamadores reais |

**Conclusão:** o sistema principal respeita a escolha de N da interface. Este ponto pode ser documentado como "não-bug" (resposta a possível crítica de revisor).

---

## 2. BUG A (CRÍTICO — PRECISA BACKPORT): GA BINÁRIO (FASE 2) SEM TIMEOUT E COM PARALELISMO

### 2.1 Evidência nos testes

- B1 N=20, seed **789**: `fase1_s = 49,8 s` (normal) mas `fase2_s = 1.971,7 s` (~33 min) — **stall na Fase 2** (seleção de colunas), não no GA contínuo.
- O GA contínuo recebeu o fix (`parallel_processing=1` + timeout de 120–150 s), mas **a Fase 2 usa `rodar_ga_feature_selection` sem nenhum timeout** e com `parallel_processing=self.n_cpus` (até 4 núcleos).

### 2.2 Código do sistema principal (intacto, NÃO foi corrigido)

```python
# etapa3/otimizador_ga_features.py
self.n_cpus = min(os.cpu_count() or 1, 4)          # L63
...
parallel_processing=self.n_cpus                    # L144  ← sem fix
...
def run(self):                                      # sem max_time_seconds
    ...
    ga_instance.run()                               # pode travar por ~30 min+ sem interrupção
```

- `rodar_ga_feature_selection(...)` é chamado pela **UI** (`app.py` → `rodar_pipeline_features`, uma vez por tabela do top-N) e pelos testes (`utils_sbti.run_pipeline_single`, B2).
- **Impacto em produção:** ao buscar "dislipidemia", a interface analisa as top-5/20 tabelas chamando este GA **sem proteção** → risco real de a UI travar ~30 min ou mais (exatamente o que ocorreu na seed 789) **sem nenhum timeout**.

### 2.3 Correção recomendada (espelhar o fix do GA contínuo)

```python
# etapa3/otimizador_ga_features.py
def run(self, max_time_seconds: int = 120):
    self._start_time = time.time()
    def on_generation_timeout(ga_instance):
        self.generation_count += 1
        if time.time() - self._start_time > max_time_seconds:
            logging.warning("[GA Features] TIMEOUT atingido. Parando.")
            return "stop"
    ga_instance = pygad.GA(..., on_generation=on_generation_timeout,
                           parallel_processing=1)   # desliga paralelismo p/ evitar stall
```

**Status:** ✅ **APLICADO ao sistema principal em 04/09/2026.** `etapa3/otimizador_ga_features.py` agora tem `run(max_time_seconds=120)` com `on_generation_timeout` (para entre gerações) e watchdog intra-geração na fitness (retorna fitness degradado ao estourar); `parallel_processing=1`; `rodar_ga_feature_selection(..., max_time_seconds=120)` propaga o valor. `app.py` e `utils_sbti.py`/B2 herdam o default de 120 s sem alteração (o chamador não precisa passar). Ver `etapa3/test_ga_feature_selector.py`.

---

## 3. BUG B (ALTA): TIMEOUT DO GA SÓ ENTRE GERAÇÕES — GERAÇÃO ÚNICA PODE TRAVAR HORAS

### 3.1 Evidência (logs do B3 N=20)

```
mutation_probability=0.2, run 2/3:
  [GA] Geração 33/50 — elapsed=32.5s
  [GA] Geração 34/50 — elapsed=10041.6s   ← 2,8 HORAS em UMA única geração
```

- O callback `on_generation_timeout` só roda **no fim de cada geração** (padrão PyGAD). Se uma geração inteira (avaliação de fitness da população) travar — ex. contenda de threads BioBERT/BERT — **o timeout nunca dispara**.
- Ocorrências no B3 N=20: stall de ~49 min (population_size=150, run 2 = 2.976 s) e ~2,8 h (mutation 0.2, run 2 = 10.041 s). A mensagem "TIMEOUT atingido" (2 ocorrências) só apareceu **depois** da geração travada terminar.
- O sistema principal (UI) herda o mesmo risco: o GA contínuo foi corrigido para parar entre gerações, mas uma geração que travar congela a UI até o PyGAD retornar.

### 3.2 Correção recomendada (produção)

- Opção 1 (robusta): executar cada GA num **subprocesso** com `timeout` de SO e matar o processo filho se estourar (ex.: `multiprocessing` + `terminate()` ou `subprocess.run(timeout=...)`).
- Opção 2 (simples): checar `elapsed > max_time` **dentro da fitness function** e lançar exceção/retornar fitness degradado quando estourar — reduz a janela de stall para o tempo de 1 avaliação.
- Registrar no artigo: "timeout de segurança por geração, com watchdog de processo" — evita claim de reprodutibilidade frágil.

**Status:** ✅ **APLICADO ao sistema principal em 04/09/2026 (opção 2).** `etapa3/otimizador_ga.py` e `etapa3/otimizador_ga_features.py` agora checam o tempo **dentro da fitness function** e devolvem fitness degradado (`-1e9`) ao estourar, reduzindo a janela de stall para o custo de 1 avaliação; a melhor solução rastreada é usada como fallback. (Opção 1 — subprocesso com timeout de SO — segue recomendada como evolução futura, não aplicada por ser mudança arquitetural.)

---

## 4. BUG C (ALTA): `tradutor.py` — CLASSE DUPLICADA, CACHE MORTO, SEM TIMEOUT/RETRY

### 4.1 Evidência (leitura do arquivo)

```python
# tradutor.py — há DUAS definições da mesma classe
class TradutorPTEN:                     # ← 1ª definição (stub com `...`)
    ...
    @lru_cache(maxsize=2048)
    def pt_para_en(self, texto): ...

    @lru_cache(maxsize=2048)
    def en_para_pt(self, texto): ...

class TradutorPTEN:                     # ← 2ª definição SOBRESCREVE a 1ª (vence)
    def __init__(self): ...
    def pt_para_en(self, texto): ...    # SEM @lru_cache
    def en_para_pt(self, texto): ...    # SEM @lru_cache
```

- **A classe viva não tem cache**: a cada chamada de `expandir_consulta()` (e são ~1 PT→EN + 20 EN→PT por execução com N=20), o sistema faz chamadas de rede ao Google Translate **sem cache**, **sem timeout** e **sem retry**.
- Efeitos observados nos testes: **56 erros de tradução** nos logs do B3 N=20 (`logs/b3_20termos_*.log`), tempos de expansão variando de ~10 s a ~70 s, e **resultados não-determinísticos** entre execuções com a mesma seed (traduções que falham retornam o termo original, alterando a lista de termos expandidos).
- No **sistema principal**, toda busca do usuário dispara N+1 traduções web sem proteção → latência, fragilidade e não-reprodutibilidade.

### 4.2 Correção recomendada

1. Remover a 1ª definição morta (ou a 2ª, mantendo UMA classe).
2. Aplicar `@lru_cache(maxsize=2048)` **na classe viva** (em `pt_para_en` e `en_para_pt`) — reduz drasticamente chamadas de rede e estabiliza runtime.
3. Adicionar timeout explícito e retry (ex.: parâmetro `timeout` do `GoogleTranslator`/requests) com fallback controlado.
4. Considerar cache **persistente em disco** das traduções termo→termo para reprodutibilidade total entre sessões (seed não controla rede!).

**Status:** ✅ **APLICADO ao sistema principal em 04/09/2026.** `tradutor.py` reescrito: 1 classe única (stub morto removido), cache em memória apenas de traduções bem-sucedidas, retry com backoff (`MAX_RETRIES=3`) e fallback controlado para o texto original. Testes: `test_tradutor.py` (6 casos, sem rede — com mocks). Cache persistente em disco continua como evolução opcional.

---

## 5. BUG D (CRÍTICO — RESULTADOS): REGRESSÃO NO SWEEP DO B3 — B3 N=20 É INVÁLIDO COMO SENSIBILIDADE

### 5.1 O que aconteceu

- Código original do B3: `import config as _config` → `_config.GA_PARAMS.update(overrides)`. Esse `config` **é o `config.py` da raiz** (que o `etapa3/otimizador_ga.py` importa como `config` e lê em `GAOptimizer.__init__` via `getattr(config, "GA_PARAMS")`). → **B3 N=10 (rodado em 28/08 com esse código) funciona** e mostra variação real (0,27–0,60).
- Durante a preparação do N=20, o import foi "corrigido" para `import config_sbti as _config` (diagnosticado erroneamente como "módulo `config` não existe" — **ele existe, na raiz**).
- `config_sbti.GA_PARAMS` é um **dict diferente** de `config.GA_PARAMS` (verificado: `SAME OBJECT? False`; ids distintos). O override passou a escrever num dict que **o GA nunca lê** → **varredura N=20 ficou INERTE**.

### 5.2 Evidência conclusiva

| Evidência | Detalhe |
|---|---|
| **Denominador das gerações** | 100% das **3.420** linhas `Geração X/50` nos logs N=20 têm `/50` — ou seja, `max_num_iteration` foi **sempre 50**, mesmo quando o sweep pedia 10, 20, 30 ou 80. |
| **Constância suspeita nos resultados N=20** | `max_num_iteration`: 0,9333 para **todos** os valores (10/20/30/50/80); `crossover_probability`: 0,9333 para todos (0.3–0.9). Se o sweep funcionasse, haveria variação (como no N=10: 0,6/0,267/0,4/0,467/0,4). |
| **N=10 (código original) mostra variação real** | `max_num_iteration`: 0,6 / 0,267 / 0,4 / 0,467 / 0,4; `population_size`: 0,533 em 25 — coerente com varredura funcional. |
| **Runtime** | N=10 escala com iterações (13 s p/ 10 gens → 101 s p/ 80); N=20 fica ~constante (~60 s) — GA rodou sempre 50 gens. |
| **Teste de import** | `otimizador_ga` importa `config.py` da raiz (`__file__ = .../config.py`), **não** `config_sbti`. |

### 5.3 Consequência

- **Tudo que foi escrito sobre "B3 N=20: sensibilidade" está comprometido.** As leituras de "P@5 plano ~0,93/1,0 para todos os hiperparâmetros" NÃO são robustez a hiperparâmetros — são o comportamento do **GA default repetido** com seeds/ruído.
- Claims tipo "N=20 supera N=10 em 23/23 configurações" (presentes em `ANALISE_RESULTADOS_N20.md` PARTE 5 e no rascunho do artigo) **não são sustentáveis** pela varredura N=20 atual.
- O que **permanece válido**: B1 N=20 (10 seeds), B8 N=20, B2 N=20 e o próprio B3 N=10 (28/08) — desde que citados pelo que realmente são.

### 5.4 Correção recomendada (harness)

1. Reverter o mecanismo para o override que **o GA realmente lê**, com restauração segura:
   ```python
   # run_sensitivity_sweep.py — run_single_config
   import config as _config            # módulo raiz (mesmo que otimizador_ga lê)
   original = copy.deepcopy(_config.GA_PARAMS)
   _config.GA_PARAMS.update(ga_overrides)
   try:
       ...rodar_otimizacao_ga(...)
   finally:
       _config.GA_PARAMS.update(original)
   ```
2. **Melhor ainda** (recomendado para o futuro): passar overrides **explicitamente** ao GA, evitando mutação de módulo global:
   ```python
   def rodar_otimizacao_ga(..., ga_params: dict | None = None):
       optimizer = GAOptimizer(..., ga_params=ga_params)
   ```
3. Adicionar **log dos parâmetros efetivos** no início de cada run do GA (`sol_per_pop`, `num_generations`, `crossover_probability`, `mutation_probability`, `elit_ratio`) para auditar a varredura.
4. **Reexecutar o B3 N=20** (~2–3 h com o script resiliente, agora com override correto) OU remover os claims de B3 N=20 do artigo e reportar apenas B3 N=10 + B1/B8 N=20.

**Status:** ❌ **Regressão ativa no HARNESS de teste** (`validation_sbti`) — **NÃO aplicar ao sistema principal** (a diretriz do autor é corrigir apenas o sistema principal). O código principal `otimizador_ga.py` lê `config.py` da raiz, que segue íntegro. A correção do harness (reverter para override no módulo raiz ou passar `ga_params` explícito) e a reexecução/remoção dos claims do B3 N=20 ficam **pendentes no escopo `validation_sbti`**.

---

## 6. BUG E (MÉDIA): `config.py` DIVERGE DO CÓDIGO (CROSSOVER E PARADA ANTECIPADA)

```python
# config.py (GA_PARAMS)
'crossover_type': 'uniform',              # ← código usa "single_point"
'max_iteration_without_improv': 5         # ← NUNCA é usado (não há early stop)
```

- `etapa3/otimizador_ga.py` L205: `crossover_type="single_point"` (hardcoded).
- `etapa3/otimizador_ga_features.py` L140: `crossover_type="single_point"` (hardcoded).
- `max_iteration_without_improv` não aparece em nenhum módulo (grep confirma: só em `config.py`). Não existe lógica de parada antecipada.
- **Impacto:** o artigo/README descrevem o GA conforme `config.py` ("uniform", "parada antecipada") → descrição técnica incorreta; análise de sensibilidade de `crossover_probability` no B3 foi feita sobre um GA `single_point`, não `uniform`.

### Correção
- **Aplicado:** `config.py` agora declara `'crossover_type': 'single_point'` (o que o código usa) e a chave morta `max_iteration_without_improv` foi **removida** com nota explicativa. Ambos os otimizadores leem `ga_conf.get("crossover_type", "single_point")`.
- **Pendente (docs/artigo):** atualizar textos do artigo/README que descrevem o GA como "uniform" ou com "parada antecipada" — o comportamento real é `single_point` sem early stop.

---

## 7. BUG F (DOCS): GROUND TRUTH "35 TABELAS" DESATUALIZADO

- Correto: **175 tabelas relevantes** de 1.100 (`validation/cholesterol_y_true.json`), conforme corrigido e usado nos testes N=10/N=20.
- `validation_sbti/RESULTADOS_EXPANSAO_HARDWARE.md` (L6) ainda diz **"35 tabelas relevantes de 1.100"** → atualizar (ou marcar como legado).
- Conferir o artigo/docx atual para não citar 35.

**Status:** ⚠️ Docs do harness de teste — **NÃO aplicar ao sistema principal.** O "35" só existe em `validation_sbti/RESULTADOS_EXPANSAO_HARDWARE.md`; verificado: os artigos `.tex`/docs da raiz citam "1.100 tabelas" sem o número 35, e o código principal não referencia ground truth.

---

## 8. BUG G (MÉDIA — CONSISTÊNCIA MÉTRICA): FASE 2 MEDE TOP-20; UI ANALISA TOP-5

- `app.py`: `top_n_para_analise = st.slider(..., min_value=1, max_value=20, value=5)` → a Fase 2 da UI analisa **5 tabelas** por padrão.
- Validação (`utils_sbti.run_pipeline_single`, B2): `n_tables_f2 = min(20, len(ranking))` → P@5_F2/P@10_F2 são calculados sobre um re-ranking das **20** primeiras tabelas.
- **Consequência:** P@5_F2=1,0 (N=20) significa "entre as 20 tabelas analisadas, as 5 mais bem re-rankeadas são relevantes". Na UI com top-5, o comportamento pode diferir.
- **Recomendação:** (a) reportar no artigo que a Fase 2 foi avaliada sobre top-20; (b) opcionalmente alinhar slider default para 20 ou medir também top-5.

**Status:** ⚠️ **NÃO é bug do código principal** — é divergência de protocolo de medição entre a UI (top-5 default) e a validação (top-20). Nenhuma correção de código no sistema principal; apenas decisão de relato/experimento no artigo.

---

## 9. BUG H (ALTA — REPRODUTIBILIDADE): SEEDS NÃO CONTROLAM A TRADUÇÃO

- `set_seeds()` fixa `random`, `numpy`, `torch`, PyGAD — mas **não** a tradução (Google Translate via rede).
- Evidência: no B3 N=20, a **mesma seed** (run 2 = seed 43) produziu P@5 **0,8** em alguns grupos e **1,0** em outros (population_size 100/150, elit 0.05/0.2), com os mesmos parâmetros GA efetivos (sweep inerte) — a única fonte de variação é a tradução/estado de rede.
- **Impacto no artigo:** a afirmação "seeds fixas garantem reprodutibilidade" é verdadeira para o GA, mas **não para o pipeline completo**. Recomendações:
  1. Cache persistente de traduções (item C) — termo EN→PT idêntico sempre.
  2. Salvar nos resultados a **lista exata de termos expandidos** por seed (para auditoria).
  3. Reportar média±std sobre seeds (já feito), nunca um único run como "reprodutível".

**Status:** ✅ **Parcialmente mitigado no sistema principal em 04/09/2026** pelo cache de tradução do item C (aplicado em `tradutor.py`). Cache persistente em disco + log de termos expandidos por seed são evoluções opcionais/harness; a claim de reprodutibilidade no artigo deve usar média±std.

---

## 10. BUG I (MÉDIA): FIX DO GA CONTÍNUO SEM COMMIT + PARÂMETROS ÓRFÃOS

- `etapa3/otimizador_ga.py` já contém o fix (timeout `max_time_seconds=300` default + `parallel_processing=1`) — **mas está sem commit** (`git status`: `M etapa3/otimizador_ga.py`).
- `self.n_cpus` continua sendo calculado (L50) e nunca mais usado (ficou órfão com `parallel_processing=1`).
- `app.py` chama `rodar_otimizacao_ga(...)` **sem** `max_time_seconds` → usa default de 300 s; a UI pode ficar "pensando" até 5 min sem feedback de progresso intermediário.
- **Recomendação:** commitar o fix; remover `n_cpus`; expor `max_time_seconds` via `config.py` (ex.: `GA_TIMEOUT_SECONDS`) e usar na UI com mensagem de progresso; manter `parallel_processing` configurável por env (1 = estável, >1 = experimental).

**Status:** ✅ **Limpeza de código APLICADA em 04/09/2026** (`n_cpus` e método morto `on_generation` removidos, crossover lido do `config.py`, pesos `.npy` opt-in via `save_weights=False`). **Pendente: commitar** as alterações (incluindo as dos demais itens aplicados) e, opcionalmente, expor `max_time_seconds` com mensagem de progresso na UI.

---

## 11. BUG J (BAIXA): PERSISTÊNCIA DE PESOS A CADA EXECUÇÃO

```python
# etapa3/otimizador_ga.py (run())
np.save(os.path.join(self.persistence_dir, f"pesos_{int(time.time())}.npy"), best_sol)
```

- A cada busca da UI/execução de teste, um arquivo novo é gravado em `modelos/ga_pesos/` (ex.: dezenas criados durante B1–B8). Em produção com uso contínuo, cresce indefinidamente.
- **Recomendação:** salvar apenas sob demanda (flag `save_weights=False` default) ou manter rolling window; ou mover para diretório de logs.

**Status:** ✅ **APLICADO ao sistema principal em 04/09/2026.** `GAOptimizer.run(max_time_seconds=..., save_weights=False)` e `rodar_otimizacao_ga(..., save_weights=False)` — persistência agora é opt-in; o diretório só é criado ao salvar. Nenhum módulo lê os `pesos_*.npy`, então o default `False` não afeta o pipeline.

---

## 12. CONFIRMAÇÕES POSITIVAS (O QUE ESTÁ CERTO E NÃO PRECISA MUDAR)

| Item | Status |
|---|---|
| UI → N de termos → expansão → GA (dimensão = n) | ✅ Correto (sem N=10 fixo) |
| B1 N=20 (10 seeds) com dados salvos + top-5 por seed | ✅ Válido e íntegro |
| B8 N=20 (10 seeds) | ✅ Válido |
| B2 N=20 (5 runs, timings por fase) | ✅ Válido (após limpar runs bugados) |
| B3 N=10 (28/08, código original) | ✅ Válido como sensibilidade |
| Ground truth 175 tabelas | ✅ Usado corretamente nos testes (≠35) |
| Nomenclatura das fases (Expansão → Fase 1 Ranking → Fase 2 Seleção de Colunas) | ✅ Consistente nos docs de teste |
| Métricas calculadas (P@k, PR-AUC, Max F1) | ✅ Fórmulas corretas e alinhadas ao y_true (175) |

---

## 13. PLANO DE AÇÃO PRIORIZADO

| Prioridade | Ação | Arquivos | Esforço |
|---|---|---|---|
| P0 | **Corrigir B3 sweep** (override no módulo que o GA lê ou param explícito) e **reexecutar B3 N=20** (ou remover claims) — escopo HARNESS | `validation_sbti/B3_sensibilidade/run_sensitivity_sweep.py`; `ANALISE_RESULTADOS_N20.md`; artigo | ~2–3 h (run) + 1 h docs |
| P0 | ~~Backport do timeout + `parallel_processing=1` para o GA Binário (Fase 2)~~ | ~~`etapa3/otimizador_ga_features.py`~~ | ✅ **FEITO (04/09)** |
| P1 | ~~Corrigir `tradutor.py` (1 classe, cache, timeout/retry, fallback controlado)~~ | ~~`tradutor.py`~~ | ✅ **FEITO (04/09)** + `test_tradutor.py` |
| P1 | ~~Watchdog intra-geração (checagem na fitness)~~ | ~~`etapa3/otimizador_ga.py`, `etapa3/otimizador_ga_features.py`~~ | ✅ **FEITO (04/09)** — opção 2; subprocesso = evolução futura |
| P2 | ~~Alinhar `config.py` × código; remover `n_cpus` órfão~~ + **commit** do fix do GA contínuo | ~~`config.py`, `etapa3/otimizador_ga.py`~~ | ✅ **Código FEITO (04/09)**; ⏳ **commit pendente** |
| P2 | Atualizar `RESULTADOS_EXPANSAO_HARDWARE.md` (35→175) e revisar menções no artigo — escopo HARNESS/docs | docs | 30 min |
| P3 | Documentar top-20 (Fase 2) vs top-5 (UI); cache de tradução persistente em disco | docs + código | 1–2 h |

---

## 14. COMO ESTE DOCUMENTO FOI CONSTRUÍDO (EVIDÊNCIAS)

1. Leitura integral de `app.py`, `config.py`, `tradutor.py`, `etapa2/busca_semantica.py`, `etapa2/metadata_loader.py`, `etapa3/otimizador_ga.py`, `etapa3/otimizador_ga_features.py`, `validation_sbti/config_sbti.py`, `utils_sbti.py` e scripts B1/B2/B3/B8.
2. Comparação de ids de `config_sbti.GA_PARAMS` vs `config.GA_PARAMS` (objetos distintos → override inerte).
3. Parse dos logs do B3 N=20 (3.420 linhas `Geração X/50`; 69 runs; 2 stalls de geração; 56 erros de tradução; 2 timeouts entre gerações).
4. Dados de `results/B1/all_seeds_20termos.json` (seed 789: fase2 1.971 s), `results/B3/sensitivity_raw.json` (N=10 válido) e `sensitivity_raw_20termos.json` (N=20 inerte).
5. `git diff`/`git status` para separar o que já foi alterado no sistema principal (`etapa3/otimizador_ga.py` — sem commit) do que só existe no harness.

> ⚠️ **Nota de honestidade científica:** o achado D (regressão do B3) foi causado por uma correção aplicada durante a preparação dos testes N=20, que partiu de um diagnóstico incorreto ("módulo `config` não existe"). Este documento registra o erro e a correção, para que as conclusões do artigo reflitam apenas experimentos válidos.

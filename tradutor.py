# tradutor.py
"""Tradução PT <-> EN com cadeia de provedores (deep_translator).

Provedores (em ordem de preferência):
  1. GoogleTranslator — qualidade padrão;
  2. MyMemoryTranslator — fallback gratuito sem chave, usado quando o Google
     falha (bloqueio de rede / captcha / rate-limit / mudança do endpoint).

Motivo do fallback: o erro "No translation was found using the current
translator" observado em produção é o GoogleTranslator do deep_translator
devolvendo ``TranslationNotFound`` — normalmente porque o endpoint público do
Google Translate respondeu sem tradução. Antes, as 3 tentativas esgotadas
caíam direto no texto original; agora o MyMemory é tentado antes do fallback
final, reduzindo a degradação silenciosa da expansão de consulta.

Bug C (backport): existiam DUAS definições de ``TradutorPTEN`` — um stub morto
com ``@lru_cache`` (sobrescrito pela segunda classe) e a classe viva SEM cache,
sem timeout e sem retry. Isso gerava chamadas de rede repetidas a cada expansão
de consulta, erros intermitentes de tradução e não-determinismo entre execuções.

Correção: uma única classe, com cache em memória dos métodos vivos, retry com
backoff, cadeia de provedores e fallback controlado (retorna o texto original
somente após esgotar todos os provedores, sem propagar exceção para o pipeline).

Nota de design: apenas traduções BEM-SUCEDIDAS entram no cache. Se uma falha
temporária de rede (ou o fallback) fosse cacheada, o termo ficaria "preso" no
texto original pelo resto do processo mesmo depois da rede voltar.
"""
from deep_translator import GoogleTranslator, MyMemoryTranslator
from deep_translator.exceptions import TranslationNotFound
import logging
import time

logger = logging.getLogger(__name__)

# Nº máximo de tentativas por chamada de rede antes do fallback controlado.
MAX_RETRIES = 3
# Backoff base (segundos) entre tentativas: tentativa n dorme BASE * n.
RETRY_BACKOFF_SECONDS = 1.0


class TradutorPTEN:
    """Traduz automaticamente entre português e inglês (bidirecional)."""

    def __init__(self, max_retries: int = MAX_RETRIES):
        # Provedor primário (mantido como atributo público por compatibilidade
        # com callers/tests que o referenciam diretamente).
        self.pt2en = GoogleTranslator(source='pt', target='en')
        self.en2pt = GoogleTranslator(source='en', target='pt')
        # Provedores de fallback (gratuitos, sem chave de API).
        # Atenção: o MyMemoryTranslator do deep_translator NÃO aceita códigos
        # ISO ('pt'/'en') — exige nomes de língua ('portuguese brazil' etc.).
        self._fallback_pt_en = MyMemoryTranslator(
            source='portuguese brazil', target='english')
        self._fallback_en_pt = MyMemoryTranslator(
            source='english', target='portuguese brazil')
        self.max_retries = max(1, int(max_retries))

        # Cache apenas de traduções bem-sucedidas (termo original -> tradução).
        # Um dict simples é intencional: evita cachear o fallback de falhas.
        self._cache_pt_en = {}
        self._cache_en_pt = {}

    def _traduzir_com_retry(self, provedor_principal, provedor_fallback, texto, cache):
        """Traduz com cache + retry + cadeia de provedores e fallback controlado."""
        if not texto:
            return texto

        # Cache hit (somente traduções que já funcionaram)
        if texto in cache:
            return cache[texto]

        for provedor in (provedor_principal, provedor_fallback):
            nome = type(provedor).__name__
            for tentativa in range(1, self.max_retries + 1):
                try:
                    traduzido = provedor.translate(texto)
                    logger.debug(f"[Tradução OK] '{texto}' → '{traduzido}' ({nome})")
                    if provedor is provedor_fallback:
                        # Visibilidade: mostra quando o fallback (ex.: MyMemory)
                        # resolveu — ajuda a diagnosticar provedores quebrados.
                        logger.info(f"🔁 Tradução via fallback ({nome}): '{texto}' → '{traduzido}'")
                    cache[texto] = traduzido
                    return traduzido
                except TranslationNotFound:
                    # Falha "dura" do provedor (ex.: Google bloqueando a
                    # requisição / "No translation was found..."). Tentar de
                    # novo no MESMO provedor não ajuda — pula direto para o
                    # próximo da cadeia, sem gastar retries nem backoff.
                    logger.warning(
                        f"Provedor {nome} não encontrou tradução para '{texto}'; "
                        f"tentando próximo provedor."
                    )
                    break
                except Exception as e:
                    logger.warning(
                        f"Erro na tradução de '{texto}' via {nome} "
                        f"(tentativa {tentativa}/{self.max_retries}): {e}"
                    )
                    if tentativa < self.max_retries:
                        time.sleep(RETRY_BACKOFF_SECONDS * tentativa)

        logger.error(
            f"Falha na tradução de '{texto}' após 2 provedores x "
            f"{self.max_retries} tentativas. Usando texto original como fallback."
        )
        return texto  # Fallback controlado: nunca derruba o pipeline

    def pt_para_en(self, texto):
        return self._traduzir_com_retry(
            self.pt2en, self._fallback_pt_en, texto, self._cache_pt_en
        )

    def en_para_pt(self, texto):
        return self._traduzir_com_retry(
            self.en2pt, self._fallback_en_pt, texto, self._cache_en_pt
        )

    def traduz_lista(self, lista, direcao="en2pt"):
        traduzida = []
        for termo in lista:
            t = self.en_para_pt(termo) if direcao == "en2pt" else self.pt_para_en(termo)
            traduzida.append(t)
        return traduzida

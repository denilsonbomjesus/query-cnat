# tradutor.py
"""Tradução PT <-> EN usando Google Translate (deep_translator).

Bug C (backport): existiam DUAS definições de ``TradutorPTEN`` — um stub morto
com ``@lru_cache`` (sobrescrito pela segunda classe) e a classe viva SEM cache,
sem timeout e sem retry. Isso gerava chamadas de rede repetidas a cada expansão
de consulta, erros intermitentes de tradução e não-determinismo entre execuções.

Correção: uma única classe, com cache em memória dos métodos vivos, retry com
backoff e fallback controlado (retorna o texto original somente após esgotar as
tentativas, sem propagar exceção para o pipeline).

Nota de design: apenas traduções BEM-SUCEDIDAS entram no cache. Se uma falha
temporária de rede (ou o fallback) fosse cacheada, o termo ficaria "preso" no
texto original pelo resto do processo mesmo depois da rede voltar.
"""
from deep_translator import GoogleTranslator
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
        # Define os tradutores
        self.pt2en = GoogleTranslator(source='pt', target='en')
        self.en2pt = GoogleTranslator(source='en', target='pt')
        self.max_retries = max(1, int(max_retries))

        # Cache apenas de traduções bem-sucedidas (termo original -> tradução).
        # Um dict simples é intencional: evita cachear o fallback de falhas.
        self._cache_pt_en = {}
        self._cache_en_pt = {}

    def _traduzir_com_retry(self, translator, texto, cache):
        """Traduz com cache + retry e fallback controlado."""
        if not texto:
            return texto

        # Cache hit (somente traduções que já funcionaram)
        if texto in cache:
            return cache[texto]

        for tentativa in range(1, self.max_retries + 1):
            try:
                traduzido = translator.translate(texto)
                logger.debug(f"[Tradução OK] '{texto}' → '{traduzido}'")
                cache[texto] = traduzido
                return traduzido
            except Exception as e:
                logger.warning(
                    f"Erro na tradução de '{texto}' (tentativa {tentativa}/{self.max_retries}): {e}"
                )
                if tentativa < self.max_retries:
                    time.sleep(RETRY_BACKOFF_SECONDS * tentativa)

        logger.error(
            f"Falha na tradução de '{texto}' após {self.max_retries} tentativas. "
            f"Usando texto original como fallback."
        )
        return texto  # Fallback controlado: nunca derruba o pipeline

    def pt_para_en(self, texto):
        return self._traduzir_com_retry(self.pt2en, texto, self._cache_pt_en)

    def en_para_pt(self, texto):
        return self._traduzir_com_retry(self.en2pt, texto, self._cache_en_pt)

    def traduz_lista(self, lista, direcao="en2pt"):
        traduzida = []
        for termo in lista:
            t = self.en_para_pt(termo) if direcao == "en2pt" else self.pt_para_en(termo)
            traduzida.append(t)
        return traduzida

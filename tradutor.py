# tradutor.py
from deep_translator import GoogleTranslator
import logging

from functools import lru_cache

class TradutorPTEN:
    ...
    @lru_cache(maxsize=2048)
    def pt_para_en(self, texto): ...
    
    @lru_cache(maxsize=2048)
    def en_para_pt(self, texto): ...

class TradutorPTEN:
    """Traduz automaticamente entre português e inglês (bidirecional)."""

    def __init__(self):
        # Define os tradutores
        self.pt2en = GoogleTranslator(source='pt', target='en')
        self.en2pt = GoogleTranslator(source='en', target='pt')

    def pt_para_en(self, texto):
        try:
            if not texto:
                return texto
            traduzido = self.pt2en.translate(texto)
            logging.debug(f"[PT→EN] '{texto}' → '{traduzido}'")
            return traduzido
        except Exception as e:
            logging.error(f"Erro na tradução PT→EN: {e}")
            return texto

    def en_para_pt(self, texto):
        try:
            if not texto:
                return texto
            traduzido = self.en2pt.translate(texto)
            logging.debug(f"[EN→PT] '{texto}' → '{traduzido}'")
            return traduzido
        except Exception as e:
            logging.error(f"Erro na tradução EN→PT: {e}")
            return texto

    def traduz_lista(self, lista, direcao="en2pt"):
        traduzida = []
        for termo in lista:
            t = self.en_para_pt(termo) if direcao == "en2pt" else self.pt_para_en(termo)
            traduzida.append(t)
        return traduzida

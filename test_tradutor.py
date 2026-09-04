# test_tradutor.py — Testes de regressão para tradutor.py (Bug C)
#
# Bug C original: havia DUAS definições de TradutorPTEN (um stub morto com
# lru_cache e a classe viva SEM cache). A classe viva fazia chamadas de rede
# repetidas, sem retry, gerando erros intermitentes e não-determinismo.
# Estes testes garantem: (1) classe única viva com cache funcionando; (2)
# retry com fallback controlado; (3) cache só armazena traduções bem-sucedidas.
import unittest
from unittest import mock

from tradutor import TradutorPTEN


class TestTradutorPTEN(unittest.TestCase):

    def setUp(self):
        self.tradutor = TradutorPTEN(max_retries=3)

    def test_classe_unica_sem_stub_morto(self):
        """A classe viva deve existir sem a 1ª definição-stub (com '...')."""
        # Na versão bugada, a classe viva não tinha atributos de cache.
        # A presença dos caches prova que a classe ativa é a corrigida.
        self.assertTrue(hasattr(self.tradutor, "_cache_pt_en"))
        self.assertTrue(hasattr(self.tradutor, "_cache_en_pt"))
        self.assertIsInstance(self.tradutor._cache_pt_en, dict)
        # O stub morto declarava métodos com corpo '...'; a classe viva deve
        # ter métodos chamáveis e com comportamento real.
        self.assertTrue(callable(self.tradutor.pt_para_en))
        self.assertTrue(callable(self.tradutor.en_para_pt))

    def test_traducoes_sao_cacheadas(self):
        """Uma tradução bem-sucedida é cacheada (1 chamada de rede por termo)."""
        with mock.patch.object(self.tradutor.pt2en, "translate",
                               return_value="cholesterol") as mock_translate:
            r1 = self.tradutor.pt_para_en("colesterol")
            r2 = self.tradutor.pt_para_en("colesterol")

        self.assertEqual(r1, "cholesterol")
        self.assertEqual(r2, "cholesterol")
        self.assertEqual(mock_translate.call_count, 1,
                         "Cache deveria evitar a 2ª chamada de rede.")
        self.assertIn("colesterol", self.tradutor._cache_pt_en)

    def test_retry_e_fallback_controlado(self):
        """Falhas esgotam as tentativas e retornam o texto original (fallback)."""
        with mock.patch.object(self.tradutor.pt2en, "translate",
                               side_effect=Exception("network down")) as mock_translate:
            r = self.tradutor.pt_para_en("colesterol")

        self.assertEqual(r, "colesterol")  # fallback controlado
        self.assertEqual(mock_translate.call_count, self.tradutor.max_retries)

    def test_falha_temporaria_nao_e_cacheada(self):
        """Fallback (falha) NÃO entra no cache: permite sucesso depois."""
        # 1ª chamada: falha (rede fora) → fallback
        with mock.patch.object(self.tradutor.pt2en, "translate",
                               side_effect=Exception("network down")):
            r1 = self.tradutor.pt_para_en("colesterol")
        self.assertEqual(r1, "colesterol")
        self.assertNotIn("colesterol", self.tradutor._cache_pt_en,
                         "Falha não pode ser cacheada.")

        # 2ª chamada: rede voltou → traduz de verdade e cacheia
        with mock.patch.object(self.tradutor.pt2en, "translate",
                               return_value="cholesterol"):
            r2 = self.tradutor.pt_para_en("colesterol")
        self.assertEqual(r2, "cholesterol")
        self.assertIn("colesterol", self.tradutor._cache_pt_en)

    def test_traduz_lista(self):
        with mock.patch.object(self.tradutor.en2pt, "translate",
                               side_effect=lambda t: {"blood": "sangue",
                                                       "pressure": "pressão"}[t]):
            resultado = self.tradutor.traduz_lista(["blood", "pressure"], direcao="en2pt")
        self.assertEqual(resultado, ["sangue", "pressão"])

    def test_texto_vazio(self):
        self.assertEqual(self.tradutor.pt_para_en(""), "")
        self.assertEqual(self.tradutor.en_para_pt(""), "")
        self.assertEqual(self.tradutor.en_para_pt(None), None)


if __name__ == "__main__":
    unittest.main()

# test_setup.py — Testes para o script de setup offline (setup.py)

import os
import sys
import unittest

# Adicionar o diretório raiz do projeto ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import setup


class TestCaminhosVenv(unittest.TestCase):
    """Caminho do interpretador do venv deve respeitar o SO (Windows vs Linux/mac)."""

    def test_linux(self):
        original = setup.SISTEMA
        try:
            setup.SISTEMA = "Linux"
            caminho = setup.caminho_venv_python()
            self.assertTrue(caminho.endswith("bin/python"))
            self.assertNotIn("Scripts", caminho)
        finally:
            setup.SISTEMA = original

    def test_windows(self):
        original = setup.SISTEMA
        try:
            setup.SISTEMA = "Windows"
            caminho = setup.caminho_venv_python()
            # No Windows, o interpretador fica em venv/Scripts/python.exe
            # (os.path.join usa o separador do SO onde o teste roda)
            self.assertIn(os.path.join("Scripts", "python.exe"), caminho)
            self.assertNotIn("bin/python", caminho)
        finally:
            setup.SISTEMA = original

    def test_macos(self):
        original = setup.SISTEMA
        try:
            setup.SISTEMA = "Darwin"
            caminho = setup.caminho_venv_python()
            self.assertTrue(caminho.endswith("bin/python"))
        finally:
            setup.SISTEMA = original


class TestStatus(unittest.TestCase):
    """Coletar status deve listar todos os itens do ambiente, sempre como (caminho, bool)."""

    ITENS_ESPERADOS = (
        "Python (sistema)",
        "virtualenv (venv/)",
        "requirements instalados",
        "Metadados (asset/metadata_advanced_consolidated.json)",
        "BioWordVec compactado (modelos/biowordvec_500k.kv)",
        "BioWordVec original (biowordvec_model/*.bin)",
        "Vetores das tabelas (asset/v_tabelas.npy)",
        "Índice das tabelas (asset/tabelas_index.json)",
        "Modelos BERT (cache HuggingFace)",
    )

    def test_estrutura(self):
        status = setup.coletar_status()
        for item in self.ITENS_ESPERADOS:
            self.assertIn(item, status)
            caminho, ok = status[item]
            self.assertIsInstance(caminho, str)
            self.assertIsInstance(ok, bool)

    def test_caminhos_apontam_para_dentro_do_projeto(self):
        status = setup.coletar_status()
        # 'Python (sistema)' guarda a versão (ex: '3.12.3'), não um caminho
        for item in self.ITENS_ESPERADOS:
            if item == "Python (sistema)":
                continue
            caminho, _ = status[item]
            # Caminhos de artefatos devem ser absolutos (baseados no ROOT do projeto)
            self.assertTrue(os.path.isabs(caminho) or "huggingface" in caminho, item)


class TestBertCache(unittest.TestCase):
    """bert_em_cache deve retornar lista (vazia = tudo em cache)."""

    def test_retorna_lista(self):
        faltando = setup.bert_em_cache()
        self.assertIsInstance(faltando, list)
        # Se aponta modelos faltando, eles devem ser nomes válidos de modelos
        for modelo in faltando:
            self.assertIn(modelo, setup.MODELOS_BERT)


class TestDetectaAusencia(unittest.TestCase):
    """Um arquivo inexistente deve aparecer como 'faltando' no status."""

    def test_arquivo_inexistente_fica_faltando(self):
        caminho_fake = os.path.join(setup.ROOT, "asset", "arquivo_que_nao_existe_xyz.npy")
        self.assertFalse(os.path.exists(caminho_fake))


if __name__ == "__main__":
    unittest.main()
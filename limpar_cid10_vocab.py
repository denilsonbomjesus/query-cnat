# limpar_cid10_vocab.py
import re

input_file = "asset/cid10_vocab.txt"
output_file = "asset/cid10_vocab_limpo.txt"

with open(input_file, "r", encoding="utf-8") as f:
    linhas = f.readlines()

termos = []
for linha in linhas:
    linha = linha.strip()
    # Remove códigos como "A00", "B25.1", "Z43.7", etc.
    linha = re.sub(r"^[A-Z]\d{2}(\.\d+)?\s*", "", linha)
    # Remove quebras de linha soltas e espaços duplicados
    linha = re.sub(r"\s+", " ", linha)
    if len(linha) > 3:
        termos.append(linha)

# Remove duplicatas
termos = sorted(set(termos))

with open(output_file, "w", encoding="utf-8") as f:
    for termo in termos:
        f.write(termo + "\n")

print(f"✅ {len(termos)} termos médicos salvos em '{output_file}'")

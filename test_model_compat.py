# test_model_compat.py
from transformers import AutoTokenizer, AutoModel
model_id = "pucpr/biobertpt-all"
print("Testando:", model_id)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    print("Tokenizer e Model carregados OK.")
    # testa inferência curta
    sample = "Paciente com pressão arterial alta e dor de cabeça."
    inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    # checa atributo last_hidden_state
    if hasattr(outputs, "last_hidden_state"):
        cls = outputs.last_hidden_state[0,0,:]
        print("last_hidden_state existe. shape CLS:", tuple(cls.shape))
    else:
        print("ERRO: output nao tem last_hidden_state — modelo incompatível com pipeline atual.")
except Exception as e:
    print("Falha ao carregar / inferir:", e)

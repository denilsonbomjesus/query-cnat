# etapa2_teste_busca_simples.py

from busca_semantica import BuscadorSemantico
import logging
import time

def main():
    """
    Testa o fluxo completo da busca semântica (Etapas 1 e 3 do contexto)
    mas SEM a otimização do AG (Etapa 2 do contexto).
    
    Isso simula uma busca "base" usando pesos manuais.
    """
    logging.info("--- INICIANDO TESTE DA ETAPA 2 ---")
    
    # 1. Carregar o buscador (isso inicializa todos os modelos e vetores)
    # (Pode demorar alguns segundos, especialmente o W2V)
    try:
        start_load = time.time()
        buscador = BuscadorSemantico()
        logging.info(f"Buscador carregado em {time.time() - start_load:.2f}s")
    except Exception as e:
        logging.error(f"Falha ao carregar o BuscadorSemantico. Verifique os arquivos da Etapa 1. Erro: {e}")
        return

    # 2. Definir a consulta do usuário
    # query_usuario = "pre-eclampsia"
    query_usuario = "hypertension"
    N_TERMOS = 10 # Número de termos para expansão (o AG otimizará N pesos)

    # --- Início do pipeline (o que rodará em tempo real) ---
    start_search = time.time()

    # 3. (Processo 1.1) Expandir a consulta com W2V
    # Retorna: [("pré-eclâmpsia", 1.0), ("hipertensão", 0.8), ...]
    termos_candidatos_tuplas = buscador.expandir_consulta(query_usuario, n=N_TERMOS)
    termos_candidatos = [termo for termo, score in termos_candidatos_tuplas]
    
    logging.info(f"Termos candidatos para '{query_usuario}': {termos_candidatos}")

    # 4. (Processo 1.2) Vetorizar os termos candidatos com BERT
    # Retorna: [vetor_pe, vetor_hipertensao, ...]
    v_candidatos = buscador.vetorizar_termos_candidatos(termos_candidatos)

    # 5. *** SIMULAÇÃO (sem AG) ***
    # Vamos criar um vetor de pesos "base" manualmente.
    # Onde [1.0, 0.0, 0.0, ...] significa "use APENAS o primeiro termo"
    # (que é a própria consulta "pré-eclâmpsia").
    # Isso nos dará o ranking base ANTES da otimização do AG.
    
    # Garante que a lista de pesos tenha o tamanho correto
    pesos_base = [0.0] * len(v_candidatos)
    if len(pesos_base) > 0:
        pesos_base[0] = 1.0  # Dá peso total ao primeiro termo (a consulta original)
    
    logging.info(f"Pesos de teste (sem AG): {pesos_base}")

    # 6. (Processo 3.1) Criar o vetor de consulta final
    v_query_final = buscador.criar_vetor_consulta_ponderado(v_candidatos, pesos_base)

    # 7. (Processo 3.2) Executar o ranking final
    ranking_final = buscador.ranking_por_similaridade(v_query_final)
    
    end_search = time.time()
    
    # --- Exibir resultados ---
    logging.info(f"Busca concluída em {end_search - start_search:.4f}s")
    logging.info(f"\n--- Top 15 Tabelas para a consulta (base): '{query_usuario}' ---")
    
    for i, (tabela, score) in enumerate(ranking_final[:15]):
        print(f"  {i+1:2}. {tabela:50} (Similaridade: {score:.4f})")
        
    logging.info("--- TESTE DA ETAPA 2 CONCLUÍDO ---")

if __name__ == "__main__":
    main()
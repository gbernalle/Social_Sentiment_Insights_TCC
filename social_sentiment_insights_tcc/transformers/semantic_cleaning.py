import pandas as pd
import re
import logging

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def filter_contextual_noise(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Recebe o DataFrame do Bloco 2 (já em português e limpo)
    e aplica um filtro de contexto para remover falsos positivos
    como "meu", "meus", "me", "das", "da".
    
    'data' é o DataFrame retornado diretamente pelo Bloco 2.
    """
    
    if data.empty:
        logging.warning("O Bloco 2 (Transform) retornou um DataFrame vazio. Pulando o filtro.")
        return pd.DataFrame() 

    if 'text_clean' not in data.columns or 'text_raw' not in data.columns:
        logging.error("DataFrame de entrada não contém 'text_clean' ou 'text_raw'. Verifique o Bloco 2.")
        return data

    logging.info(f"Iniciando filtro de contexto. Registros antes: {len(data)}")

    #Definir palavras-chave que CONFIRMAM o contexto de negócio.
    business_keywords = [
        'mei', 'cnpj', 'empresa', 'imposto', 'simples nacional', 
        'abrir', 'faturamento', 'microempresa', 'negocio', 'contabilidade',
        'empréstimo', 'fisco', 'receita', 'pj', 'sócio', 'lucro',
        'pagar o das', 'pagamento das', 'guia das' # Contextos fortes para DAS
    ]
    
    # Padrões de ruído (pronomes/artigos)
    # \b = "boundary" (limite da palavra), garante que pegue "me" e não "medium"
    noise_pattern_ME = r'\b(meu|meus|minha|minhas|me)\b'
    noise_pattern_DAS = r'\b(das|da)\b' # Pega "das" e "da"

    def check_relevance_row(row):
        text_clean = row['text_clean']
        text_raw = row['text_raw']
        
        if not isinstance(text_clean, str) or not isinstance(text_raw, str):
            return False
            
        # MANTÉM: Se tiver palavras-chave de negócio de alta confiança
        if any(keyword in text_clean for keyword in business_keywords):
            return True  
        
        # MANTÉM: Se "DAS" (o imposto) aparecer em MAIÚSCULAS no texto original
        if "DAS" in text_raw:
            return True

        # REMOVE: Se "ME" (do scraper) for na verdade um pronome
        if re.search(noise_pattern_ME, text_clean):
            return False 
            
        # REMOVE: Se "DAS" (do scraper) for na verdade um artigo/preposição
        if re.search(noise_pattern_DAS, text_clean):
            return False
            
        # Se não foi pego por nenhum filtro de ruído, MANTÉM por segurança.
        return True 

    # Aplicar a função de filtro
    keep_mask = data.apply(check_relevance_row, axis=1)
    
    filtered_data = data[keep_mask].reset_index(drop=True)
    
    num_removed = len(data) - len(filtered_data)
    logging.info(f"Filtro de contexto aplicado. {num_removed} registros de ruído (pronomes/artigos) removidos.")
    
    return filtered_data
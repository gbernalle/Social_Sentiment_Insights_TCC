import pandas as pd
import logging
import os
from mage_ai.settings.repo import get_repo_path

try:
    # Importamos AutoTokenizer
    from transformers import pipeline, AutoTokenizer
except ImportError:
    logging.error("Biblioteca 'transformers', 'torch' ou 'AutoTokenizer' não encontrada. Por favor, instale com: pip install transformers torch")
    raise

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

MODEL_NAME = "ricardo-filho/bert-base-portuguese-cased-nli-assin-2"


# --- Carregamento do Modelo e Tokenizer ---
classifier = None
tokenizer = None
MODEL_MAX_LENGTH = 512 

try:
    logging.info(f"Carregando modelo Zero-Shot (da Internet): {MODEL_NAME}...")
    
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        hypothesis_template="Este texto é sobre {}."
    )
    logging.info("Modelo Zero-Shot carregado com sucesso.")

    logging.info(f"Carregando Tokenizer (da Internet): {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logging.info("Tokenizer carregado com sucesso.")
    
except Exception as e:
    logging.error(f"Erro ao carregar modelo/tokenizer da Internet: {e}")
    classifier = None
    tokenizer = None

@transformer
def filter_by_context(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    
    if data.empty:
        logging.warning("O Bloco 2 (Transform) retornou um DataFrame vazio. Pulando o filtro.")
        return pd.DataFrame() 

    if not classifier or not tokenizer:
        logging.error("Modelo de NLP 'Zero-Shot' ou Tokenizer não foi carregado. Abortando.")
        raise Exception("Modelo de NLP 'Zero-Shot' ou Tokenizer não carregado.")

    if 'text_clean' not in data.columns:
        logging.error("DataFrame de entrada não contém a coluna 'text_clean'. Verifique o Bloco 2.")
        return data

    logging.info(f"Iniciando filtro de contexto (Zero-Shot) em {len(data)} registros...")

    # Etiquetas otimizadas
    label_positiva = 'discussão sobre microempreendedorismo, impostos e negócios (MEI, CNPJ, DAS, Simples Nacional)'
    label_negativa = 'outros tópicos (desabafos, relacionamentos, política, conversas gerais)'
    
    candidate_labels = [label_positiva, label_negativa]
    
    # ==========================================================
    # >>> CORREÇÃO (V17) - MARGEM DE SEGURANÇA <<<
    
    logging.info("Calculando o comprimento máximo permitido para os textos (com margem de segurança)...")
    
    pos_label_tokens = tokenizer(label_positiva, add_special_tokens=False)['input_ids']
    neg_label_tokens = tokenizer(label_negativa, add_special_tokens=False)['input_ids']
    
    max_label_len = max(len(pos_label_tokens), len(neg_label_tokens))
    buffer_len = max_label_len + 3 # 3 tokens: [CLS]...[SEP]...[SEP]
    
    # Esta é a correção: Adicionamos uma margem de 10 tokens
    # para compensar qualquer token extra que o pipeline adicione.
    SAFETY_MARGIN = 10 
    
    max_text_len_allowed = MODEL_MAX_LENGTH - buffer_len - SAFETY_MARGIN
    
    logging.info(f"Etiqueta mais longa: {max_label_len} tokens. Buffer: {buffer_len} tokens. Margem: {SAFETY_MARGIN} tokens.")
    logging.info(f"Comprimento máximo permitido para um TEXTO: {max_text_len_allowed} tokens.")
    
    texts_series = data['text_clean'].fillna('')
    token_lengths = [len(ids) for ids in tokenizer(texts_series.tolist(), add_special_tokens=False)['input_ids']]
    
    is_too_long = pd.Series(token_lengths, index=texts_series.index) > max_text_len_allowed
    num_removed = is_too_long.sum()
    
    if num_removed > 0:
        logging.info(f"REMOVENDO {num_removed} registros por excederem o limite (com margem de segurança).")
        data = data[~is_too_long].copy()
    else:
        logging.info("Nenhum texto excedeu o limite do modelo.")
        
    if data.empty:
        logging.warning("Nenhum registro restou após o filtro de comprimento.")
        return pd.DataFrame()
    # ==========================================================
    
    texts_to_classify = data['text_clean'].dropna().tolist()
    
    if not texts_to_classify:
        logging.warning("Não há textos limpos para classificar após os filtros.")
        return pd.DataFrame()

    try:
        logging.info(f"Classificando {len(texts_to_classify)} textos válidos (curtos)...")
        
        results = classifier(
            texts_to_classify, 
            candidate_labels, 
            multi_label=False, 
            batch_size=16
        )
        logging.info("Classificação concluída.")

        df_results = pd.DataFrame(results)
        df_results['contexto'] = df_results['labels'].str[0]
        df_results['contexto_score'] = df_results['scores'].str[0]
        
        data['contexto'] = df_results['contexto'].values
        data['contexto_score'] = df_results['contexto_score'].values

        filtered_data = data[data['contexto'] == label_positiva].copy()
        
        num_removidos_contexto = len(data) - len(filtered_data)
        logging.info(f"Filtro de contexto aplicado. {num_removidos_contexto} registros de 'outros tópicos' removidos.")
        logging.info(f"Registros restantes: {len(filtered_data)}")

        return filtered_data

    except Exception as e:
        logging.error(f"Falha duringa a classificação Zero-Shot: {e}")
        return data
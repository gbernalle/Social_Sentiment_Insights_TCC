import pandas as pd
import logging
import os
from mage_ai.settings.repo import get_repo_path

try:
    from transformers import pipeline, AutoTokenizer
except ImportError:
    logging.error("Biblioteca 'transformers', 'torch' ou 'AutoTokenizer' não encontrada. Por favor, instale com: pip install transformers torch")
    raise

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

MODEL_NAME = "ricardo-filho/bert-base-portuguese-cased-nli-assin-2"

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

candidate_labels = [
    "dúvidas técnicas sobre burocracia (impostos, DAS, CNPJ, nota fiscal)",
    "medo, insegurança, dívidas ou sobrecarga de trabalho",
    "dúvidas ou queixas sobre direitos trabalhistas (férias, INSS, aposentadoria)",
    "conflito entre o 'sonho' de ser chefe e a 'realidade' do trabalho",
    "discussão sobre clientes, marketing e vendas",
]

labels_subject = [
    "Burocracia e Dúvidas",
    "Vulnerabilidade e Risco",
    "Percepção de Direitos",
    "Identidade e Conflito",
    "Operação e Vendas",
]

label_map = dict(zip(candidate_labels, labels_subject))

@transformer #type:ignore
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

    logging.info(f"Iniciando CATEGORIZAÇÃO de contexto (Zero-Shot) em {len(data)} registros...")
    logging.info(f"Calculando o comprimento máximo permitido (baseado nas {len(candidate_labels)} categorias)...")
    
    all_label_tokens = [len(tokenizer(label, add_special_tokens=False)['input_ids']) for label in candidate_labels]
    
    max_label_len = max(all_label_tokens)
    buffer_len = max_label_len + 3 
    
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
                
        logging.info("Classificação concluída. Mapeando resultados...")

        df_results = pd.DataFrame(results) #type: ignore
        
        # Pega a label com maior score (ex: "dúvidas técnicas sobre burocracia...")
        data['categoria_raw'] = df_results['labels'].str[0].values
        
        # Pega o score dessa label
        data['categoria_score'] = df_results['scores'].str[0].values

        # Mapeia a label longa para o "apelido" curto (ex: "Burocracia e Dúvidas")
        # Usando o 'label_map' definido fora da função
        data['categoria_tcc'] = data['categoria_raw'].map(label_map)

        data = data.drop(columns=['categoria_raw']) 

        logging.info(f"Categorização por tópico concluída. {len(data)} registros foram categorizados.")
        logging.info("Novas colunas adicionadas: 'categoria_tcc' e 'categoria_score'")

        return data

    except Exception as e:
        logging.error(f"Falha durante a classificação Zero-Shot: {e}")
        return data
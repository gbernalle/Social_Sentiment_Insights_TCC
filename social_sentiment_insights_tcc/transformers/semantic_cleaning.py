import pandas as pd
import logging
import os
import torch
from mage_ai.settings.repo import get_repo_path

try:
    from transformers import pipeline, AutoTokenizer, BatchEncoding
except ImportError:
    logging.error("Libraries 'transformers', 'torch' or 'AutoTokenizer' not found. Please install them.")
    raise

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

device_id = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if device_id == 0 else "CPU"
logging.info(f"Zero-Shot NLP running on: {device_name}")

MODEL_NAME = "ricardo-filho/bert-base-portuguese-cased-nli-assin-2"
MODEL_MAX_LENGTH = 512 

classifier = None
tokenizer = None

def load_models():
    """Loads the model into VRAM if not already loaded."""
    global classifier, tokenizer
    if classifier is None:
        logging.info(f"Loading Zero-Shot model on {device_name}...")
        try:
            classifier = pipeline(
                "zero-shot-classification",
                model=MODEL_NAME,
                hypothesis_template="Este texto é sobre {}.",
                device=device_id 
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            logging.info("Models loaded into VRAM.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise e
    return classifier, tokenizer

candidate_labels = [
    # Capturing "Pejotização" (Lima & Oliveira): Disguised subordination
    "vaga de emprego mascarada de pessoa jurídica com cumprimento de horário e subordinação a chefe",
    
    # Captures "Precarization" (Antunes): The lack of social protection
    "ausência de direitos trabalhistas, férias remuneradas, décimo terceiro ou segurança social",
    
    # Capture the "Fiscal Risk" (Alert from the Attorney General's Office/Government): The fear of debt.
    "problemas com dívidas de impostos, DAS atrasado, nome sujo ou medo da Receita Federal",
    
    # It captures "Survival Entrepreneurship" (Post-pandemic reality)
    "trabalho por necessidade imediata, bicos, entregas ou luta para pagar contas básicas",
    
    # Capture the "Opportunity Entrepreneurship" (Official/Liberal discourse - for contrast)
    "estratégias de crescimento do negócio, investimentos, marketing e expansão de clientes",
]

labels_subject = [
    "Pejotização e Subordinação",    # The False Self-Employed
    "Precarização de Direitos",      # The loss of the CLT (Consolidation of Labor Laws)
    "Risco Fiscal e Dívida",         # The weight of the State
    "Sobrevivência e Necessidade",   # The reality of the crisis
    "Gestão e Oportunidade",         # The "success" (Control group)
]

label_map = dict(zip(candidate_labels, labels_subject))

@transformer
def filter_by_context(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    
    if data.empty:
        logging.warning("Previous block returned empty DataFrame. Skipping.")
        return pd.DataFrame()

    classifier, tokenizer = load_models()
    
    logging.info(f"Calculando limites de tokens para Zero-Shot...")
    
    reported_max_len = getattr(tokenizer, 'model_max_length', 512)
    if reported_max_len > 512:
        MODEL_MAX_LENGTH = 512
        logging.warning(f"Tokenizer reported max_len={reported_max_len}. Forcing manual limit to 512 to avoid RuntimeError.")
    else:
        MODEL_MAX_LENGTH = reported_max_len
    
    all_label_tokens = [len(tokenizer(label, add_special_tokens=False)['input_ids']) for label in candidate_labels]
    max_label_len = max(all_label_tokens) if all_label_tokens else 0
    
    HYPOTHESIS_TEMPLATE_BUFFER = 12 
    SPECIAL_TOKENS_BUFFER = 3       
    SAFETY_MARGIN = 5               
    
    reserved_tokens = max_label_len + HYPOTHESIS_TEMPLATE_BUFFER + SPECIAL_TOKENS_BUFFER + SAFETY_MARGIN
    max_text_tokens_allowed = MODEL_MAX_LENGTH - reserved_tokens
    
    if max_text_tokens_allowed < 10:
        raise ValueError("Labels too long for this model context.")

    logging.info(f"Hard Limit: {MODEL_MAX_LENGTH} | Reserved: {reserved_tokens} | Max Input Text: {max_text_tokens_allowed} tokens")

    def truncate_text_by_tokens(text: str) -> str:
        if not text or pd.isna(text):
            return ""
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_text_tokens_allowed:
            return text
            
        truncated_ids = tokens[:max_text_tokens_allowed]
        
        return tokenizer.decode(truncated_ids, skip_special_tokens=True)

    logging.info(f"Starting Zero-Shot Classification on {len(data)} records...")
    
    try:
        texts_raw = data['text_clean'].fillna('').astype(str).tolist()
        texts_to_classify = [truncate_text_by_tokens(t) for t in texts_raw]

        if len(texts_to_classify) > 0:
            len_first = len(tokenizer.encode(texts_to_classify[0], add_special_tokens=False))
            logging.info(f"Sample truncated length (tokens): {len_first} (Limit: {max_text_tokens_allowed})")

        results = classifier(
            texts_to_classify, 
            candidate_labels, 
            multi_label=False, 
            batch_size=32,
            truncation=True
        )
        
        df_results = pd.DataFrame(results)
        
        data['category_raw'] = df_results['labels'].str[0].values
        data['category_score'] = df_results['scores'].str[0].values
        
        if 'label_map' in globals():
            data['category_tcc'] = data['category_raw'].map(label_map)
        else:
             data['category_tcc'] = data['category_raw']
        
        if 'category_raw' in data.columns:
            data = data.drop(columns=['category_raw'])

        base_path = get_repo_path() if 'get_repo_path' in globals() else "."
        cache_path = os.path.join(base_path, "cache_semantic.parquet")
        data.to_parquet(cache_path)
        logging.info(f"Checkpoint saved at: {cache_path}")

        return data

    except RuntimeError as re:
        logging.error(f"RuntimeError (likely CUDA OOM or Shape Mismatch): {re}")
        raise re
    except Exception as e:
        logging.error(f"Generic Error during Zero-Shot: {str(e)}")
        raise e
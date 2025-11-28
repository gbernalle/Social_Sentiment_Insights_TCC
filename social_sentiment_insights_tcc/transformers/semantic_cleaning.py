import pandas as pd
import logging
import os
import torch
from mage_ai.settings.repo import get_repo_path

try:
    from transformers import pipeline, AutoTokenizer
except ImportError:
    logging.error("Libraries 'transformers', 'torch' or 'AutoTokenizer' not found. Please install them.")
    raise

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

# --- GPU Configuration ---
device_id = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if device_id == 0 else "CPU"
logging.info(f"Zero-Shot NLP running on: {device_name}")

MODEL_NAME = "ricardo-filho/bert-base-portuguese-cased-nli-assin-2"
MODEL_MAX_LENGTH = 512 

# Global variables for model caching in memory
classifier = None
tokenizer = None

def load_models():
    """Loads the model into VRAM if not already loaded."""
    global classifier, tokenizer
    if classifier is None:
        logging.info(f"Loading Zero-Shot model on {device_name}...")
        try:
            # The 'device' parameter sends the model directly to the GPU
            classifier = pipeline(
                "zero-shot-classification",
                model=MODEL_NAME,
                hypothesis_template="Este texto é sobre {}.", # Hypothesis remains in PT because the model is PT
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

@transformer #type:ignore
def filter_by_context(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    
    if data.empty:
        logging.warning("Previous block returned empty DataFrame. Skipping.")
        return pd.DataFrame()

    classifier, tokenizer = load_models()

    logging.info(f"Starting Zero-Shot Classification on {len(data)} records...")
    
    texts_to_classify = data['text_clean'].fillna('').tolist()
    
    try:
        # GPU Magic: batch_size > 1
        # With 6GB VRAM, batch_size=32 is generally safe for BERT Base.
        results = classifier(
            texts_to_classify, 
            candidate_labels, 
            multi_label=False, 
            batch_size=32,
            truncation=True 
        )
        
        df_results = pd.DataFrame(results) #type:ignore
        
        # Get the label with the highest score
        data['category_raw'] = df_results['labels'].str[0].values
        data['category_score'] = df_results['scores'].str[0].values
        
        # Map to English labels
        data['category_tcc'] = data['category_raw'].map(label_map)
        
        # Drop the raw Portuguese label column to keep it clean
        data = data.drop(columns=['category_raw'])

        # Save local checkpoint (safety measure)
        cache_path = os.path.join(get_repo_path(), "cache_semantic.parquet")
        data.to_parquet(cache_path)
        logging.info(f"Checkpoint saved at: {cache_path}")

        return data

    except Exception as e:
        logging.error(f"Error during GPU processing: {e}")
        # Hint: Try reducing batch_size if Out Of Memory (OOM) occurs
        raise e
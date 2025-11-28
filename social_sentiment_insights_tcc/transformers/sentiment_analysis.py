import pandas as pd
import logging
import os
import torch
from mage_ai.settings.repo import get_repo_path

try:
    from transformers import pipeline
except ImportError:
    logging.error("Please install transformers and torch.")
    raise

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

# --- GPU Configuration ---
device_id = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if device_id == 0 else "CPU"
logging.info(f"Sentiment Analysis running on: {device_name}")

# Logic to load local model or fallback to HuggingFace
MODEL_PATH_LOCAL = os.path.join(get_repo_path(), "local_models", "sentiment_model")
# Using a specific multilingual model good for social media sentiment
MODEL_ID_FALLBACK = "cardiffnlp/twitter-xlm-roberta-base-sentiment" 

sentiment_pipe = None

def load_model():
    """Loads sentiment model into memory."""
    global sentiment_pipe
    if sentiment_pipe is None:
        try:
            model_to_use = MODEL_PATH_LOCAL if os.path.exists(MODEL_PATH_LOCAL) else MODEL_ID_FALLBACK
            logging.info(f"Loading model: {model_to_use}")
            
            sentiment_pipe = pipeline(
                "sentiment-analysis", #type:ignore
                model=model_to_use, 
                device=device_id,
                truncation=True, #type:ignore
                max_length=512 
            )
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise e
    return sentiment_pipe

@transformer #type:ignore
def analyze_sentiment(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    pipe = load_model()
    
    texts = data['text_clean'].fillna('').tolist()
    logging.info(f"Analyzing sentiment of {len(texts)} texts (Batch processing on GPU)...")

    # Aggressive batch size for GPU (64 or 128 is usually fine for short text)
    try:
        results = pipe(texts, batch_size=64)
        
        # Output depends on the model. XLM-Roberta returns 'positive', 'negative', 'neutral'
        # or 'LABEL_0', 'LABEL_1', etc.
        
        labels = [r['label'] for r in results]
        scores = [r['score'] for r in results]
        
        data['sentiment'] = labels
        data['sentiment_score'] = scores
        
        # Optional Mapping (Adjust based on the specific model used)
        # cardiffnlp model returns: label_0 (negative), label_1 (neutral), label_2 (positive)
        map_labels = {
            'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive',
            'negative': 'Negative', 'neutral': 'Neutral', 'positive': 'Positive',
            '1 star': 'Negative', '5 stars': 'Positive' 
        }
        
        # Normalize labels to English standard
        if data['sentiment'].iloc[0] in map_labels:
             data['sentiment'] = data['sentiment'].map(map_labels).fillna(data['sentiment'])

        # Save checkpoint
        cache_path = os.path.join(get_repo_path(), "cache_sentiment.parquet")
        data.to_parquet(cache_path)
        logging.info(f"Checkpoint saved at: {cache_path}")
        
        return data

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise e
from mage_ai.data_preparation.decorators import transformer
import pandas as pd
import logging
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import torch
from mage_ai.settings.repo import get_repo_path

# --- GPU Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"BERTopic running on: {device}")

@transformer
def generate_topics(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    
    if df.empty:
        return df

    # Final cleanup to ensure BERTopic doesn't break
    df = df.dropna(subset=['text_clean', 'created_at']).copy()
    docs = df['text_clean'].tolist()
    timestamps = df['created_at'].tolist()

    logging.info(f"Starting BERTopic on {len(docs)} documents...")

    # 1. Load Embeddings (Uses GPU)
    # Lightweight multilingual model efficient for Portuguese
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

    # 2. Model Configuration
    # min_topic_size=50 to avoid noise in large datasets
    topic_model = BERTopic(
        embedding_model=embedding_model, 
        min_topic_size=50, 
        verbose=True,
        calculate_probabilities=False,
        n_gram_range=(1, 2) # Captures phrases like "falso autonomo"
    )

    # 3. Fit and Transform
    topics, probs = topic_model.fit_transform(docs)
    
    # 4. Temporal Analysis (Dynamic Topic Modeling - DTM)
    # This is crucial for comparing Pandemic vs Post-Pandemic periods
    logging.info("Generating Topics Over Time (DTM)...")
    
    # nr_bins=20 groups time into 20 intervals for easier visualization
    topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
    
    # SAVE TEMPORAL DATA (Critical for Looker Studio)
    # Mage expects a single DataFrame return in the main flow.
    # We save the DTM to a separate CSV for later loading or auxiliary table creation.
    dtm_path = os.path.join(get_repo_path(), "topics_over_time.csv")
    topics_over_time.to_csv(dtm_path, index=False)
    logging.info(f"Temporal data saved at: {dtm_path}")

    # 5. Enrich Main DataFrame
    df['topic_id'] = topics
    topic_info = topic_model.get_topic_info()
    
    # Create readable name map
    # Uses the top 3 words of the topic to form the name
    topic_map = dict(zip(topic_info['Topic'], topic_info['Name']))
    df['topic_name'] = df['topic_id'].map(topic_map)
    
    # Optional: Remove outliers (-1) if you want a cleaner final dataset
    # df = df[df['topic_id'] != -1]

    return df
from mage_ai.settings.repo import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.google_bigquery import GoogleBigQuery
from pandas import DataFrame
import os
import logging
import pandas as pd

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter #type:ignore
def export_data_to_big_query(data: DataFrame, *args, **kwargs) -> None:
    """
    Saves the final DataFrame to a Google BigQuery table.
    Includes data type sanitization to prevent errors with NLP outputs.
    """

    # --- CONFIGURATION ---
    # Ensure these match your GCP setup
    project_id = "desafio-estagio-concert"
    dataset_id = "First_Test_Sentiment_Analysis"
    table_id = 'analise_reddit_v1'
    
    # We use 'default' because we set up io_config.yaml to use env vars under 'default'
    config_profile = 'default' 
    
    table_full_id = f'{project_id}.{dataset_id}.{table_id}'

    # --- INITIAL VALIDATION ---
    if data is None or data.empty:
        logging.warning("The DataFrame from the previous block is empty. Nothing to export.")
        return

    # --- DATA SANITIZATION (CRITICAL FOR NLP/BIGQUERY) ---
    # BigQuery does not accept column names with spaces or special characters
    data.columns = [c.lower().replace(' ', '_').replace('-', '_') for c in data.columns]

    # Convert complex types (lists, numpy arrays from BERT) to string
    # BigQuery often fails if we try to upload lists directly without a defined SCHEMA
    for col in data.columns:
        # Check if column object type contains lists or dicts and convert to string representation
        if data[col].dtype == 'object':
            # Attempt to convert complex structures to string
            data[col] = data[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
            # Ensure everything else is string to avoid mixed-type errors
            data[col] = data[col].astype(str)
        
        # Ensure dates are in a format BigQuery understands
        if 'created_at' in col or 'date' in col:
            try:
                data[col] = pd.to_datetime(data[col])
            except:
                data[col] = data[col].astype(str)

    # --- EXPORT ---
    config_path = os.path.join(get_repo_path(), 'io_config.yaml')
    
    logging.info(f"Exporting {len(data)} rows to BigQuery table: {table_full_id}...")
    logging.info(f"Columns being exported: {list(data.columns)}")

    try:
        loader = ConfigFileLoader(config_path, config_profile) #type:ignore
        
        GoogleBigQuery.with_config(loader).export(
            data,
            table_full_id,
            if_exists='replace', # 'replace' drops and recreates. Use 'append' for historical accumulation.
        )
        logging.info(f"SUCCESS: Data loaded into {table_full_id}.")
        
    except Exception as e:
        logging.error("CRITICAL EXPORT ERROR:")
        logging.error(f"1. Check if Dataset '{dataset_id}' exists in BigQuery (must be created manually first).")
        logging.error(f"2. Check if Service Account has 'BigQuery Data Editor' and 'Job User' roles.")
        logging.error(f"Error details: {e}")
        raise e
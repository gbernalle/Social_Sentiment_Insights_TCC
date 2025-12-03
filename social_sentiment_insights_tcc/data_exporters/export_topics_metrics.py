from mage_ai.settings.repo import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.bigquery import BigQuery
import pandas as pd
import os
import logging

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_topics_to_bq(*args, **kwargs) -> None:
    project_id = "desafio-estagio-concert"
    dataset_id = "First_Test_Sentiment_Analysis"
    table_id = 'historico_topicos_mei'
    config_profile = 'default'
    
    table_full_id = f'{project_id}.{dataset_id}.{table_id}'
    
    file_path = os.path.join(get_repo_path(), "topics_over_time_refined.csv")
    
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}.")
        return

    df_topics = pd.read_csv(file_path)
    
    if 'Timestamp' in df_topics.columns:
        df_topics['Timestamp'] = pd.to_datetime(df_topics['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    if 'Words' in df_topics.columns:
        df_topics['Words'] = df_topics['Words'].astype(str)
    if 'Topic_Group' in df_topics.columns:
        df_topics['Topic_Group'] = df_topics['Topic_Group'].astype(str)

    config_path = os.path.join(get_repo_path(), 'io_config.yaml')
    
    try:
        loader = ConfigFileLoader(config_path, config_profile)
        
        BigQuery.with_config(loader).export(
            df_topics,
            table_full_id,
            if_exists='replace', 
        )
        logging.info(f"SUCCESS: Data loaded into {table_full_id}.")
        
    except Exception as e:
        logging.error("CRITICAL EXPORT ERROR:")
        logging.error(f"Error details: {e}")
        raise e
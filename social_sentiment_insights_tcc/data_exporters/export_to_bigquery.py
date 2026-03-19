from mage_ai.settings.repo import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.bigquery import BigQuery
from pandas import DataFrame
import os
import logging
import pandas as pd

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data_to_big_query(data: DataFrame, *args, **kwargs) -> None:

    project_id = "desafio-estagio-concert"
    dataset_id = "First_Test_Sentiment_Analysis"
    table_id = 'tabela_completa_vFinal'
    config_profile = 'default' 
    table_full_id = f'{project_id}.{dataset_id}.{table_id}'

    if data is None or data.empty:
        logging.warning("The DataFrame from the previous block is empty. Nothing to export.")
        return

    data.columns = [c.lower().replace(' ', '_').replace('-', '_') for c in data.columns]

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
            data[col] = data[col].astype(str)
        
        if 'created_at' in col or 'date' in col:
            try:
                data[col] = pd.to_datetime(data[col])
            except:
                data[col] = data[col].astype(str)

    config_path = os.path.join(get_repo_path(), 'io_config.yaml')
    
    logging.info(f"Exporting {len(data)} rows to BigQuery table: {table_full_id}...")

    try:
        loader = ConfigFileLoader(config_path, config_profile)
        
        BigQuery.with_config(loader).export(
            data,
            table_full_id,
            if_exists='replace', 
        )
        logging.info(f"SUCCESS: Data loaded into {table_full_id}.")
        
    except Exception as e:
        logging.error("CRITICAL EXPORT ERROR:")
        logging.error(f"Error details: {e}")
        raise e
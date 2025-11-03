from mage_ai.settings.repo import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.google_bigquery import GoogleBigQuery
from pandas import DataFrame
import os
import logging

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data_to_big_query(data: DataFrame, *args, **kwargs) -> None:
    """
    Salva o DataFrame final em uma tabela no Google BigQuery.
    """

    project_id = "desafio-estagio-concert"
    dataset_id = "First_Test_Sentiment_Analysis"
    table_id = 'analise_reddit_v1'

    table_full_id = f'{project_id}.{dataset_id}.{table_id}'
    
    if data.empty:
        logging.warning("O DataFrame do Bloco 4 está vazio. Nada para exportar.")
        return

    # 2. Carregar a configuração do io_config.yaml
    config_path = os.path.join(get_repo_path(), 'io_config.yaml')
    config_profile = 'mage-pipeline-robot'

    logging.info(f"Exportando {len(data)} linhas para o BigQuery: {table_full_id}")

    try:
        GoogleBigQuery.with_config(ConfigFileLoader(config_path, config_profile)).export(
            data,
            table_full_id,
            if_exists='replace',  # 'replace' apaga a tabela e recria. Use 'append' para adicionar.
        )
        logging.info(f"Exportação para {table_full_id} concluída com sucesso.")
    
    except Exception as e:
        logging.error(f"Falha ao exportar para o BigQuery. Verifique suas permissões e o io_config.yaml.")
        logging.error(e)
        raise e
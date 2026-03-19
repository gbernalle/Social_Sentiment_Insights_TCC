import pandas as pd
import sidrapy
import logging

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def load_ibge_data(*args, **kwargs) -> pd.DataFrame:
    logging.info("Iniciando extração de dados do IBGE (SIDRA - PNAD Contínua)...")
    
    try:
        # Tabela 7169: Taxa de desocupação (PNAD Contínua Mensal)
        # Usamos period="all" para evitar bugs da API e filtramos no Pandas
        ibge_raw = sidrapy.get_table(
            table_code="6381",
            territorial_level="1",       # 1 = Nível Brasil
            ibge_territorial_code="all",
            variable="4099",             # 9324 = Taxa de desocupação
            period="all"               
        )
        
        df_ibge = ibge_raw.iloc[1:].copy()
        
        # Seleciona e renomeia apenas as colunas que importam
        # V = Valor da Taxa, D2C = Código do Mês/Ano (ex: 202101)
        df_ibge = df_ibge[['V', 'D2C']]
        df_ibge.columns = ['taxa_desemprego', 'mes_ano']
        
        # Converte a coluna de data 
        df_ibge['data_referencia'] = pd.to_datetime(df_ibge['mes_ano'], format='%Y%m')
        
        # Converte a taxa de desemprego de texto para número decimal (float)
        df_ibge['taxa_desemprego'] = df_ibge['taxa_desemprego'].astype(float)
        
        # Corta fora tudo antes de 2021 para alinhar com a linha do tempo do Reddit
        df_ibge = df_ibge[df_ibge['data_referencia'].dt.year >= 2021]
        
        # Organiza o DataFrame 
        df_ibge = df_ibge[['data_referencia', 'taxa_desemprego']].sort_values('data_referencia').reset_index(drop=True)
        
        logging.info(f"Extração concluída com sucesso! {len(df_ibge)} meses de dados recuperados.")
        
        return df_ibge
        
    except Exception as e:
        logging.error(f"Erro ao acessar a API do IBGE: {e}")
        raise e
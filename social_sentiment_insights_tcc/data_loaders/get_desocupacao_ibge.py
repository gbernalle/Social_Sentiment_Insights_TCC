import pandas as pd
import sidrapy #type: ignore
import logging

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader #type: ignore
def load_ibge_data(*args, **kwargs) -> pd.DataFrame:

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
        
        df_ibge['data_referencia'] = pd.to_datetime(df_ibge['mes_ano'], format='%Y%m')
        
        df_ibge['taxa_desemprego'] = df_ibge['taxa_desemprego'].astype(float)        
        
        df_ibge = df_ibge[df_ibge['data_referencia'].dt.year >= 2018]
        
        # Organiza o DataFrame 
        df_ibge = df_ibge[['data_referencia', 'taxa_desemprego']].sort_values('data_referencia').reset_index(drop=True)
                
        return df_ibge
        
    except Exception as e:
        logging.error(f"API Error: {e}")
        raise e
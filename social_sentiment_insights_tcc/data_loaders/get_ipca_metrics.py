import pandas as pd
import sidrapy #type: ignore

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader #type: ignore
def load_data(*args, **kwargs):
    # Tabela 1737: IPCA - Série histórica com número-índice e variações
    # Variável 2265: IPCA - Variação acumulada em 12 meses (%) -> A inflação "real" percebida no ano
    
    try:
        ipca_raw = sidrapy.get_table(
            table_code="1737",
            territorial_level="1",     # 1 = Brasil (Nacional)
            ibge_territorial_code="all",
            variable="2265",           # Variação acumulada em 12 meses
            period="all"               # Histórico completo
        )
    except Exception as e:
        raise RuntimeError(f"Erro ao conectar na API do SIDRA: {e}")
    
    ipca_raw.columns = ipca_raw.iloc[0]
    ipca_raw = ipca_raw[1:].copy()
    
    # Selecionando apenas as colunas que importam para o Data Warehouse
    df_ipca = ipca_raw[['Mês (Código)', 'Valor']].copy()
    
    df_ipca = df_ipca.rename(columns={
        'Mês (Código)': 'mes_ano',
        'Valor': 'inflacao_acumulada_12m'
    })
    
    df_ipca = df_ipca[df_ipca['mes_ano'] >= '201801'].copy()
    
    df_ipca['inflacao_acumulada_12m'] = pd.to_numeric(df_ipca['inflacao_acumulada_12m'], errors='coerce')
    
    df_ipca['data'] = pd.to_datetime(df_ipca['mes_ano'], format='%Y%m')
    df_ipca['ano'] = df_ipca['data'].dt.year
    df_ipca['mes'] = df_ipca['data'].dt.month
    
    df_ipca = df_ipca.sort_values('data').reset_index(drop=True)
        
    return df_ipca

@test #type: ignore
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
    assert not output.empty, 'Empty DataFrame.'
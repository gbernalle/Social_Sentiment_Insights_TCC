import pandas as pd
import requests

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader # type: ignore
def load_data(*args, **kwargs):
    # Função interna para buscar qualquer série do BCB
    def get_bcb_series(series_code, col_name):
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados?formato=json"
        response = requests.get(url)
        response.raise_for_status() # Trava se a API cair
        
        df = pd.DataFrame(response.json())
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        df[col_name] = pd.to_numeric(df['valor'])
        df = df.drop(columns=['valor'])
        return df

    # Puxando o Endividamento das Famílias (Série 20400)
    df_endividamento = get_bcb_series(29037, 'perc_endividamento')

    # Puxando a Inadimplência Pessoa Física (Série 21084)
    df_inadimplencia = get_bcb_series(21084, 'perc_inadimplencia')

    # Cruzando os dois dados pela Data (Join)
    df_final = pd.merge(df_endividamento, df_inadimplencia, on='data', how='inner')

    # Criando colunas de Ano e Mês para casar com os dados do IBGE e TST no Looker
    df_final['ano'] = df_final['data'].dt.year
    df_final['mes'] = df_final['data'].dt.month

    df_final = df_final[df_final['ano'] >= 2018].sort_values('data').reset_index(drop=True)

    return df_final

@test # type: ignore
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
    assert not output.empty, 'Empty DataFrame.'
import pandas as pd
import os
import glob
from mage_ai.settings.repo import get_repo_path

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader #type: ignore
def load_data(*args, **kwargs):
    base_path = get_repo_path()
    folder_path = os.path.join(base_path, 'dados_gov', 'dados_tst') 
    
    file_pattern = os.path.join(folder_path, '*Casos_Novos*.xlsx')
    arquivos = glob.glob(file_pattern)
    
    print(folder_path)

    if not arquivos:
        raise FileNotFoundError(f"File not found")

    arquivo_excel = arquivos[0] 
    
    try:
        dicionario_abas = pd.read_excel(arquivo_excel, sheet_name=None, skiprows=8)
    except Exception as e:
         raise RuntimeError(f"Check 'openpyxl'. Error: {e}")
    
    lista_dfs = []
    
    for nome_aba, df_aba in dicionario_abas.items():        
        if df_aba.empty or len(df_aba.columns) < 2:
            continue
            
        df_aba['Ano_Origem'] = str(nome_aba).strip()
        lista_dfs.append(df_aba)
        
    df_completo = pd.concat(lista_dfs, ignore_index=True)
    
    return df_completo

@test #type: ignore
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
    assert not output.empty, 'Empty DataFrame'
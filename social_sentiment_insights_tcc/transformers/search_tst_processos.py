import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transform(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    colunas_niveis = ['Nível 1', 'Nível 2', 'Nível 3', 'Nível 4', 'Nível 5', 'Nível 6']
    colunas_presentes = [col for col in colunas_niveis if col in data.columns]
    
    # Preenche os espaços vazios (NaN) com string vazia para o texto não quebrar
    for col in colunas_presentes:
        data[col] = data[col].fillna('').astype(str)
        
    data['Assunto_Completo'] = data[colunas_presentes].agg(' | '.join, axis=1)
    
    palavras_chave = ['relação de emprego', 'vínculo empregatício', 'reconhecimento de relação']
    padrao_busca = '|'.join(palavras_chave)
    
    df_filtrado = data[data['Assunto_Completo'].str.lower().str.contains(padrao_busca, na=False)].copy()
    
    if df_filtrado['Total'].dtype == object:
        df_filtrado['Total'] = df_filtrado['Total'].str.replace('.', '', regex=False)
        
    df_filtrado['Total'] = pd.to_numeric(df_filtrado['Total'], errors='coerce').fillna(0).astype(int)
    
    df_filtrado['Ano_Origem'] = pd.to_numeric(df_filtrado['Ano_Origem'], errors='coerce').fillna(0).astype(int)
    
    df_filtrado = df_filtrado[df_filtrado['Ano_Origem'] > 2000]
    
    df_agrupado = df_filtrado.groupby('Ano_Origem')['Total'].sum().reset_index()
    
    df_agrupado = df_agrupado.rename(columns={
        'Ano_Origem': 'ano',
        'Total': 'qtd_processos_vinculo_emprego'
    })
    
    df_agrupado = df_agrupado.sort_values('ano').reset_index(drop=True)
        
    return df_agrupado
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from mage_ai.settings.repo import get_repo_path
import os

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

base_path = get_repo_path() if 'get_repo_path' in globals() else "."

@data_loader # type: ignore
def load_data(*args, **kwargs):
    def get_bcb_series(series_code, col_name, data_inicial):
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados?formato=json&dataInicial={data_inicial}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() 
        
        if not response.text.strip():
            raise ValueError("A API do Banco Central retornou um corpo vazio (sem dados).")
            
        try:
            dados_json = response.json()
        except requests.exceptions.JSONDecodeError:
            raise ValueError(f"O BCB bloqueou a requisição e retornou HTML. Retorno: {response.text[:250]}")
        
        df = pd.DataFrame(dados_json)
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        df[col_name] = pd.to_numeric(df['valor'])
        df = df.drop(columns=['valor'])
        return df

    df_selic = get_bcb_series(432, 'taxa_selic', '01/01/2018')

    df_selic['ano'] = df_selic['data'].dt.year
    df_selic['mes'] = df_selic['data'].dt.month

    df_selic = df_selic.sort_values('data').reset_index(drop=True)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'serif']
    sns.set_theme(style="white")

    fig, ax = plt.subplots(figsize=(12, 7))
    color_main = '#4C72B0'
    x = df_selic['data']
    y = df_selic['taxa_selic']

    ax.plot(x, y, color=color_main, linewidth=2.5)
    ax.fill_between(x, y, color=color_main, alpha=0.15)

    anos = df_selic['data'].dt.year.unique()
    altura_texto = 15.5 

    for ano in anos:
        if ano >= 2019:
            data_inicio_ano = pd.to_datetime(f"{ano}-01-01")

            if data_inicio_ano <= x.max():
                ax.axvline(x=data_inicio_ano, color='#E5E5E5', linestyle='--', linewidth=1.5, zorder=0)
                ax.text(data_inicio_ano, altura_texto, str(ano), ha='center', va='center',
                        fontsize=14, fontfamily='serif', color='#333333',
                        bbox=dict(facecolor='#F8F9FA', edgecolor='#E5E5E5', boxstyle='round,pad=0.3', linewidth=1))

    ax.set_ylim(0, 16.5)
    ax.set_xlim(x.min(), x.max())

    y_ticks = range(0, 17, 2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(i) for i in y_ticks], fontsize=14, color='#555555')

    ax.set_ylabel("Taxa Selic (%)", fontsize=16, color='#333333', labelpad=15, family='serif')
    ax.set_xlabel("Período Analisado", fontsize=16, color='#333333', labelpad=15, family='serif')

    ax.set_xticks([])
    
    sns.despine(top=True, right=True, left=True, bottom=False)
    ax.spines['bottom'].set_color('#CCCCCC')

    plt.tight_layout()
    plot_path = os.path.join(base_path, "grafico_selic_tcc.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Gráfico da Selic salvo com sucesso em: {plot_path}")

    dt_path = os.path.join(base_path, "df_selic_bcb.csv")
    df_selic.to_csv(dt_path, index=False)

    return df_selic

@test # type: ignore
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
    assert not output.empty, 'Empty DataFrame.'
    assert 'taxa_selic' in output.columns, 'Coluna de taxa_selic não encontrada.'
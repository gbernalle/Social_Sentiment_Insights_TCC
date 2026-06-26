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
    def get_bcb_series(series_code, col_name):
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados?formato=json"
        response = requests.get(url)
        response.raise_for_status() 
        
        df = pd.DataFrame(response.json())
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        df[col_name] = pd.to_numeric(df['valor'])
        df = df.drop(columns=['valor'])
        return df

    df_endividamento = get_bcb_series(29037, 'perc_endividamento')

    df_inadimplencia = get_bcb_series(21084, 'perc_inadimplencia')

    df_final = pd.merge(df_endividamento, df_inadimplencia, on='data', how='inner')

    df_final['ano'] = df_final['data'].dt.year
    df_final['mes'] = df_final['data'].dt.month

    df_final = df_final[df_final['ano'] >= 2018].sort_values('data').reset_index(drop=True)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'serif']
    sns.set_theme(style="white")
    color_main = '#4C72B0'
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    x1 = df_final['data']
    y1 = df_final['perc_endividamento']

    ax1.plot(x1, y1, color=color_main, linewidth=2.5)
    ax1.fill_between(x1, y1, color=color_main, alpha=0.15)

    altura_texto1 = y1.max() + 1.2
    anos = df_final['data'].dt.year.unique()
    
    for ano in anos:
        if ano >= 2019:
            data_inicio_ano = pd.to_datetime(f"{ano}-01-01")
            if data_inicio_ano <= x1.max():
                ax1.axvline(x=data_inicio_ano, color='#E5E5E5', linestyle='--', linewidth=1.5, zorder=0)
                ax1.text(data_inicio_ano, altura_texto1, str(ano), ha='center', va='center',
                        fontsize=14, fontfamily='serif', color='#333333',
                        bbox=dict(facecolor='#F8F9FA', edgecolor='#E5E5E5', boxstyle='round,pad=0.3', linewidth=1))

    ax1.set_ylim(int(y1.min()) - 1, altura_texto1 + 1)
    ax1.set_xlim(x1.min(), x1.max())

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{int(val)}%"))
    ax1.tick_params(axis='y', labelsize=14, colors='#555555')
    ax1.set_ylabel("Endividamento das Famílias (%)", fontsize=16, color='#333333', labelpad=15, family='serif')
    ax1.set_xlabel("Período Analisado", fontsize=16, color='#333333', labelpad=15, family='serif')
    ax1.set_xticks([])

    sns.despine(ax=ax1, top=True, right=True, left=True, bottom=False)
    ax1.spines['bottom'].set_color('#CCCCCC')

    plt.tight_layout()
    plot_path1 = os.path.join(base_path, "grafico_endividamento_tcc.png")
    fig1.savefig(plot_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
   
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    x2 = df_final['data']
    y2 = df_final['perc_inadimplencia']

    ax2.plot(x2, y2, color=color_main, linewidth=2.5)
    ax2.fill_between(x2, y2, color=color_main, alpha=0.15)

    altura_texto2 = y2.max() + 0.3 
    
    for ano in anos:
        if ano >= 2019:
            data_inicio_ano = pd.to_datetime(f"{ano}-01-01")
            if data_inicio_ano <= x2.max():
                ax2.axvline(x=data_inicio_ano, color='#E5E5E5', linestyle='--', linewidth=1.5, zorder=0)
                ax2.text(data_inicio_ano, altura_texto2, str(ano), ha='center', va='center',
                        fontsize=14, fontfamily='serif', color='#333333',
                        bbox=dict(facecolor='#F8F9FA', edgecolor='#E5E5E5', boxstyle='round,pad=0.3', linewidth=1))

    ax2.set_ylim(2.5, altura_texto2 + 0.3)
    ax2.set_xlim(x2.min(), x2.max())

    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val:.1f}%".replace('.', ',')))
    ax2.tick_params(axis='y', labelsize=14, colors='#555555')
    ax2.set_ylabel("Taxa de Inadimplência (%)", fontsize=16, color='#333333', labelpad=15, family='serif')
    ax2.set_xlabel("Período Analisado", fontsize=16, color='#333333', labelpad=15, family='serif')
    ax2.set_xticks([])

    sns.despine(ax=ax2, top=True, right=True, left=True, bottom=False)
    ax2.spines['bottom'].set_color('#CCCCCC')

    plt.tight_layout()
    plot_path2 = os.path.join(base_path, "grafico_inadimplencia_tcc.png")
    fig2.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Gráfico de Inadimplência salvo com sucesso em: {plot_path2}")
    
    dt_path = os.path.join(base_path,"df_endiv_inadi_bc.csv")
    df_final.to_csv(dt_path,index=False)

    return df_final

@test # type: ignore
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
    assert not output.empty, 'Empty DataFrame.'
import pandas as pd
import sidrapy #type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from mage_ai.settings.repo import get_repo_path
import os

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

base_path = get_repo_path() if 'get_repo_path' in globals() else "."

@data_loader #type: ignore
def load_data(*args, **kwargs):
    try:
        ipca_raw = sidrapy.get_table(
            table_code="1737",
            territorial_level="1",
            ibge_territorial_code="all",
            variable="2265",
            period="all"
        )
    except Exception as e:
        raise RuntimeError(f"Erro ao conectar na API do SIDRA: {e}")
    
    ipca_raw.columns = ipca_raw.iloc[0]
    ipca_raw = ipca_raw[1:].copy()
    
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

    print("Gerando gráfico do IPCA Acumulado (IBGE)...")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'serif']
    sns.set_theme(style="white")

    fig, ax = plt.subplots(figsize=(12, 7))
    color_main = '#4C72B0' 
    
    x = df_ipca['data']
    y = df_ipca['inflacao_acumulada_12m']

    ax.plot(x, y, color=color_main, linewidth=2.5)
    ax.fill_between(x, y, color=color_main, alpha=0.15)

    anos = df_ipca['data'].dt.year.unique()
    for ano in anos:
        if ano >= 2019:
            data_inicio_ano = pd.to_datetime(f"{ano}-01-01")

            if data_inicio_ano <= x.max():
                ax.axvline(x=data_inicio_ano, color='#E5E5E5', linestyle='--', linewidth=1.5, zorder=0)
                ax.text(data_inicio_ano, 13.5, str(ano), ha='center', va='center',
                        fontsize=14, fontfamily='serif', color='#333333',
                        bbox=dict(facecolor='#F8F9FA', edgecolor='#E5E5E5', boxstyle='round,pad=0.3', linewidth=1))

    ax.set_ylim(0, 14.5)
    ax.set_xlim(x.min(), x.max())

    y_ticks = range(0, 15, 2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{i}%" for i in y_ticks], fontsize=14, color='#555555')

    ax.set_ylabel("IPCA Acumulado 12m (%)", fontsize=16, color='#333333', labelpad=15, family='serif')
    ax.set_xlabel("Período Analisado", fontsize=16, color='#333333', labelpad=15, family='serif')

    ax.set_xticks([])

    sns.despine(top=True, right=True, left=True, bottom=False)
    ax.spines['bottom'].set_color('#CCCCCC')

    plt.tight_layout()
    plot_path = os.path.join(base_path, "grafico_ipca_tcc.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Gráfico IPCA salvo com sucesso em: {plot_path}")

    dt_path = os.path.join(base_path,"df_ipca_ibge.csv")
    df_ipca.to_csv(dt_path,index=False)    

    return df_ipca

@test #type: ignore
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
    assert not output.empty, 'Empty DataFrame.'
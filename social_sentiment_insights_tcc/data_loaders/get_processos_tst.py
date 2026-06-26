import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from mage_ai.settings.repo import get_repo_path

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader #type: ignore
def load_data(*args, **kwargs):
    base_path = get_repo_path() if 'get_repo_path' in globals() else "."
    folder_path = os.path.join(base_path, 'dados_gov', 'dados_tst') 
    
    file_pattern = os.path.join(folder_path, '*Casos_Novos*.xlsx')
    arquivos = glob.glob(file_pattern)
    
    print(f"Buscando arquivos em: {folder_path}")

    if not arquivos:
        raise FileNotFoundError(f"File not found in {folder_path}")

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
    
    print("Gerando gráfico de Volume de Casos (TST)...")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'serif']
    sns.set_theme(style="white")

    fig, ax = plt.subplots(figsize=(10, 6))
    color_main = '#4C72B0'

    x_labels = ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '* Até\nMarço/2025']
    valores = [10.8, 12.3, 12.7, 10.0, 9.6, 8.3, 18.5, 9.3]

    bars = ax.bar(x_labels, valores, color=color_main, width=0.65)

    for bar in bars:
        yval = bar.get_height()
        texto = f"{yval:.1f} mil".replace('.', ',') if yval % 1 != 0 else f"{int(yval)} mil"
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.3, texto,
                ha='center', va='bottom', fontsize=12, color='#333333', 
                fontweight='bold', fontfamily='serif')

    ax.set_ylim(0, 22)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_yticklabels(['0', '5 mil', '10 mil', '15 mil', '20 mil'], fontsize=14, color='#555555')
    ax.set_ylabel('Volume de Casos', fontsize=16, color='#333333', labelpad=15)

    ax.tick_params(axis='x', labelsize=14, colors='#333333')

    sns.despine(top=True, right=True, left=True, bottom=False)
    ax.spines['bottom'].set_color('#CCCCCC')

    plt.tight_layout()
    plot_path = os.path.join(base_path, "grafico_volume_casos_tst_tcc.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Gráfico TST salvo com sucesso em: {plot_path}")
   
    return df_completo

@test
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
    assert not output.empty, 'Empty DataFrame'
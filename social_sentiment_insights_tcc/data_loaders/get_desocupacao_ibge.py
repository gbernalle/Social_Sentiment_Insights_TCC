import pandas as pd
import sidrapy #type: ignore
import logging
from mage_ai.settings.repo import get_repo_path
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

base_path = get_repo_path() if 'get_repo_path' in globals() else "."

@data_loader #type: ignore
def load_ibge_data(*args, **kwargs) -> pd.DataFrame:

    try:
        ibge_raw = sidrapy.get_table(
            table_code="6381",
            territorial_level="1",
            ibge_territorial_code="all",
            variable="4099",
            period="all"               
        )
        
        df_ibge = ibge_raw.iloc[1:].copy()
        
        df_ibge = df_ibge[['V', 'D2C']]
        df_ibge.columns = ['taxa_desemprego', 'mes_ano']
        
        df_ibge['data_referencia'] = pd.to_datetime(df_ibge['mes_ano'], format='%Y%m')
        df_ibge['taxa_desemprego'] = df_ibge['taxa_desemprego'].astype(float)        
        
        df_ibge = df_ibge[df_ibge['data_referencia'].dt.year >= 2018]
        df_ibge = df_ibge[['data_referencia', 'taxa_desemprego']].sort_values('data_referencia').reset_index(drop=True)

       
        logging.info("Gerando gráfico da Taxa de Desocupação (IBGE)...")
        
        # Configurações base
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'serif']
        sns.set_theme(style="white") 
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        color_main = '#4C72B0' 
        
        x = df_ibge['data_referencia']
        y = df_ibge['taxa_desemprego']
        
        ax.plot(x, y, color=color_main, linewidth=2.5)
        ax.fill_between(x, y, color=color_main, alpha=0.15) 
        
        x_num = mdates.date2num(x)
        z = np.polyfit(x_num, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x_num), color='#A0A0A0', linestyle=':', linewidth=2, alpha=0.8) 
        
        anos = df_ibge['data_referencia'].dt.year.unique()
        for ano in anos:
            if ano >= 2019: 
                data_inicio_ano = pd.to_datetime(f"{ano}-01-01")
                
                if data_inicio_ano <= x.max():
                    ax.axvline(x=data_inicio_ano, color='#E5E5E5', linestyle='--', linewidth=1.5, zorder=0)
                    ax.text(data_inicio_ano, 15.3, str(ano), ha='center', va='center',
                            fontsize=14, fontfamily='Times New Roman', color='#333333',
                            bbox=dict(facecolor='#F8F9FA', edgecolor='#E5E5E5', boxstyle='round,pad=0.3', linewidth=1))

        ax.set_ylim(0, 16.5) 
        ax.set_xlim(x.min(), x.max())
        
        ax.set_ylabel("Taxa de Desocupação (%)", fontsize=16, color='#333333', labelpad=15, family='Times New Roman')
        ax.set_xlabel("Período Analisado", fontsize=16, color='#333333', labelpad=15, family='Times New Roman')
        
        ax.tick_params(axis='y', labelsize=14, colors='#555555')
        ax.set_xticks([]) 
        
        sns.despine(top=True, right=True, left=True, bottom=False)
        ax.spines['bottom'].set_color('#CCCCCC')

        plt.tight_layout()
        plot_path = os.path.join(base_path, "grafico_desocupacao_tcc.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        logging.info(f"Gráfico IBGE salvo com sucesso em: {plot_path}")

        dt_path = os.path.join(base_path,"df_desocupacao_ibge.csv")
        df_ibge.to_csv(dt_path,index=False)
            
        return df_ibge
        
    except Exception as e:
        logging.error(f"API Error: {e}")
        raise e
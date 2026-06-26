import pandas as pd
import logging
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from mage_ai.settings.repo import get_repo_path

try:
    from transformers import pipeline
except ImportError:
    logging.error("Please install transformers and torch.")
    raise

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

device_id = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if device_id == 0 else "CPU"
logging.info(f"Sentiment Analysis running on: {device_name}")

MODEL_PATH_LOCAL = os.path.join(get_repo_path(), "local_models", "sentiment_model")
MODEL_ID_FALLBACK = "cardiffnlp/twitter-xlm-roberta-base-sentiment" 

sentiment_pipe = None

def load_model():
    """Loads sentiment model into memory."""
    global sentiment_pipe
    if sentiment_pipe is None:
        try:
            model_to_use = MODEL_PATH_LOCAL if os.path.exists(MODEL_PATH_LOCAL) else MODEL_ID_FALLBACK
            logging.info(f"Loading model: {model_to_use}")
            
            sentiment_pipe = pipeline(
                "sentiment-analysis", 
                model=model_to_use, 
                device=device_id,
                truncation=True, 
                max_length=512 
            )
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise e
    return sentiment_pipe

@transformer 
def analyze_sentiment(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    pipe = load_model()
    
    texts = data['text_clean'].fillna('').astype(str).tolist()
    logging.info(f"Analyzing sentiment of {len(texts)} texts (Batch processing on GPU)...")

    try:
        results = pipe(texts, batch_size=16)
        
        labels = [r['label'] for r in results]
        scores = [r['score'] for r in results]
        
        data['sentiment_raw'] = labels
        data['sentiment_score'] = scores
        
        map_labels = {
            'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive',
            'negative': 'Negative', 'neutral': 'Neutral', 'positive': 'Positive',
            '1 star': 'Negative', '5 stars': 'Positive' 
        }
        
        data['sentiment'] = data['sentiment_raw'].replace(map_labels)
        
        base_path = get_repo_path() if 'get_repo_path' in globals() else "."
        
        total_records = len(data)
        sent_counts = data['sentiment'].value_counts()
        sent_pct = data['sentiment'].value_counts(normalize=True) * 100

        print(f"Total de registros analisados: {total_records}")
        
        for sent in ['Negative', 'Neutral', 'Positive']:
            count = sent_counts.get(sent, 0)
            pct = sent_pct.get(sent, 0)
            print(f"{sent}: {count} registros ({pct:.2f}%)")
            
        print(f"Média de Confiança do Modelo: {(data['sentiment_score'].mean() * 100):.2f}%")
        
        paleta_sentimentos = {'Negative': '#E63946', 'Neutral': '#F4A261', 'Positive': '#2A9D8F'}
        ordem_sentimentos = ['Negative', 'Neutral', 'Positive']

        plt.figure(figsize=(8, 5))
        sns.set_theme(style="whitegrid")
        ax = sns.countplot(
            data=data, 
            x='sentiment', 
            order=ordem_sentimentos, 
            palette=paleta_sentimentos
        )

        sns.despine(top=True, right=True, left=True, bottom=True)
        
        plt.xlabel('Sentimento Inferido', fontsize=12)
        plt.ylabel('Quantidade de Publicações', fontsize=12)
        
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
                
        plot1_path = os.path.join(base_path, "grafico_sentimento_geral_tcc.png")
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        plt.close()

        if 'category_tcc' in data.columns:
            
            cross_tab = pd.crosstab(data['category_tcc'], data['sentiment'], normalize='index') * 100
            
            if 'Negative' in cross_tab.columns:
                cross_tab = cross_tab.sort_values(by='Negative', ascending=True)
                
            cols = [c for c in ordem_sentimentos if c in cross_tab.columns]
            cross_tab = cross_tab[cols]
            plot_colors = [paleta_sentimentos[c] for c in cols]

            ax2 = cross_tab.plot(kind='barh', stacked=True, color=plot_colors, figsize=(12, 7), width=0.7)
            
            sns.despine(ax=ax2, top=True, right=True, left=True, bottom=True)
            
            plt.xlabel('Proporção (%)', fontsize=12)
            plt.ylabel('Categoria Identificada', fontsize=12)

            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            
            for c in ax2.containers:
                labels = [f'{v.get_width():.1f}%' if v.get_width() > 5 else '' for v in c]
                ax2.bar_label(c, labels=labels, label_type='center', color='white', weight='bold', fontsize=10)
                
            plt.tight_layout()
            plot2_path = os.path.join(base_path, "grafico_sentimento_categoria_tcc.png")
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            plt.close()

        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=data, 
            x='sentiment', 
            y='sentiment_score', 
            order=ordem_sentimentos, 
            palette=paleta_sentimentos,
            showmeans=True,    
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"}
        )

        sns.despine(top=True, right=True, left=True, bottom=True)
        
        plt.xlabel('Sentimento Inferido', fontsize=12)
        plt.ylabel('Score de Confiança (0.0 a 1.0)', fontsize=12)
        
        plot3_path = os.path.join(base_path, "grafico_sentimento_confianca_tcc.png")
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        plt.close()

        if 'sentiment_raw' in data.columns:
            data = data.drop(columns=['sentiment_raw'])

        cache_path = os.path.join(base_path, "cache_sentiment.parquet")
        data.to_parquet(cache_path)
        logging.info(f"Checkpoint saved at: {cache_path}")
        
        return data

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise e
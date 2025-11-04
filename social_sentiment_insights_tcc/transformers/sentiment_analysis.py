import pandas as pd
import logging
import os
from mage_ai.settings.repo import get_repo_path

try:
    from transformers import pipeline
except ImportError:
    logging.error("Biblioteca 'transformers' ou 'torch' não encontrada. Por favor, instale com: pip install transformers torch")
    raise

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

base_path = get_repo_path()
MODEL_NAME = os.path.join(base_path, "local_models", "sentiment_model")
logging.info(f"Verificando se o modelo existe localmente em: {MODEL_NAME}")

try:
    logging.info(f"Carregando modelo de análise de sentimento: {MODEL_NAME}...")
    sentiment_classifier = pipeline(
        model=MODEL_NAME,
        task="sentiment-analysis"
    )
    logging.info("Modelo de sentimento carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar modelo de NLP: {e}")
    logging.error("VERIFIQUE: Você baixou os 4 arquivos do modelo para a pasta 'local_models/sentiment_model'?")
    sentiment_classifier = None

@transformer
def analyze_sentiment(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    if data.empty:
        logging.warning("O Bloco 3 retornou um DataFrame vazio. Pulando a análise.")
        return pd.DataFrame()
    if not sentiment_classifier:
        logging.error("Modelo de NLP não foi carregado. Abortando.")
        raise Exception("Modelo de NLP não carregado.")
    if 'text_clean' not in data.columns:
        logging.error("DataFrame de entrada não contém a coluna 'text_clean'. Verifique o Bloco 3.")
        return data

    logging.info(f"Iniciando Análise de Sentimento em {len(data)} registros...")

    def get_sentiment(text):
        if not text:
            return None, 0.0
        try:
            result = sentiment_classifier(text[:512])[0]
            label = result.get('label')
            score = result.get('score')

            if label == 'LABEL_2':
                sentiment = 'Positive'
            elif label == 'LABEL_0':
                sentiment = 'Negative'
            else: # LABEL_1
                sentiment = 'Neutral'
            return sentiment, score
        except Exception as e:
            logging.warning(f"Falha na análise de sentimento para o texto: {text[:50]}... Erro: {e}")
            return None, 0.0

    data[['sentiment', 'sentiment_score']] = data['text_clean'].apply(
        lambda x: pd.Series(get_sentiment(x))
    )
    logging.info("Análise de sentimento concluída.")
    return data
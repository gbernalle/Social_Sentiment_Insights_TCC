import pandas as pd
import logging

try:
    # Importa o pipeline da Hugging Face
    from transformers import pipeline
except ImportError:
    logging.error("Biblioteca 'transformers' ou 'torch' não encontrada. Por favor, instale com: pip install transformers torch")
    raise

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

# --- Carregando o Modelo de NLP (NOVO MODELO) ---
# Este é um modelo muito robusto e popular para sentimento em Português.
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-portuguese"

try:
    logging.info(f"Carregando modelo de análise de sentimento: {MODEL_NAME}...")
    sentiment_classifier = pipeline(
        model=MODEL_NAME,
        task="sentiment-analysis"
    )
    logging.info("Modelo de sentimento carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar modelo de NLP: {e}")
    sentiment_classifier = None

# --- Fim do carregamento do modelo ---


@transformer
def analyze_sentiment(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Recebe o DataFrame limpo do Bloco 3 e aplica a
    Análise de Sentimento em cada post/comentário.
    """
    
    if data.empty:
        logging.warning("O Bloco 3 (Filtro de Contexto) retornou um DataFrame vazio. Pulando a análise.")
        return pd.DataFrame()
        
    if not sentiment_classifier:
        logging.error("Modelo de NLP não foi carregado. Abortando.")
        raise Exception("Modelo de NLP não carregado.")

    if 'text_clean' not in data.columns:
        logging.error("DataFrame de entrada não contém a coluna 'text_clean'. Verifique o Bloco 3.")
        return data

    logging.info(f"Iniciando Análise de Sentimento em {len(data)} registros...")

    # Função para aplicar o modelo e extrair os resultados
    def get_sentiment(text):
        if not text:
            return None, 0.0
            
        try:
            # Truncamos em 512 caracteres, que é o limite
            result = sentiment_classifier(text[:512])[0]
            
            label = result.get('label')
            score = result.get('score')
            
            # --- LÓGICA DE MAPEAMENTO CORRIGIDA ---
            # O modelo 'cardiffnlp' usa:
            # LABEL_2 = Positive
            # LABEL_1 = Neutral
            # LABEL_0 = Negative
            
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

    # Aplicar a função ao DataFrame
    # Isso pode demorar um pouco se você tiver milhares de linhas
    data[['sentiment', 'sentiment_score']] = data['text_clean'].apply(
        lambda x: pd.Series(get_sentiment(x))
    )

    logging.info("Análise de sentimento concluída.")
    
    return data
import pandas as pd
import json
import re
import logging
from pathlib import Path

try:
    from langdetect import detect, LangDetectException
except ImportError:
    logging.error("Biblioteca 'langdetect' não encontrada. Por favor, instale com: pip install langdetect")
    raise

# Funções auxiliares de limpeza de texto
def clean_text(text):
    """Uma função simples para limpar o texto para NLP."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()  # 1. Converte para minúsculas
    text = re.sub(r'http\S+', '', text)  # 2. Remove URLs
    text = re.sub(r'\[.*?\]', '', text)  # 3. Remove texto entre colchetes (ex: [removed])
    text = re.sub(r'\n', ' ', text)  # 4. Remove quebras de linha
    text = re.sub(r'[^a-z0-9à-ú\s]', '', text)  # 5. Remove pontuação e caracteres especiais (mantém acentos)
    text = re.sub(r'\s+', ' ', text).strip()  # 6. Remove espaços extras
    return text

def detect_language(text):
    """
    Detecta o idioma de um texto. Retorna o código 'pt', 'en', etc.
    Retorna 'und' (indeterminado) se não conseguir detectar ou se o texto for muito curto.
    """
    if not text or not isinstance(text, str) or len(text.strip()) < 25:
        return 'und'
    try:
        return detect(text)
    except LangDetectException:
        return 'und'


@transformer
def transform_raw_reddit_data(data_from_loader, *args, **kwargs):
    """
    Lê todos os JSONs da pasta raw, aplica filtro de idioma,
    limpa e retorna um DataFrame padronizado.
    """
    print("Passou")
    print(data_from_loader)
    print("Info atras")

    # --- VERIFICAÇÃO DE ROBUSTEZ ---
    if not data_from_loader or not isinstance(data_from_loader, list) or len(data_from_loader) == 0:
        logging.warning("O Bloco 1 (Data Loader) não retornou nenhum dado. O Bloco 1 foi executado primeiro?")
        return pd.DataFrame(columns=[
            'id', 'parent_post_id', 'type', 'subreddit', 'created_at', 
            'text_raw', 'text_clean', 'language', 'url'
        ])
    
    logging.info(f"Recebido {len(data_from_loader)} output(s) do bloco anterior.")
    
    # 1. Obter o caminho da pasta onde o Bloco 1 salvou os dados
    # Acessa o primeiro (e único) item da lista
    raw_data_path = Path(data_from_loader[0]['raw_data_path'])
    

    all_data = [] 
    json_files = list(raw_data_path.glob("*.json"))
    logging.info(f"Encontrados {len(json_files)} arquivos JSON para processar.")

    for file_path in json_files:
        subreddit_name = file_path.stem.split('_')[0]
        logging.info(f"Processando arquivo: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                posts = json.load(f) 

                for post in posts:
                    post_text = post.get('post_title', '') + ' ' + post.get('post_body', '')
                    all_data.append({
                        'id': post.get('post_id'), 'parent_post_id': post.get('post_id'),
                        'type': 'post', 'text_raw': post_text,
                        'created_utc': post.get('post_created_utc'),
                        'url': post.get('post_url'), 'subreddit': subreddit_name
                    })
                    
                    for comment in post.get('comments', []):
                        all_data.append({
                            'id': comment.get('comment_id'), 'parent_post_id': post.get('post_id'),
                            'type': 'comment', 'text_raw': comment.get('comment_body'),
                            'created_utc': comment.get('comment_created_utc'),
                            'url': post.get('post_url', '') + comment.get('comment_id', ''),
                            'subreddit': subreddit_name
                        })
        except Exception as e:
            logging.error(f"Erro ao processar o arquivo {file_path.name}: {e}")
            continue 

    logging.info(f"Total de {len(all_data)} registros (posts + comentários) extraídos.")

    if not all_data:
        logging.warning("Nenhum dado foi extraído dos arquivos JSON. Retornando DataFrame vazio.")
        return pd.DataFrame(columns=['id', 'parent_post_id', 'type', 'subreddit', 'created_at', 'text_raw', 'text_clean', 'language', 'url'])

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=['id'])

    logging.info("Iniciando detecção de idioma...")
    df['language'] = df['text_raw'].apply(detect_language)
    logging.info("Detecção de idioma concluída.")

    # --- CORREÇÃO DE LÓGICA ---
    # 1. Filtra para 'pt'
    df_portugues = df[df['language'] == 'pt'].copy()
    
    num_removidos = len(df) - len(df_portugues)
    logging.info(f"Filtro de idioma aplicado. {num_removidos} registros (provavelmente em inglês) removidos.")

    if df_portugues.empty:
        logging.warning("Nenhum dado em português encontrado após o filtro.")
        return pd.DataFrame(columns=['id', 'parent_post_id', 'type', 'subreddit', 'created_at', 'text_raw', 'text_clean', 'language', 'url'])

    # 2. Limpa os dados (agora trabalhando apenas em 'df_portugues')
    df_portugues['created_at'] = pd.to_datetime(df_portugues['created_utc'], unit='s')
    df_portugues['text_clean'] = df_portugues['text_raw'].apply(clean_text)

    # 3. Garante que as operações de filtro não retornem 'None'
    df_portugues_clean = df_portugues.dropna(subset=['text_clean'])
    df_portugues_clean = df_portugues_clean[df_portugues_clean['text_clean'] != '']
    df_portugues_clean = df_portugues_clean[df_portugues_clean['text_clean'] != 'deleted']
    df_portugues_clean = df_portugues_clean[df_portugues_clean['text_clean'] != 'removed']

    final_columns = [
        'id', 'parent_post_id', 'type', 'subreddit', 'created_at', 
        'text_raw', 'text_clean', 'language', 'url'
    ]
    
    # 4. Usa o DataFrame final limpo para o reindex
    df_final = df_portugues_clean.reindex(columns=final_columns)
    df_final = df_final.sort_values(by='created_at').reset_index(drop=True)
    
    logging.info(f"Processamento concluído. {len(df_final)} registros limpos e padronizados.")
    
    return df_final
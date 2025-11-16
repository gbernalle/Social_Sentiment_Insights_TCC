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

def clean_text(text) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z0-9à-ú\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_language(text) -> str:
    if not text or not isinstance(text, str) or len(text.strip()) < 25:
        return 'und'
    try:
        return detect(text)
    except LangDetectException:
        return 'und'


@transformer # type: ignore
def transform_raw_reddit_data(data_from_loader: dict, *args, **kwargs):
    """
    Lê todos os JSONs da pasta raw, aplica filtro de idioma,
    limpa e retorna um DataFrame padronizado.
    """
    
    if not data_from_loader or not isinstance(data_from_loader, dict):
        logging.warning("O Bloco 1 (Data Loader) não retornou um dicionário. O Bloco 1 foi executado primeiro?")
        return pd.DataFrame()

    if 'raw_data_path' not in data_from_loader:
        logging.warning("Dicionário do Bloco 1 não contém a chave 'raw_data_path'.")
        return pd.DataFrame()
    
    logging.info("Recebido com sucesso o dicionário do Bloco 1.")
    
    raw_data_path = Path(data_from_loader['raw_data_path'])
    
    all_data = [] 
    json_files = list(raw_data_path.glob("*.json"))
    logging.info(f"Encontrados {len(json_files)} arquivos JSON para processar em {raw_data_path}")

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
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=['id'])

    logging.info("Iniciando detecção de idioma...")
    df['language'] = df['text_raw'].apply(detect_language)
    logging.info("Detecção de idioma concluída.")

    df_portugues = df[df['language'] == 'pt'].copy()
    
    num_removidos = len(df) - len(df_portugues)
    logging.info(f"Filtro de idioma aplicado. {num_removidos} registros (provavelmente em inglês) removidos.")

    if df_portugues.empty:
        logging.warning("Nenhum dado em português encontrado após o filtro.")
        return pd.DataFrame()

    df_portugues['created_at'] = pd.to_datetime(df_portugues['created_utc'], unit='s')

    pandemic_end_date = pd.to_datetime("2022-05-22")

    df_filtered_date = df_portugues[df_portugues['created_at'] >= pandemic_end_date].copy()

    logging.info(f"Filtro temporal aplicado. Removidos {len(df_portugues) - len(df_filtered_date)} registros pré-pandemia")
    logging.info(f"Registros restantes: {len(df_filtered_date)}")

    if df_filtered_date.empty:
        logging.warning("nenhum dado encontrado após o filtro temporal.")
        return pd.DataFrame()
    
    df_filtered_date['created_at'] = df_filtered_date['created_at'].dt.strftime('%Y-%m-%d')
    df_filtered_date['text_clean'] = df_filtered_date['text_raw'].apply(clean_text)

    df_final_clean = df_filtered_date.dropna(subset=['text_clean']).copy()

    unwanted_texts = ['', 'deleted', 'removed']
    df_final_clean = df_filtered_date.dropna(subset=['text_clean'])
    df_final_clean = df_final_clean[~df_final_clean['text_clean'].isin(unwanted_texts)]

    final_columns = [
        'id', 'parent_post_id', 'type', 'subreddit', 'created_at', 
        'text_raw', 'text_clean', 'language', 'url'
    ]
    
    df_final = df_final_clean.reindex(columns=final_columns) 
    df_final = df_final.sort_values(by='created_at').reset_index(drop=True)
    
    logging.info(f"DataFrame pré-filtro de keyword tem: {len(df_final)} linhas.")

    safe_keywords = r'\b(?:cnpj|simples nacional)\b'
    
    # DAS (Contextualizado - do filtro anterior)
    #    (o das, pagar das, guia das, etc.)
    das_context = r'\b(?:o|do|no|pagar|guia|boleto|valor) das\b'
    
    # Imposto (Contextualizado - FORMA DE SUBSTANTIVO)
    #    Queremos: (pagar imposto, o imposto, imposto de, imposto mei)
    #    NÃO queremos: (me imposto, se imposto)
    imposto_context = (
        r'(?:\b(?:pagar|pago|paguei|o|do|no|um|qual|quanto) imposto\b'  # Contexto ANTES
        r'|\b(imposto) (?:de|do|da|mei|simples|sobre)\b)'                 # Contexto DEPOIS
    )
    
    # 4. MEI (Contextualizado - FORMA DE NEGÓCIO)
    #    Queremos: (o mei, meu mei, sou mei, abrir mei, mei da, mei é)
    #    NÃO queremos: (me deixou mei triste, fiquei mei assim)
    mei_context = (
        r'(?:\b(?:o|do|no|pro|meu|um|abrir|sou|virar|ser|pagar|guia|boleto) mei\b' # Contexto ANTES
        r'|\b(mei) (?:ta|é|atrasado|da|de|pra|cnpj|me)\b)'                        # Contexto DEPOIS
    )
    
    regex_pattern = f'(?:{safe_keywords}|{das_context}|{imposto_context}|{mei_context})'
    
    logging.info(f"Usando padrão de regex Super-Refinado: {regex_pattern}")

    df_com_keywords = df_final[df_final['text_clean'].str.contains(regex_pattern, na=False, case=False)].copy()

    num_removidos = len(df_final) - len(df_com_keywords)
    logging.info(f"Filtro de Keyword (grosso) removeu {num_removidos} posts irrelevantes.")
    logging.info(f"Registros restantes para a IA: {len(df_com_keywords)}")

    if df_com_keywords.empty:
        logging.warning("Nenhum post sobreviveu ao filtro de keywords.")
        return pd.DataFrame()

    df_final = df_com_keywords.reset_index(drop=True)

    logging.info(f"Processamento concluído. {len(df_final)} registros limpos e padronizados.")
    
    return df_final
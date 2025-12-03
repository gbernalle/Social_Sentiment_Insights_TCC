import pandas as pd
import json
import re
import logging
import os
from pathlib import Path
from mage_ai.settings.repo import get_repo_path

try:
    from langdetect import detect, LangDetectException
except ImportError:
    logging.error("Library 'langdetect' not found. Please install with: pip install langdetect")
    raise

def clean_text(text) -> str:

    """Standardizes text: lowercase, removes links, special characters, and extra spaces."""

    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z0-9à-ú\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_language(text) -> str:

    """Detects the language of the text. Returns 'und' if undetectable or too short."""

    if not text or not isinstance(text, str) or len(text.strip()) < 25:
        return 'und'
    try:
        return detect(text)
    except LangDetectException:
        return 'und'

@transformer
def transform_raw_reddit_data(data_from_loader: dict, *args, **kwargs):
    """
    Reads all JSON files, filters by language, applies Regex for keywords, 
    and returns a standardized DataFrame.
    """
    
    if not data_from_loader or not isinstance(data_from_loader, dict):
        logging.warning("Block 1 did not return a valid dictionary.")
        return pd.DataFrame()

    if 'raw_data_path' not in data_from_loader:
        logging.warning("Key 'raw_data_path' not found in loader output.")
        return pd.DataFrame()
    
    raw_data_path = Path(data_from_loader['raw_data_path'])
    all_data = [] 
    json_files = list(raw_data_path.glob("*.json"))
    logging.info(f"Found {len(json_files)} JSON files in {raw_data_path}")

    for file_path in json_files:
        subreddit_name = file_path.stem.split('_')[0]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                posts = json.load(f) 
                for post in posts:
                    post_text = (post.get('post_title', '') or '') + ' ' + (post.get('post_body', '') or '')
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
            logging.error(f"Error processing file {file_path.name}: {e}")
            continue 

    if not all_data:
        logging.warning("No data extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=['id'])

    logging.info("Starting language detection...")
    df['language'] = df['text_raw'].apply(detect_language)
    
    df_portuguese = df[df['language'] == 'pt'].copy()
    logging.info(f"Language filter: {len(df)} -> {len(df_portuguese)} records.")

    if df_portuguese.empty:
        return pd.DataFrame()

    df_portuguese['created_at'] = pd.to_datetime(df_portuguese['created_utc'], unit='s')
        
    df_portuguese['created_at'] = df_portuguese['created_at'].dt.strftime('%Y-%m-%d')
    df_portuguese['text_clean'] = df_portuguese['text_raw'].apply(clean_text)

    unwanted_texts = ['', 'deleted', 'removed']
    df_final_clean = df_portuguese.dropna(subset=['text_clean'])
    df_final_clean = df_final_clean[~df_final_clean['text_clean'].isin(unwanted_texts)]

    safe_keywords = r'\b(?:cnpj|simples nacional|inss|previdência|mei)\b'
    das_context = r'\b(?:o|do|no|pagar|guia|boleto|valor|atrasado) das\b'
    tax_context = r'(?:\b(?:pagar|pago|paguei|o|do|no|um|qual|quanto|sonegar) imposto\b|\b(imposto) (?:de|do|da|mei|simples|sobre|renda)\b)'
    contractor_context = r'(?:\b(?:vaga|contrato|regime|trabalhar|sou|virar|ser|clt|versus|vs|híbrido) pj\b|\b(pj) (?:ou|vs|versus|clt|sem|com|pagar|receber)\b)'
    gig_economy_context = r'\b(?:uber|99|indriver|ifood|rappi|loggi|entregador|motoboy|motorista de aplicativo)\b'
    precarious_context = r'\b(?:sem férias|sem décimo|sem 13º|sem fgts|sem direitos|falso autônomo|pejotização|uberização)\b'
    mei_context = r'(?:\b(?:o|do|no|pro|meu|um|abrir|sou|virar|ser|pagar|guia|boleto) mei\b|\b(mei) (?:ta|é|atrasado|da|de|pra|cnpj|me)\b)'

    regex_pattern = f'(?:{safe_keywords}|{das_context}|{tax_context}|{contractor_context}|{gig_economy_context}|{precarious_context}|{mei_context})'
    
    logging.info(f"Applying refined Regex filtering...")
    df_with_keywords = df_final_clean[df_final_clean['text_clean'].str.contains(regex_pattern, na=False, case=False)].copy()

    logging.info(f"Keyword Filtering: {len(df_final_clean)} -> {len(df_with_keywords)} relevant records.")

    final_columns = ['id', 'parent_post_id', 'type', 'subreddit', 'created_at', 'text_raw', 'text_clean', 'language', 'url']
    df_final = df_with_keywords.reindex(columns=final_columns).reset_index(drop=True)
    
    return df_final
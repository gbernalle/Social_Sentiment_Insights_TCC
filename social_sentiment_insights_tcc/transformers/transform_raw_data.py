import pandas as pd
import json
import re
import logging
import os
from pathlib import Path
from mage_ai.settings.repo import get_repo_path


def clean_text(text) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@transformer
def transform_raw_reddit_data(data_from_loader: dict, *args, **kwargs):
    if not data_from_loader or not isinstance(data_from_loader, dict):
        logging.warning("Block 1 did not return a valid dictionary.")
        return pd.DataFrame()

    if 'raw_data_path' not in data_from_loader:
        logging.warning("Key 'raw_data_path' not found in loader output.")
        return pd.DataFrame()
    
    raw_data_path = Path(data_from_loader['raw_data_path'])
    json_files = list(raw_data_path.glob("*.json"))
    all_data = [] 

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

    df['created_at'] = pd.to_datetime(df['created_utc'], unit='s').dt.strftime('%Y-%m-%d')
    
    noise_pattern = r'\[deleted\]|\[removed\]|\[video\]|\[image\]|http[s]?://|www\.'
    df = df[~df['text_raw'].str.contains(noise_pattern, case=False, na=False)]

    df['text_clean'] = df['text_raw'].apply(clean_text)

    df = df.dropna(subset=['text_clean'])
    df = df[df['text_clean'].str.strip() != '']

    regex_dict = {
        'safe_keywords' : r'(?:\b(?:abrir|fechar|meu|nosso|ter|tenho|usar|emitir|pagar|pagando|desconto|descontar|recolher|contribuir|optante|enquadrado|imposto|guia|tributação|trabalhar|trabalho|atuar)\s+(?:um|o|a|do|da|no|na|pro|pra|para|pelo|como)?\s*(?:cnpj|simples nacional|inss|previdência)\b|\b(?:cnpj|simples nacional|inss|previdência)\s+(?:ativo|inativo|aberto|descontado|recolhido|pago)\b)',
        'das_context' : r'\b(?:o|do|no|pagar|guia|boleto|valor|atrasado)\s+das\b',
        'tax_context' : r'(?:\b(?:pagar|pago|paguei|o|do|no|um|qual|quanto|sonegar)\s+imposto\b|\bimposto\s+(?:de|do|da|mei|simples|sobre|renda)\b)',
        'pj_context' : r'(?:\b(?:vaga|contrato|regime|trabalhar|sou|virar|ser|clt|versus|vs|híbrido)\s+pj\b|\bpj\s+(?:ou|vs|versus|clt|sem|com|pagar|receber)\b)',
        'mei_context' : r'(?:\b(?:o|do|no|pro|meu|um|abrir|sou|virar|ser|pagar|guia|boleto)\s+mei\b|\bmei\s+(?:ta|é|atrasado|da|de|pra|cnpj|me)\b)',
        'uberizacao_context' : r'(?:\b(?:uberizaç[ãa]o|uberizad[oa]s?|motoboys?|motoristas? de aplicativo|entregadores? de aplicativo)\b|\b(?:trabalhar|trabalho|trampo|rodar|rodando|fazer|fazendo|motorista|entregador|corrida|taxa|bloqueado)\s+(?:de|com|no|na|pro|pra|para|o|a)?\s*(?:uber|99|indriver|ifood|rappi|loggi)\b|\b(?:uber|99|indriver|ifood|rappi|loggi)\s+(?:paga|pagando|bloqueou|taxa|corrida|entrega|desconta)\b)',
        'precarious_context' : r'\b(?:sem férias|sem décimo|sem 13º|sem fgts|sem direitos|falso autônomo|pejotização|uberização)\b',
    }
    
    all_patterns = '|'.join(regex_dict.values())
    df_with_keywords = df[df['text_clean'].str.contains(all_patterns, na=False, case=False)].copy()

    for col_name, pattern in regex_dict.items():
        df_with_keywords[col_name] = df_with_keywords['text_clean'].str.contains(pattern, case=False, na=False)
  
    final_columns = [
        'id','subreddit', 'created_at', 
        'text_raw', 'text_clean', 'url'
    ] + list(regex_dict.keys())

    df_final = df_with_keywords.reindex(columns=final_columns).reset_index(drop=True)

    return df_final 
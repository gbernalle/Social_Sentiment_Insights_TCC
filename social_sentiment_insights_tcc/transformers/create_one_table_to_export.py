import pandas as pd
import json
import re
import logging
import os
from pathlib import Path
from mage_ai.settings.repo import get_repo_path

@transformer
def unifyTables(data_from_loader, *args, **kwargs):
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

    return pd.DataFrame(all_data) 
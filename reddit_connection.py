import os
import json
import praw
from dotenv import load_dotenv
#Teste Bypass empresa
from requests import Session
import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter('ignore', InsecureRequestWarning)
custom_session = Session()
custom_session.verify = False
#Até aqui pode apagar no código final

load_dotenv(dotenv_path='keyword.env')
client_id = os.getenv('CLIENT_ID')
secret_key = os.getenv('SECRET_KEY')
password = os.getenv('PASSWORD')
username = os.getenv('USER_REDDIT')

reddit = praw.Reddit(client_id = client_id,
                     client_secret=secret_key,
                     username=username,
                     password=password,
                     user_agent='TccTeste/0.0.1',
                     #Remover depois: Isso é só para o pc da empresa
                     requestor_kwargs={'session': custom_session}
                     )

sub_reddit = reddit.subreddit('apple+iphone+ios')

termos_busca = "iOS 26 OR (novo iOS) OR (atualização iOS)"

all_data = []

# Busca por posts (submissions) que contenham os termos
for post in sub_reddit.search(termos_busca, sort="hot", limit=1000):
    print(f"Post encontrado: {post.title}")
  
    post_data = {
        "post_id": post.id,
        "post_title": post.title,
        "post_created_utc": post.created_utc,
        "comments": []
    }
    
    # Pega todos os comentários --Limit=0 faz isso--
    post.comments.replace_more(limit=0) 
    
    for comment in post.comments.list():
        if hasattr(comment,"body"):
            comment_data = {
                "comment_id": comment.id, # type: ignore
                "comment_body": comment.body, # type: ignore
                "comment_created_utc": comment.created_utc # type: ignore
            }
        post_data["comments"].append(comment_data) # type: ignore

    
    all_data.append(post_data)

file_name = 'reddit_data.json'
with open(file_name, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

print(f"Dados salvos com sucesso em '{file_name}'")
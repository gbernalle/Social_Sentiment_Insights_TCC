import os
import praw
from dotenv import load_dotenv

load_dotenv(dotenv_path='keyword.env')
client_id = os.getenv('CLIENT_ID')
secret_key = os.getenv('SECRET_KEY')
password = os.getenv('PASSWORD')
user_name = os.getenv('USER_REDDIT')

reddit = praw.Reddit(client_id = client_id,
                     client_secret=secret_key,
                     username=user_name,
                     password=password,
                     user_agent='TccTeste/0.0.1'
                     )

sub_reddit = reddit.subreddit('apple+iphone+ios')

termos_busca = "iOS 26 OR (novo iOS) OR (atualização iOS)"

# Busca por posts (submissions) que contenham seus termos
for post in sub_reddit.search(termos_busca, sort="new", limit=1000):
    print(f"Post encontrado: {post.title}")
    # Agora, você pode pegar os comentários desse post
    post.comments.replace_more(limit=0) # Pega todos os comentários
    for comment in post.comments.list():
        print(f"--- Comentário: {comment.body}")
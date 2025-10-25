import os
import json
import praw
import logging
from dataclasses import dataclass
from prawcore import exceptions as praw_exceptions
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class Connection:
    def __init__(self):
        load_dotenv(dotenv_path="keyword.env")
        self.client_id = os.getenv("CLIENT_ID")
        self.secret_key = os.getenv("SECRET_KEY")
        self.password = os.getenv("PASSWORD")
        self.username = os.getenv("USER_REDDIT")
        self.reddit = None
        logging.info("Cliente inicializado, credenciais carregadas.")

    def connect_to_api(self) -> bool:
        try:
            if not all([self.client_id, self.secret_key, self.password, self.username]):
                logging.error("Variáveis de ambiente não foram carregadas corretamente")
                return False

            # Objeto de conexão
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.secret_key,
                username=self.username,
                password=self.password,
                user_agent="TccTeste/0.0.1",
            )

            user = self.reddit.user.me()
            logging.info(f"Conexão bem-sucedida como: {user.name}")  # type: ignore
            return True

        except praw_exceptions.OAuthException as e:
            logging.error(f"Ocorreu um erro de autenticação: {e}")
            self.reddit = None
            return False
        except Exception as e:
            logging.error(f"Erro inesperado ao conectar: {e}")
            self.reddit = None
            return False

    def get_top_posts_and_comments(
        self, subreddit_name: str, search: str, limit: int = 1000
    ) -> list[dict]:
        if not self.reddit:
            logging.error("Não é possível obter posts. Conexão não estabelecida.")
            return []
        try:
            sub_reddit = self.reddit.subreddit(subreddit_name)
            all_data = []
            termos_busca = search

            # Busca por posts (submissions) que contenham os termos
            for post in sub_reddit.search(termos_busca, sort="hot", limit=limit):
                print(f"Post encontrado: {post.title}")

                post_data = {
                    "post_id": post.id,
                    "post_title": post.title,
                    "post_created_utc": post.created_utc,
                    "comments": [],
                }

                # Pega todos os comentários --Limit=0 faz isso--
                post.comments.replace_more(limit=0)

                for comment in post.comments.list():
                    if hasattr(comment, "body"):
                        comment_data = {
                            "comment_id": comment.id,  # type: ignore
                            "comment_body": comment.body,  # type: ignore
                            "comment_created_utc": comment.created_utc,  # type: ignore
                        }
                    post_data["comments"].append(comment_data)  # type: ignore

                all_data.append(post_data)
            return all_data

        except Exception as e:
            logging.error(f"Erro ao buscar posts: {e}")
            return []
        
    def save_in_file(self, file_json, file_name) -> bool:
        try:
            destiny_file = Path("raw_data")
            complete_path = destiny_file / file_name

            with open(complete_path, "w", encoding="utf-8") as f:
                json.dump(file_json, f, ensure_ascii=False, indent=4)
            print(f"Arquivo '{file_name}' salvo com sucesso!")
            return True
        
        except Exception as e:
            logging.error(f"Ocorreu um erro: {e}")
            return False

client = Connection()

if client.connect_to_api():
    logging.info("Iniciando busca de posts...")
    top_posts = client.get_top_posts_and_comments("python", "social sentiment")
    client.save_in_file(top_posts,"social_sentiments.json")
else:
    logging.error("Falha ao conectar. Verificar credenciais ou conexão. Encerrando.")

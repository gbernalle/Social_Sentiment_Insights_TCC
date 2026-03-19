import os
import json
import praw
import logging
import time
import random
import concurrent.futures
from mage_ai.settings.repo import get_repo_path
from dataclasses import dataclass
from prawcore import exceptions as praw_exceptions
from praw.models import Comment
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass
class Reddit_Connection:
    def __init__(self):
        self.client_id = os.getenv("CLIENT_ID")
        self.secret_key = os.getenv("SECRET_KEY")
        self.password = os.getenv("PASSWORD")
        self.username = os.getenv("USER_REDDIT")
        self.reddit = None
        logging.info("Client initialized, Reddit credentials loaded.")

    def connect_to_api_reddit(self) -> bool:
        try:
            if not all([self.client_id, self.secret_key, self.password, self.username]):
                logging.error("Environment variables were not loaded correctly.")
                return False
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.secret_key,
                username=self.username,
                password=self.password,
                user_agent="TccTeste/0.0.1",
            )
            self.reddit.user.me()
            return True
        except praw_exceptions.OAuthException as e:
            logging.error(f"An authentication error occurred: {e}")
            self.reddit = None
            return False
        except Exception as e:
            logging.error(f"Unexpected error when connecting: {e}")
            self.reddit = None
            return False

    def get_top_posts_and_comments(
        self, subreddit_name: str, search_topics: str, limit: int = 1000
    ) -> list[dict]:
        if not self.reddit:
            logging.error("Unable to retrieve posts. Connection not established.")
            return []
        try:
            sub_reddit = self.reddit.subreddit(subreddit_name)
            all_data = []
            for post in sub_reddit.search(search_topics, sort="hot", limit=limit):
                post_data = {
                    "post_id": post.id,
                    "post_title": post.title,
                    "post_body": post.selftext,
                    "post_url": post.url,
                    "is_self_post": post.is_self,
                    "post_created_utc": post.created_utc,
                    "comments": [],
                }
                try:
                    post.comments.replace_more(limit=0)
                except Exception as e:
                    logging.error(
                        f"Error loading more comments for the post {post.id}: {e}"
                    )
                for comment in post.comments.list():
                    if isinstance(comment, Comment):
                        comment_data = {
                            "comment_id": comment.id, 
                            "comment_body": comment.body, 
                            "comment_score": comment.score,
                            "comment_created_utc": comment.created_utc, 
                        }
                        post_data["comments"].append(comment_data) 
                all_data.append(post_data)
            return all_data
        except Exception as e:
            logging.error(f"Error searching for posts: {e}")
            return []

    def save_in_file(self, file_json, file_name) -> bool:
        try:
            base_path = get_repo_path()
            destiny_file = Path(base_path) / "raw_data_reddit"
            
            destiny_file.mkdir(parents=True, exist_ok=True)
            complete_path = destiny_file / file_name
            
            with open(complete_path, "w", encoding="utf-8") as f:
                json.dump(file_json, f, ensure_ascii=False, indent=4)
            return True

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return False

def scrape_and_save_task(task_params):
    top, word = task_params
    logging.info(f"[TASK STARTED] - Subreddit: {top}, Word: {word}")
    
    time.sleep(random.uniform(2.0, 5.0)) 
    
    thread_client = Reddit_Connection()
    if not thread_client.connect_to_api_reddit():
        logging.error(f"[TASK FAILED] - Connection: {top} / {word}")
        return f"FAILURE (Connection): {top} / {word}"
        
    try:
        top_posts = thread_client.get_top_posts_and_comments(top, word)
        
        if not top_posts:
            logging.warning(f"[TASK EMPTY] - No posts found for: {top} / {word}")
            return f"OK (Empty): {top} / {word}"
            
        safe_palavra = word.replace(" ", "_")
        filename = f"{top}_{safe_palavra}.json"
        thread_client.save_in_file(top_posts, filename)
        
        logging.info(f"[TASK COMPLETED] - Saved:{filename} ({len(top_posts)} posts)")
        return f"OK (Saved): {filename}"
        
    except Exception as e:
        logging.error(f"[TASK FAILED] - Error during execution {top} / {word}: {e}")
        return f"FAILURE (Error): {top} / {word}"

@data_loader #type: ignore
def load_reddit_data(*args, **kwargs):
    
    topics = [
        "investimentos", "brasil",  "farialimabets", "empreendedorismo", "MicroEmpresas","devBR",
        "brdev", "Empreendedor", "Liderança", "StartupsAjudaStartups", "Os Fundadores",
        "crescermeunegócio", "capital de risco", "empreendedor avançado", "produtividade", 
        "mídias sociais", "MicroEmpresas", "MeuNegocio","antitrampo"
    ]

    keywords = [
        "ME", "CNPJ", "MEI", "Simples Nacional", "abrir empresa", "imposto MEI", "imposto",
        "imposto ME", "imposto CNPJ", "DAS", "INSS MEI", "governo+empresa", "IRPF",
        "Uber", "iFood", "pj", "emprego","Uberização"
    ]

    check_client = Reddit_Connection()
    if not check_client.connect_to_api_reddit():
        logging.error("Failed to connect. Check credentials. Signing out.")
        raise Exception("Failed to connect. Check credentials.") 

    tasks = [(top, word) for top in topics for word in keywords]
    
    MAX_WORKERS = 3

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(scrape_and_save_task, tasks))

    logging.info("All tasks have been processed.")
    
    for r in results:
        if "FAILURE" in r:
            logging.error(f"Final result: {r}")
        else:
            logging.info(f"Final result: {r}")
            
    base_path = get_repo_path()
    raw_data_path = os.path.join(base_path, "raw_data_reddit")

    logging.info(f"Dados brutos salvos em: {raw_data_path}")
    
    return {"raw_data_path": raw_data_path}
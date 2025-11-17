import os
import json
import praw
import logging
import concurrent.futures
from mage_ai.settings.repo import get_repo_path
from dataclasses import dataclass
from prawcore import exceptions as praw_exceptions
from praw.models import Comment
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass
class Reddit_Connection:
    def __init__(self):
        env_path = os.path.join(get_repo_path(), 'keyword.env')
        logging.info(f"Tentando carregar credenciais de: {env_path}")

        load_dotenv(dotenv_path=env_path) 
        self.client_id = os.getenv("CLIENT_ID")
        self.secret_key = os.getenv("SECRET_KEY")
        self.password = os.getenv("PASSWORD")
        self.username = os.getenv("USER_REDDIT")
        self.reddit = None
        logging.info("Cliente inicializado, credenciais do Reddit carregadas.")

    def connect_to_api_reddit(self) -> bool:
        try:
            if not all([self.client_id, self.secret_key, self.password, self.username]):
                logging.error("Variáveis de ambiente não foram carregadas corretamente")
                return False
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.secret_key,
                username=self.username,
                password=self.password,
                user_agent="TccTeste/0.0.1",
            )
            user = self.reddit.user.me()
            logging.info(f"Conexão bem-sucedida como: {user.name}") #type: ignore
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
        self, subreddit_name: str, search_topics: str, limit: int = 1000
    ) -> list[dict]:
        if not self.reddit:
            logging.error("Não é possível obter posts. Conexão não estabelecida.")
            return []
        try:
            sub_reddit = self.reddit.subreddit(subreddit_name)
            all_data = []
            for post in sub_reddit.search(search_topics, sort="hot", limit=limit):
                print(f"Post encontrado: {post.title}")
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
                    post.comments.replace_more(limit=None)
                except Exception as e:
                    logging.error(
                        f"Erro ao carregar mais comentários para o post {post.id}: {e}"
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
            logging.error(f"Erro ao buscar posts: {e}")
            return []

    def save_in_file(self, file_json, file_name) -> bool:
        try:
            base_path = get_repo_path()
            destiny_file = Path(base_path) / "raw_data_reddit"
            
            destiny_file.mkdir(parents=True, exist_ok=True) # Garante que a pasta exista
            complete_path = destiny_file / file_name
            
            with open(complete_path, "w", encoding="utf-8") as f:
                json.dump(file_json, f, ensure_ascii=False, indent=4)
            print(f"Arquivo '{file_name}' salvo com sucesso em: {complete_path}") # Log atualizado
            return True

        except Exception as e:
            logging.error(f"Ocorreu um erro: {e}")
            return False

def scrape_and_save_task(task_params):
    top, word = task_params
    logging.info(f"[TASK INICIADA] - Subreddit: {top}, Palavra: {word}")
    thread_client = Reddit_Connection()
    if not thread_client.connect_to_api_reddit():
        logging.error(f"[TASK FALHOU] - Conexão: {top} / {word}")
        return f"FALHA (Conexão): {top} / {word}"
    try:
        top_posts = thread_client.get_top_posts_and_comments(top, word)
        if not top_posts:
            logging.warning(
                f"[TASK VAZIA] - Nenhum post encontrado para: {top} / {word}"
            )
            return f"OK (Vazio): {top} / {word}"
        safe_palavra = word.replace(" ", "_")
        filename = f"{top}_{safe_palavra}.json"
        thread_client.save_in_file(top_posts, filename)
        logging.info(f"[TASK CONCLUÍDA] - Salvo: {filename} ({len(top_posts)} posts)")
        return f"OK (Salvo): {filename}"
    except Exception as e:
        logging.error(f"[TASK FALHOU] - Erro durante execução {top} / {word}: {e}")
        return f"FALHA (Erro): {top} / {word}"

@data_loader #type: ignore
def load_reddit_data(*args, **kwargs):
    """
    Este bloco executa o script de scraping completo.
    """
        
    topics = [
        "investimentos", "brasil",  "farialimabets", "empreendedorismo", "MicroEmpresas","devBR",
        "brdev", "Empreendedor", "Liderança", "StartupsAjudaStartups", "Os Fundadores",
        "crescermeunegócio", "capital de risco", "empreendedor avançado", "produtividade", 
        "mídias sociais", "MicroEmpresas", "MeuNegocio"
    ]

    keywords = [
        "ME", "CNPJ", "MEI", "Simples Nacional", "abrir empresa", "imposto MEI", "imposto"
        "imposto ME", "imposto CNPJ", "DAS", "INSS MEI", "governo+empresa", "IRPF",
    ]

    logging.info("Verificando conexão principal antes de iniciar threads...")
    check_client = Reddit_Connection()
    if not check_client.connect_to_api_reddit():
        logging.error("Falha ao conectar. Verificar credenciais. Encerrando.")
        raise Exception("Falha ao conectar. Verificar credenciais.") 

    logging.info("Conexão principal OK. Criando pool de tarefas...")

    tasks = [(top, word) for top in topics for word in keywords]
    logging.info(f"Total de {len(tasks)} tarefas para executar.")
    
    MAX_WORKERS = 5

    logging.info(f"Iniciando {MAX_WORKERS} threads paralelas...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(scrape_and_save_task, tasks))

    logging.info("--- Processamento de todas as tarefas concluído. ---")
    
    for r in results:
        if "FALHA" in r:
            logging.error(f"Resultado final: {r}")
        else:
            logging.info(f"Resultado final: {r}")
            
    base_path = get_repo_path()
    path_para_dados = os.path.join(base_path, "raw_data_reddit")

    logging.info(f"Dados brutos salvos em: {path_para_dados}")
    
    # Isso passa o caminho ABSOLUTO para o próximo bloco
    return {"raw_data_path": path_para_dados}
import pandas as pd
import logging
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import torch
from mage_ai.settings.repo import get_repo_path

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_topic_group_by_words(topic_keywords):

    if not isinstance(topic_keywords, str):
        return "Outros/Geral"
        
    keywords = topic_keywords.lower()
    
    # Uberização (Trabalho por App)
    if any(x in keywords for x in ['uber', '99', 'ifood', 'entregador', 'corrida', 'taxa', 'moto', 'bike', 'plataforma']):
        return "Uberização e Apps"
    
    # Pejotização e Direitos
    elif any(x in keywords for x in ['pj', 'clt', 'férias', 'ferias', 'décimo', 'fgts', 'inss', 'carteira', 'vínculo', 'chefe', 'subordinação', 'horário', 'recrutador']):
        return "Pejotização e Direitos Trabalhistas"
    
    # -Risco Financeiro e Sobrevivência
    elif any(x in keywords for x in ['dívida', 'divida', 'banco', 'empréstimo', 'emprestimo', 'nome sujo', 'serasa', 'falência', 'fome', 'conta', 'sobrevivência', 'pagar', 'dinheiro']):
        return "Risco Financeiro e Sobrevivência"

    # Burocracia MEI
    elif any(x in keywords for x in ['cnpj', 'abrir', 'nota fiscal', 'receita', 'alvará', 'limite', 'desenquadramento', 'formalização']):
        return "Burocracia e Formalização"
    
    # Tributação
    elif any(x in keywords for x in ['das', 'imposto', 'boleto', 'tributo', 'leão']):
        return "Carga Tributária (DAS/Impostos)"
        
    # Empreendedorismo "Real"
    elif any(x in keywords for x in ['investimento', 'marketing', 'cliente', 'vendas', 'lucro', 'estratégia']):
        return "Gestão e Oportunidade"

    # 7. Outros
    else:
        return "Outros/Geral"

@transformer
def generate_topics(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    
    if df.empty:
        logging.warning("Empty DataFrame.")
        return df

    df = df.dropna(subset=['text_clean', 'created_at']).copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df = df.dropna(subset=['created_at'])

    docs = df['text_clean'].tolist()
    timestamps = df['created_at'].tolist()

    seed_topic_list = [
        ["uber", "ifood", "entregador", "app", "corrida"], 
        ["pj", "clt", "férias", "fgts", "chefe", "vínculo"], 
        ["mei", "das", "imposto", "dívida", "boleto"], 
        ["salário", "sobrevivência", "dinheiro", "fome"],
        ["investimento", "lucro", "marketing"]
    ]

    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

    topic_model = BERTopic(
        embedding_model=embedding_model, 
        seed_topic_list=seed_topic_list,
        min_topic_size=15, 
        verbose=True,
        calculate_probabilities=False,
        n_gram_range=(1, 2)
    )

    topics, probs = topic_model.fit_transform(docs)
    
    topic_info = topic_model.get_topic_info()
    
    topic_words_map = {}
    for index, row in topic_info.iterrows():
        try:
            if 'Representation' in row and isinstance(row['Representation'], list):
                words = ", ".join(row['Representation'][:5])
            else:
                words = str(row['Name'])
        except:
            words = str(row['Name'])
            
        topic_words_map[row['Topic']] = words

    df['topic_id'] = topics
    df['topic_keywords'] = df['topic_id'].map(topic_words_map)
    df['topic_group'] = df['topic_keywords'].apply(get_topic_group_by_words)

    topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
    
    topics_over_time['topic_keywords'] = topics_over_time['Topic'].map(topic_words_map)
    topics_over_time['Topic_Group'] = topics_over_time['topic_keywords'].apply(get_topic_group_by_words)
    
    dtm_path = os.path.join(get_repo_path(), "topics_over_time_refined.csv")
    topics_over_time.to_csv(dtm_path, index=False)
    
    info_path = os.path.join(get_repo_path(), "topic_info.csv")
    topic_info.to_csv(info_path, index=False)
    
    return df
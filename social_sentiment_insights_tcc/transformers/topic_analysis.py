import pandas as pd
import logging
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import torch
from mage_ai.settings.repo import get_repo_path

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_topic_group_by_words(topic_keywords):
    if not isinstance(topic_keywords, str):
        return "Outros/Geral"
        
    keywords = topic_keywords.lower()
    
    if any(x in keywords for x in ['uber', '99', 'ifood', 'entregador', 'corrida', 'taxa', 'moto', 'bike', 'plataforma']):
        return "Uberização e Apps"
    
    elif any(x in keywords for x in ['pj', 'clt', 'férias', 'ferias', 'décimo', 'fgts', 'inss', 'carteira', 'vínculo', 'chefe', 'subordinação', 'horário', 'recrutador']):
        return "Pejotização e Direitos Trabalhistas"
    
    elif any(x in keywords for x in ['dívida', 'divida', 'banco', 'empréstimo', 'emprestimo', 'nome sujo', 'serasa', 'falência', 'fome', 'conta', 'sobrevivência', 'pagar', 'dinheiro']):
        return "Risco Financeiro e Sobrevivência"

    elif any(x in keywords for x in ['cnpj', 'abrir', 'nota fiscal', 'receita', 'alvará', 'limite', 'desenquadramento', 'formalização']):
        return "Burocracia e Formalização"
    
    elif any(x in keywords for x in ['das', 'imposto', 'boleto', 'tributo', 'leão']):
        return "Carga Tributária (DAS/Impostos)"
        
    elif any(x in keywords for x in ['investimento', 'marketing', 'cliente', 'vendas', 'lucro', 'estratégia']):
        return "Gestão e Oportunidade"

    else:
        return "Outros/Geral"

@transformer # type:ignore
def generate_topics(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    if df.empty:
        logging.warning("DataFrame vazio. Pulando BERTopic.")
        return df

    torch.cuda.empty_cache()

    df = df.dropna(subset=['text_clean', 'created_at']).copy()
    
    # Garantir que a data está correta para o Topics Over Time
    if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df = df.dropna(subset=['created_at'])

    df = df[df['created_at'].dt.year >= 2021].copy()
    logging.info(f"Filtro Pós-Pandemia aplicado. Restaram {len(df)} registros a partir de 2021.")

    docs = df['text_clean'].tolist()
    timestamps = df['created_at'].tolist()

    logging.info("Carregando modelo de Embeddings MPNet...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)

    # Configuração do CountVectorizer (Remoção de Stopwords)
    stop_words_pt = stopwords.words('portuguese')
    custom_stops = ['pra', 'pro', 'q', 'vc', 'tá', 'ta', 'aí', 'lá', 'nao', 'já', 
                    'vai', 'pode', 'fazer', 'ter', 'ser', 'sobre', 'aqui', 'tudo', 'pq', '00h']
    stop_words_pt.extend(custom_stops)
    
    vectorizer_model = CountVectorizer(stop_words=stop_words_pt, ngram_range=(1, 1))

    # Configuração do BERTopic Guiado
    logging.info("Treinando BERTopic...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model, 
        seed_topic_list=[
            ["uber", "ifood", "entregador", "app", "corrida"], 
            ["pj", "clt", "férias", "fgts", "chefe", "vínculo"], 
            ["mei", "das", "imposto", "dívida", "boleto", "falência"]
        ],
        min_topic_size=30,
        nr_topics="auto",
        verbose=True,
        calculate_probabilities=False
    )

    topics, _ = topic_model.fit_transform(docs)
    
    topic_info = topic_model.get_topic_info()
    topic_words_map = {row['Topic']: ", ".join(row['Representation'][:5]) 
                       for _, row in topic_info.iterrows()}

    df['topic_id'] = topics
    df['topic_keywords'] = df['topic_id'].map(topic_words_map)
    df['topic_group'] = df['topic_keywords'].apply(get_topic_group_by_words)

    base_path = get_repo_path() if 'get_repo_path' in globals() else "."

    # Modelagem Dinâmica (DTM) e Geração de Gráficos
    try:
        logging.info("Calculando a evolução dos tópicos no tempo (DTM)...")
        topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=10)
        topics_over_time['topic_keywords'] = topics_over_time['Topic'].map(topic_words_map)
        topics_over_time['Topic_Group'] = topics_over_time['topic_keywords'].apply(get_topic_group_by_words)
        
        dtm_path = os.path.join(base_path, "topics_over_time_refined.csv")
        topics_over_time.to_csv(dtm_path, index=False)
        logging.info(f"DTM salvo em: {dtm_path}")
        
        logging.info("Gerando gráficos visuais dos Tópicos...")
        
        labels_list = []
        for topic in topic_info['Topic']:
            if topic == -1:
                labels_list.append("Ruído / Outliers")
            else:
                keywords_full = topic_words_map.get(topic, "")
                grupo_sociologico = get_topic_group_by_words(keywords_full)
                
                top_words = topic_model.get_topic(topic)
                assinatura = ", ".join([word[0] for word in top_words[:2]])
                
                labels_list.append(f"{grupo_sociologico} ({assinatura})")
                
        topic_model.set_topic_labels(labels_list)
        
        qtd_topicos_validos = len(topic_info[topic_info['Topic'] != -1])
        top_n = min(6, qtd_topicos_validos)
        
        if top_n > 0:
            fig_bar = topic_model.visualize_barchart(
                top_n_topics=top_n, 
                title="Principais Narrativas do Mercado de Trabalho (Termos Mais Relevantes)",
                custom_labels=True
            )
            
            for annotation in fig_bar.layout.annotations:
                texto_antigo = annotation.text
                if texto_antigo and "Topic " in texto_antigo:
                    try:
                        num_topico = int(texto_antigo.replace("Topic ", "").strip())
                        keywords_full = topic_words_map.get(num_topico, "")
                        grupo_sociologico = get_topic_group_by_words(keywords_full)
                        
                        top_words = topic_model.get_topic(num_topico)
                        assinatura = ", ".join([word[0] for word in top_words[:2]])
                        
                        novo_nome = f"{grupo_sociologico}<br>({assinatura})" 
                        annotation.text = f"<b>{novo_nome}</b>" 
                    except ValueError:
                        pass
            
            fig_time = topic_model.visualize_topics_over_time(
                topics_over_time, 
                top_n_topics=top_n, 
                normalize_frequency=True,
                custom_labels=True,
                title="Evolução Temporal do Discurso sobre Precarização e Empreendedorismo"
            )
            
            barchart_path = os.path.join(base_path, "bertopic_barchart_tcc.png")
            timechart_path = os.path.join(base_path, "bertopic_over_time_tcc.png")
            
            fig_bar.write_image(barchart_path, width=1300, height=800, scale=2)
            fig_time.write_image(timechart_path, width=1300, height=600, scale=2)
            
            logging.info(f"Gráficos do BERTopic salvos em: {barchart_path} e {timechart_path}")
        else:
            logging.warning("Não há tópicos válidos suficientes para gerar gráficos.")

    except Exception as e:
        logging.error(f"Erro no Topics over Time ou na geração de gráficos: {e}")

    info_path = os.path.join(base_path, "topic_info.csv")
    topic_info.to_csv(info_path, index=False)
    
    del topic_model
    del embedding_model
    torch.cuda.empty_cache() 

    return df
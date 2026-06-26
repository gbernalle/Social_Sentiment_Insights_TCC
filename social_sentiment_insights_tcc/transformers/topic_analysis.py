import pandas as pd
import logging
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from mage_ai.settings.repo import get_repo_path

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

device = "cuda" if torch.cuda.is_available() else "cpu"

MAPA_CORES_TCC = {
    "Pejotização e Precarização dos Vínculos Empregatícios": "#2E75B6", # Azul
    "Uberização e Trabalho em Plataformas Digitais": "#E07B2A",         # Laranja
    "Vulnerabilidade Financeira e Risco Social": "#70AD47",             # Verde
    "Formalização Jurídica e Barreiras Burocráticas": "#7030A0",        # Roxo
    "Carga Tributária e Obrigações Fiscais do MEI": "#C00000",          # Vermelho
    "Gestão Empresarial e Estratégias de Mercado": "#00B0F0",           # Azul Claro
    "Discursos Gerais sobre Trabalho Informal": "#808080",              # Cinza
    "Ruído e Não Classificados": "#D3D3D3"                              # Cinza Claro
}

def get_topic_group_by_words(topic_keywords):
    if not isinstance(topic_keywords, str):
        return "Não Classificado"

    keywords = topic_keywords.lower()

    if any(x in keywords for x in ['uber', '99', 'ifood', 'entregador', 'corrida', 'taxa', 'moto', 'bike', 'plataforma']):
        return "Uberização e Trabalho em Plataformas Digitais"

    elif any(x in keywords for x in ['pj', 'clt', 'férias', 'ferias', 'décimo', 'fgts', 'inss', 'carteira', 'vínculo', 'chefe', 'subordinação', 'horário', 'recrutador']):
        return "Pejotização e Precarização dos Vínculos Empregatícios"

    elif any(x in keywords for x in ['dívida', 'divida', 'banco', 'empréstimo', 'emprestimo', 'nome sujo', 'serasa', 'falência', 'fome', 'conta', 'sobrevivência', 'pagar', 'dinheiro']):
        return "Vulnerabilidade Financeira e Risco Social"

    elif any(x in keywords for x in ['cnpj', 'abrir', 'nota fiscal', 'receita', 'alvará', 'limite', 'desenquadramento', 'formalização']):
        return "Formalização Jurídica e Barreiras Burocráticas"

    elif any(x in keywords for x in ['das', 'imposto', 'boleto', 'tributo', 'leão']):
        return "Carga Tributária e Obrigações Fiscais do MEI"

    elif any(x in keywords for x in ['investimento', 'marketing', 'cliente', 'vendas', 'lucro', 'estratégia']):
        return "Gestão Empresarial e Estratégias de Mercado"

    else:
        return "Discursos Gerais sobre Trabalho Informal"


def build_vertical_barchart(topic_model, topic_info, labels_list, valid_topics_chunk):
    n = len(valid_topics_chunk)

    if n == 0:
        return None

    fig = make_subplots(
        rows=1, cols=n, horizontal_spacing=0.08,
    )

    for col_idx, topic_id in enumerate(valid_topics_chunk, start=1):
        top_words = topic_model.get_topic(topic_id)
        if not top_words: continue

        words  = [w[0] for w in top_words[:8]]
        scores = [round(w[1], 4) for w in top_words[:8]]
        pairs  = sorted(zip(scores, words), key=lambda x: x[0])
        scores = [p[0] for p in pairs]
        words  = [p[1] for p in pairs]
        
        idx_geral = list(topic_info['Topic']).index(topic_id)
        
        grupo_nome = labels_list[idx_geral]
        color = MAPA_CORES_TCC.get(grupo_nome, "#333333")

        fig.add_trace(
            go.Bar(
                x=words, y=scores, orientation="v", marker_color=color, showlegend=False,
                text=[f"{s:.3f}" for s in scores], 
                textposition="outside", 
                textfont=dict(size=22),
            ), row=1, col=col_idx,
        )

        yaxis_key = "yaxis" if col_idx == 1 else f"yaxis{col_idx}"
        fig.update_layout(**{yaxis_key: dict(
            title="Score de Relevância" if col_idx == 1 else "", 
            showgrid=True, 
            gridcolor="#E5E5E5", 
            title_font=dict(size=22),
            tickfont=dict(size=20) 
        )}) 
        
        xaxis_key = "xaxis" if col_idx == 1 else f"xaxis{col_idx}"
        fig.update_layout(**{xaxis_key: dict(
            tickangle=-35, 
            tickfont=dict(size=22)
        )}) 

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", 
        margin=dict(l=100, r=40, t=60, b=150),
        font=dict(family="Times New Roman, serif", size=24), height=650, width=1100, 
    )

    return fig

@transformer  # type:ignore
def generate_topics(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    if df.empty:
        logging.warning("DataFrame vazio. Pulando BERTopic.")
        return df

    torch.cuda.empty_cache()

    df = df.dropna(subset=['text_clean', 'created_at']).copy()

    if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df = df.dropna(subset=['created_at'])

    df = df[df['created_at'].dt.year >= 2018].copy()
    
    total_docs = len(df)
    logging.info(f"Filtro temporal aplicado. Restaram {total_docs} registros a partir de 2018.")

    docs = df['text_clean'].tolist()
    timestamps = df['created_at'].tolist()

    embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)

    stop_words_pt  = stopwords.words('portuguese')
    custom_stops   = ['pra', 'pro', 'q', 'vc', 'tá', 'ta', 'aí', 'lá', 'nao', 'já', 'vai', 'pode', 'fazer', 'ter', 'ser', 'sobre', 'aqui', 'tudo', 'pq', '00h']
    stop_words_pt.extend(custom_stops)
    vectorizer_model = CountVectorizer(stop_words=stop_words_pt, ngram_range=(1, 1))

    logging.info("Treinando BERTopic com lista de sementes temáticas...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        seed_topic_list=[
            ["uber", "ifood", "motoboy", "corrida", "motorista", "app"],
            ["pj", "clt", "férias", "fgts", "vínculo", "direitos"],
            ["mei", "das", "imposto", "dívida", "boleto", "falência"],
        ],
        min_topic_size=50,
        nr_topics=3,
        verbose=True,
        calculate_probabilities=False,
    )

    topics, probs = topic_model.fit_transform(docs)

    topic_info = topic_model.get_topic_info()
    topic_words_map = {row['Topic']: ", ".join(row['Representation'][:5]) for _, row in topic_info.iterrows()}

    df['topic_id']       = topics
    df['topic_keywords'] = df['topic_id'].map(topic_words_map)
    df['topic_group']    = df['topic_keywords'].apply(get_topic_group_by_words)

    df['topic_confidence'] = probs if probs is not None else 0.0
    
    outlier_count = len(df[df['topic_id'] == -1])
    outlier_percentage = (outlier_count / total_docs) * 100
    valid_docs_count = total_docs - outlier_count

    df_validos = df[df['topic_id'] != -1]
    media_confianca = df_validos['topic_confidence'].mean() * 100

    logging.info(f"Total de Documentos Processados: {total_docs}")
    logging.info(f"Total de Tópicos Gerados (excluindo ruído): {len(topic_info) - 1}")
    
    logging.info(f"Documentos Classificados: {valid_docs_count} ({(valid_docs_count/total_docs)*100:.1f}%)")
    logging.info(f"Documentos como Ruído/Outliers (Tópico -1): {outlier_count} ({outlier_percentage:.1f}%)")
    logging.info(f"Confiança Média do Agrupamento (Densidade HDBSCAN): {media_confianca:.2f}%")
    
    for _, row in topic_info.iterrows():
        if row['Topic'] != -1:
            grupo = get_topic_group_by_words(", ".join(row['Representation']))
            logging.info(f"Tópico {row['Topic']}: {row['Count']} docs | {grupo}")
    print("="*50 + "\n")

    base_path = get_repo_path() if 'get_repo_path' in globals() else "."

    try:
        topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=10)
        topics_over_time['topic_keywords'] = topics_over_time['Topic'].map(topic_words_map)
        topics_over_time['Topic_Group']    = topics_over_time['topic_keywords'].apply(get_topic_group_by_words)

        totals_by_timestamp = topics_over_time.groupby('Timestamp')['Frequency'].sum()
        topics_over_time['Frequency_Normalized'] = topics_over_time.apply(lambda row: row['Frequency'] / totals_by_timestamp[row['Timestamp']], axis=1)
        topics_over_time['Timestamp'] = pd.to_datetime(topics_over_time['Timestamp']).dt.strftime('%Y-%m-%d')

        dtm_path = os.path.join(base_path, "topics_over_time_refined.csv")
        topics_over_time.to_csv(dtm_path, index=False)
        logging.info(f"DTM salvo em: {dtm_path}")

        labels_list = []
        for topic in topic_info['Topic']:
            if topic == -1:
                labels_list.append("Ruído e Não Classificados")
            else:
                keywords_full    = topic_words_map.get(topic, "")
                grupo_academico  = get_topic_group_by_words(keywords_full)
                labels_list.append(grupo_academico)

        topic_model.set_topic_labels(labels_list)
        qtd_topicos_validos = len(topic_info[topic_info['Topic'] != -1])
        top_n = min(6, qtd_topicos_validos)

        if top_n > 0:
            fig_time = topic_model.visualize_topics_over_time(
                topics_over_time, top_n_topics=top_n, normalize_frequency=True, custom_labels=True,
            )
            
            for trace in fig_time.data:
                for label_key, color_hex in MAPA_CORES_TCC.items():
                    if label_key in trace.name:
                        trace.line.color = color_hex
                        break
            
            fig_time.update_layout(
                plot_bgcolor="white", 
                paper_bgcolor="white", 
                font=dict(family="Times New Roman, serif", size=16),
                
                xaxis=dict(
                    title="Ano", 
                    showgrid=True, 
                    gridcolor="#EBEBEB", 
                    tickfont=dict(size=16), 
                    title_font=dict(size=18, color="#333333")
                ), 
                
                yaxis=dict(
                    title="Frequência Normalizada", 
                    showgrid=True, 
                    gridcolor="#EBEBEB", 
                    tickfont=dict(size=16), 
                    title_font=dict(size=18, color="#333333")
                ),
                
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.97, 
                    xanchor="left",
                    x=0.02, 
                    title=None,
                    font=dict(size=16, family="Times New Roman, serif"),
                    bgcolor="rgba(255, 255, 255, 0.85)",
                    bordercolor="#E5E5E5",
                    borderwidth=1
                ),
                
                margin=dict(t=60, b=80, l=80, r=40) 
            )
            
            fig_time.layout.title = None 
            
            timechart_path = os.path.join(base_path, "bertopic_over_time_tcc.png")
            fig_time.write_image(timechart_path, width=1300, height=700, scale=2)
            
            valid_topics_all = [row['Topic'] for _, row in topic_info.iterrows() if row['Topic'] != -1][:top_n]
            
            if len(valid_topics_all) > 0:
                meio = (len(valid_topics_all) + 1) // 2 
                parte_1 = valid_topics_all[:meio]
                parte_2 = valid_topics_all[meio:]

                fig_bar_1 = build_vertical_barchart(topic_model, topic_info, labels_list, parte_1)
                if fig_bar_1:
                    barchart_path_1 = os.path.join(base_path, "bertopic_barchart_parte1_tcc.png")
                    fig_bar_1.write_image(barchart_path_1, width=1100, height=650, scale=2)

                if len(parte_2) > 0:
                    fig_bar_2 = build_vertical_barchart(topic_model, topic_info, labels_list, parte_2)
                    if fig_bar_2:
                        barchart_path_2 = os.path.join(base_path, "bertopic_barchart_parte2_tcc.png")
                        fig_bar_2.write_image(barchart_path_2, width=1100, height=650, scale=2)


            if qtd_topicos_validos > 2:
                try:
                    fig_intertopic = topic_model.visualize_topics(custom_labels=True)
                    fig_intertopic.update_layout(
                        plot_bgcolor="white", paper_bgcolor="white", 
                        font=dict(family="Times New Roman, serif", size=14)
                    )
                    intertopic_path = os.path.join(base_path, "bertopic_intertopic_map_tcc.png")
                    fig_intertopic.write_image(intertopic_path, width=900, height=700, scale=2)
                except Exception as e_umap:
                    logging.warning(f"Não foi possível gerar o Mapa Intertópico pelo UMAP (normal em bases menores): {e_umap}")
            else:
                logging.warning(f"Gráfico Intertópico ignorado: Apenas {qtd_topicos_validos} tópico(s) válido(s) encontrados. O UMAP requer no mínimo 3.")

            logging.info("Gerando Gráfico de Distribuição de Volume dos Tópicos...")
            df_valid_topics = topic_info[topic_info['Topic'] != -1].copy()
            df_valid_topics['Label'] = df_valid_topics['Topic'].apply(lambda x: labels_list[list(topic_info['Topic']).index(x)])
            
            fig_pie = px.pie(
                df_valid_topics, values='Count', names='Label', hole=0.45,
                color='Label', 
                color_discrete_map=MAPA_CORES_TCC 
            )
            
            fig_pie.update_traces(
                textposition='inside', 
                texttemplate='<b>%{label}</b><br><b>%{percent}</b>', 
                showlegend=False,
                textfont=dict(color='white', size=22, family="Times New Roman, serif"), 
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            
            fig_pie.update_layout(
                font=dict(family="Times New Roman, serif", size=16),
                title=dict(font=dict(size=22), x=0.5, xanchor="center"), 
                margin=dict(t=80, b=40, l=40, r=40)
            )
            
            piechart_path = os.path.join(base_path, "bertopic_distribution_pie_tcc.png")
            fig_pie.write_image(piechart_path, width=1000, height=800, scale=2)
        else:
            logging.warning("Nenhum tópico válido encontrado para geração de gráficos.")

    except Exception as e:
        logging.error(f"Erro no cálculo DTM ou na geração de gráficos: {e}", exc_info=True)

    info_path = os.path.join(base_path, "topic_info.csv")
    topic_info.to_csv(info_path, index=False)

    del topic_model
    del embedding_model
    torch.cuda.empty_cache()

    return df
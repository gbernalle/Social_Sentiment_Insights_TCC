import pandas as pd
import logging
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
from mage_ai.settings.repo import get_repo_path

try:
    from transformers import pipeline, AutoTokenizer, BatchEncoding
except ImportError:
    logging.error("Libraries 'transformers', 'torch' or 'AutoTokenizer' not found. Please install them.")
    raise

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

device_id = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if device_id == 0 else "CPU"
logging.info(f"Zero-Shot NLP running on: {device_name}")

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
MODEL_MAX_LENGTH = 512 

classifier = None
tokenizer = None

def load_models():
    """Loads the model into VRAM if not already loaded."""
    global classifier, tokenizer
    if classifier is None:
        logging.info(f"Loading Zero-Shot model on {device_name}...")
        try:
            classifier = pipeline(
                "zero-shot-classification",
                model=MODEL_NAME,
                hypothesis_template="Este texto é sobre {}.",
                device=device_id 
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            logging.info("Models loaded into VRAM.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise e
    return classifier, tokenizer

candidate_labels = [
    "substituição de vagas CLT por PJ, subordinação disfarçada, cumprimento de horário, ou debates e relatos sobre empresas que contratam pessoa jurídica como se fosse funcionário",
    "perda de direitos trabalhistas, desproteção do trabalhador moderno, falta de segurança social, ou ausência de benefícios como férias e décimo terceiro",
    "burocracia de ter um CNPJ, custos de manutenção, problemas com impostos, dívidas, DAS atrasado, malha fina, ou o peso financeiro e tributário de ser MEI",
    "vulnerabilidade financeira, baixa renda, dificuldade de arcar com custos, trabalho por necessidade, bicos, entregas, ou a realidade de quem tem pouco dinheiro no modelo MEI e aplicativos",
    "estratégias de negócios, crescimento de empresas, captação de clientes, investimentos, ou visões e opiniões positivas sobre a liberdade e o lucro do empreendedorismo",
    "assuntos genéricos, discussões aleatórias, desabafos pessoais, brigas, ou dúvidas não relacionadas a mercado de trabalho, pejotização ou economia"
]

labels_subject = [
    "Pejotização e Subordinação",
    "Precarização de Direitos",
    "Risco Fiscal e Dívida",
    "Sobrevivência e Necessidade", 
    "Gestão e Oportunidade",
    "Lixo/Off-Topic"
]

label_map = dict(zip(candidate_labels, labels_subject))

@transformer #type: ignore
def filter_by_context(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    
    if data.empty:
        logging.warning("Previous block returned empty DataFrame. Skipping.")
        return pd.DataFrame()

    total_linhas_iniciais = len(data)

    classifier, tokenizer = load_models()
    
    logging.info(f"Calculando limites de tokens para Zero-Shot...")
    
    reported_max_len = getattr(tokenizer, 'model_max_length', 512)
    if reported_max_len > 512:
        MODEL_MAX_LENGTH = 512
        logging.warning(f"Tokenizer reported max_len={reported_max_len}. Forcing manual limit to 512 to avoid RuntimeError.")
    else:
        MODEL_MAX_LENGTH = reported_max_len
    
    all_label_tokens = [len(tokenizer(label, add_special_tokens=False)['input_ids']) for label in candidate_labels]
    max_label_len = max(all_label_tokens) if all_label_tokens else 0
    
    HYPOTHESIS_TEMPLATE_BUFFER = 12 
    SPECIAL_TOKENS_BUFFER = 3       
    SAFETY_MARGIN = 5               
    
    reserved_tokens = max_label_len + HYPOTHESIS_TEMPLATE_BUFFER + SPECIAL_TOKENS_BUFFER + SAFETY_MARGIN
    max_text_tokens_allowed = MODEL_MAX_LENGTH - reserved_tokens
    
    if max_text_tokens_allowed < 10:
        raise ValueError("Labels too long for this model context.")

    logging.info(f"Hard Limit: {MODEL_MAX_LENGTH} | Reserved: {reserved_tokens} | Max Input Text: {max_text_tokens_allowed} tokens")

    def truncate_text_by_tokens(text: str) -> str:
        if not text or pd.isna(text):
            return ""
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_text_tokens_allowed:
            return text
            
        truncated_ids = tokens[:max_text_tokens_allowed]
        
        return tokenizer.decode(truncated_ids, skip_special_tokens=True)

    try:
        texts_raw = data['text_clean'].fillna('').astype(str).tolist()
        texts_to_classify = [truncate_text_by_tokens(t) for t in texts_raw]

        if len(texts_to_classify) > 0:
            len_first = len(tokenizer.encode(texts_to_classify[0], add_special_tokens=False))
            logging.info(f"Sample truncated length (tokens): {len_first} (Limit: {max_text_tokens_allowed})")

        results = classifier(
            texts_to_classify, 
            candidate_labels, 
            multi_label=False, 
            batch_size=32,
            truncation=True
        )
        
        df_results = pd.DataFrame(results)
        
        data['category_raw'] = df_results['labels'].str[0].values
        data['category_score'] = df_results['scores'].str[0].values
        
        if 'label_map' in globals():
            data['category_tcc'] = data['category_raw'].map(label_map)
        else:
             data['category_tcc'] = data['category_raw']
             
        thresholds_to_test = [0.0, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]
        retention_rates = []
        average_confidences = []
        metrics_report = {}
        
        df_valid = data[data['category_tcc'] != 'Lixo/Off-Topic']
        total_valid_initial = len(df_valid)
        
        base_path = get_repo_path() if 'get_repo_path' in globals() else "."
        
        if total_valid_initial > 0:
            for t in thresholds_to_test:
                df_aprovados = df_valid[df_valid['category_score'] >= t]
                taxa = (len(df_aprovados) / total_valid_initial) * 100
                retention_rates.append(taxa)
                
                if len(df_aprovados) > 0:
                    media = df_aprovados['category_score'].mean() * 100
                else:
                    media = 0
                average_confidences.append(media)

                if t == 0.55:
                    metrics_report['taxa_retencao_55'] = taxa
                    metrics_report['media_confianca_55'] = media
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            sns.set_theme(style="whitegrid")
            
            color1 = '#4C72B0'
            ax1.set_xlabel('Nível Mínimo de Confiança (Threshold)', fontsize=12)
            ax1.set_ylabel('Taxa de Retenção de Dados (%)', color=color1, fontsize=12)
            linha1 = ax1.plot(thresholds_to_test, retention_rates, marker='o', linewidth=2.5, color=color1, label='Volume Retido (%)')
            ax1.tick_params(axis='y', labelcolor=color1)
            
            ax2 = ax1.twinx()  
            color2 = '#55A868'
            ax2.set_ylabel('Média de Confiança do Dataset (%)', color=color2, fontsize=12)
            linha2 = ax2.plot(thresholds_to_test, average_confidences, marker='s', linewidth=2.5, color=color2, linestyle='-.', label='Média de Confiança (%)')
            ax2.tick_params(axis='y', labelcolor=color2)
            
            linha_corte = ax1.axvline(x=0.55, color='red', linestyle='--', label='Corte Escolhido (0.55)')
            
            linhas = linha1 + linha2 + [linha_corte]
            labels = [l.get_label() for l in linhas]
            ax1.legend(linhas, labels, loc='center right')
            
            plot_path = os.path.join(base_path, "grafico_sensibilidade_media_tcc.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close() 
            
            plt.figure(figsize=(12, 7))
            sns.set_theme(style="whitegrid")
            
            ordem_categorias = df_valid.groupby('category_tcc')['category_score'].median().sort_values(ascending=False).index
            
            sns.boxplot(
                data=df_valid, 
                y='category_tcc', 
                x='category_score', 
                order=ordem_categorias,
                palette="Blues_r",
                showmeans=True,    
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"}
            )

            sns.despine(top=True, right=True, left=True, bottom=True)
            
            plt.axvline(x=0.55, color='red', linestyle='--', linewidth=2.5, label='Corte Escolhido (0.55)')
            
            plt.xlabel('Score de Confiança (0.0 a 1.0)', fontsize=12)
            plt.ylabel('', fontsize=12) 
            plt.legend(loc='lower right')
            
            plt.tight_layout() 
            
            plot_box_path = os.path.join(base_path, "grafico_distribuicao_categorias_tcc.png")
            plt.savefig(plot_box_path, dpi=300, bbox_inches='tight')
            plt.close()     
        else:
            logging.warning("Não há dados válidos suficientes para gerar os gráficos base.")
             
        THRESHOLD = 0.55
         
        mask_low_confidence = data['category_score'] < THRESHOLD
        data.loc[mask_low_confidence, 'category_tcc'] = 'Descartado - Baixa Confiança'
        
        descartados_qtd = mask_low_confidence.sum()
        
        data_final = data[~data['category_tcc'].isin(['Lixo/Off-Topic', 'Descartado - Baixa Confiança'])].copy()
        total_linhas_finais = len(data_final)
        
    
        print(f"Total de registros recebidos do RegEx: {total_linhas_iniciais}")
        print(f"Registros classificados como 'Lixo/Off-Topic': {total_linhas_iniciais - total_valid_initial}")

        print(f" Média de Confiança Global Pós-Corte: {metrics_report.get('media_confianca_55', 0):.2f}%")
        print(f" Taxa de Retenção de Dados Úteis: {metrics_report.get('taxa_retencao_55', 0):.2f}%")
        print(f" Registros rejeitados (< {THRESHOLD}): {descartados_qtd}")
       
        print(f"REGISTROS FINAIS VALIDADOS (Aprovados): {total_linhas_finais}")
        print(f"Perda no Funil NLP: {total_linhas_iniciais - total_linhas_finais} registros.")
         
        colunas_esperadas = [
            'das_context', 'tax_context', 'pj_context', 
            'mei_context', 'uberizacao_context', 'precarious_context'
        ] 
        colunas_contexto = [col for col in colunas_esperadas if col in data_final.columns]
        
        if len(colunas_contexto) >= 3 and len(data_final) > 0:
            frequencia = data_final[colunas_contexto].mean() * 100
            
            nomes_bonitos = {
                'das_context': 'Guia DAS',
                'tax_context': 'Impostos/Tributos',
                'pj_context': 'Pejotização (PJ)',
                'mei_context': 'MEI',
                'uberizacao_context': 'Uberização e Apps',
                'precarious_context': 'Precarização do Trabalho'
            }
            labels_radar = [nomes_bonitos.get(c, c) for c in colunas_contexto]
            
            df_radar = pd.DataFrame({
                'Frequencia (%)': frequencia.values,
                'Contexto': labels_radar
            })
            
            df_radar = pd.concat([df_radar, df_radar.iloc[[0]]], ignore_index=True)
            
            fig_radar = px.line_polar(
                df_radar, 
                r='Frequencia (%)', 
                theta='Contexto', 
                line_close=True,
                title="Composição da Base de Dados Filtrada (Frequência de Contextos)",
                template="plotly_white"
            )
            
            fig_radar.update_traces(fill='toself', line_color='#4C72B0') 
            
            plot_radar_path = os.path.join(base_path, "radar_contextos_tcc.png")
            fig_radar.write_image(plot_radar_path, width=1100, height=800, scale=2)
        else:
            logging.warning(f"Apenas {len(colunas_contexto)} colunas encontradas. O radar precisa de no mínimo 3 para fechar a teia.")
        
        if 'category_raw' in data_final.columns:
            data_final = data_final.drop(columns=['category_raw'])

        cache_path = os.path.join(base_path, "cache_semantic.parquet")
        data_final.to_parquet(cache_path)
        logging.info(f"Checkpoint saved at: {cache_path}")

        return data_final

    except RuntimeError as re:
        logging.error(f"RuntimeError (likely CUDA OOM or Shape Mismatch): {re}")
        raise re
    except Exception as e:
        logging.error(f"Generic Error during Zero-Shot: {str(e)}")
        raise e
# Social Sensing Pipeline: Labor Precarization in Post-Pandemic Brazil

> End-to-end ETL pipeline for sentiment analysis, zero-shot classification, and dynamic topic modeling on micro-entrepreneurship and labor precarization — integrating unstructured social data (Reddit) with macroeconomic time series (IBGE, BCB, TST).

---

## Table of Contents

1. [Context and Problem](#1-context-and-problem)
2. [Solution Architecture](#2-solution-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Repository Structure](#4-repository-structure)
5. [Data Pipelines](#5-data-pipelines)
6. [NLP Pipeline](#6-nlp-pipeline)
7. [Design Decisions and Trade-offs](#7-design-decisions-and-trade-offs)
8. [Prerequisites and Configuration](#8-prerequisites-and-configuration)
9. [How to Run](#9-how-to-run)
10. [Outputs and Artifacts](#10-outputs-and-artifacts)
11. [References](#11-references)

---

## 1. Context and Problem

The Brazilian post-pandemic labor market presents a quantitative paradox: while the unemployment rate dropped to **5.4%** (its lowest level since 2012), the informality rate remains at **38.1%** of the employed population — roughly **40 million workers** without any social protection. At the same time, Brazil's Superior Labor Court (TST) recorded a **57% increase** in lawsuits seeking formal employment bond recognition over the last five years.

Traditional macroeconomic indicators capture the *intensity* of this phenomenon but fail to capture the *subjectivity* of the workforce — whether the growth of the MEI (Individual Micro-Entrepreneur) regime reflects genuine autonomy or a survival strategy driven by the scarcity of formal job alternatives.

**Central hypothesis:** Organic discourse on Brazilian social networks functions as an anticipatory *social sensor*, and the deterioration of macroeconomic indicators (inflation, unemployment) precedes or accompanies spikes in vulnerability, indebtedness, and precarization narratives within MEI/PJ discussions.

---

## 2. Solution Architecture

The project follows **Modern Data Stack** principles and is implemented as **two independent Mage.ai pipelines** that converge at the Data Warehouse layer, enabling isolated development, testing, and scheduling of each domain.

```
┌──────────────────────────────┐   ┌──────────────────────────────────────────┐
│   SOURCE: Reddit API (PRAW)  │   │  SOURCES: SIDRA/IBGE · BCB/SGS · TST     │
└──────────────┬───────────────┘   └───────────────────────┬──────────────────┘
               │                                           │
               ▼                                           ▼
┌──────────────────────────────┐   ┌──────────────────────────────────────────┐
│  PIPELINE 1: reddit_ingestion│   │  PIPELINE 2: gov_data_pipeline           │
│  (Mage.ai DAG — 7 blocks)    │   │  (Mage.ai DAG — 9 blocks)               │
│                              │   │                                          │
│  get_reddit_data             │   │  get_desocupacao_ibge ──► export_desoc.  │
│       │                      │   │  get_processos_tst ──► search_tst        │
│  transform_raw_data          │   │                           │              │
│       │  (Contextual Regex)  │   │                      export_processos    │
│  semantic_cleaning           │   │  get_ipca_metrics ──► export_ipca        │
│       │  (Zero-Shot NLP)     │   │  get_dividas_bc   ──► export_divida      │
│  sentiment_analysis          │   │                                          │
│       │  (XLM-RoBERTa)       │   └──────────────────────────────────────────┘
│  topic_analysis              │
│       │  (BERTopic + DTM)    │
│       ├──► export_topics_metrics
│       └──► export_to_bigquery│
└──────────────────────────────┘
               │                                           │
               └───────────────────┬───────────────────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │  Google BigQuery          │
                    │  (Data Warehouse)         │
                    │  dataset: First_Test_     │
                    │  Sentiment_Analysis       │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  Google Looker Studio     │
                    │  (Analytical Dashboard)   │
                    └──────────────────────────┘
```

---

## 3. Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Orchestration | **Mage.ai** | Pipeline-as-code with natively inspectable intermediate outputs; superior modularity over Airflow for ML-heavy workflows |
| Containerization | **Docker + CUDA** | Dev/prod environment parity; dependency isolation for GPU-dependent libraries (PyTorch/CUDA) |
| Social Ingestion | **PRAW (Reddit API)** | Official wrapper with `ThreadPoolExecutor` support for concurrent, rate-limit-aware collection |
| Structured Ingestion | **sidrapy / requests** | Direct integration with SIDRA (IBGE) and SGS (Central Bank) APIs |
| NLP — Zero-Shot | **mDeBERTa-v3-base-mnli-xnli** | Multilingual SOTA model for XNLI; eliminates the need for a manually labeled training dataset |
| NLP — Sentiment | **XLM-RoBERTa (cardiffnlp)** | Trained on multilingual social media data; robust to slang, abbreviations, and informal Portuguese |
| NLP — Topic Modeling | **BERTopic + MPNet** | Outperforms LDA on short texts; supports Guided Topic Modeling and Dynamic Topic Modeling |
| Data Warehouse | **Google BigQuery** | Serverless scalability; native integration with Looker Studio |
| Visualization | **Google Looker Studio** | Direct BigQuery consumption without an intermediate serving layer |
| GPU Acceleration | **PyTorch + CUDA** | GPU inference reduces batch processing time from hours to minutes |

---

## 4. Repository Structure

```
.
├── data_loaders/
│   ├── get_reddit_data.py          # Multithreaded collection via PRAW
│   ├── get_ipca_metrics.py         # 12-month cumulative IPCA (SIDRA/IBGE)
│   ├── get_desocupacao_ibge.py     # Unemployment rate (PNAD Contínua)
│   ├── get_dividas_bc.py           # Household debt & default rate (BCB)
│   └── get_processos_tst.py        # TST new cases (Excel, one tab per year)
│
├── transformers/
│   ├── transform_raw_data.py       # JSON parse → DataFrame + Contextual Regex
│   ├── search_tst_processos.py     # Employment bond case filter & aggregation
│   ├── semantic_cleaning.py        # Zero-Shot Classification (mDeBERTa-v3)
│   ├── sentiment_analysis.py       # Sentiment analysis (XLM-RoBERTa)
│   └── topic_analysis.py           # Guided BERTopic + DTM + chart generation
│
├── data_exporters/
│   ├── export_to_bigquery.py       # Main enriched table (vFinal)
│   ├── export_desocupacao_ibge.py
│   ├── export_divida.py
│   ├── export_ipca_metrics.py
│   ├── export_processos_tst.py
│   ├── export_semantic_clean_55ac.py
│   └── export_topics_metrics.py    # DTM time series export
│
├── dados_gov/
│   └── dados_tst/                  # TST historical spreadsheets (New Cases)
│
├── raw_data_reddit/                # Raw JSONs (generated at runtime, gitignored)
│
├── io_config.yaml                  # BigQuery credentials (Mage.ai config)
├── .env                            # Environment variables (not versioned)
├── requirements.txt
└── README.md
```

---

## 5. Data Pipelines

The project is composed of two independent Mage.ai pipelines. They share the same BigQuery dataset as a convergence point but have completely separate DAG definitions, block dependencies, and scheduling concerns.

### Pipeline 1 — `reddit_ingestion` (7 blocks)

Responsible for the full social data lifecycle: collection, cleaning, NLP enrichment, and export.

**Block dependency chain:**

```
get_reddit_data
      │
transform_raw_data        ← JSON parse + Contextual Regex + deduplication
      │
semantic_cleaning          ← Zero-Shot Classification (mDeBERTa-v3)
      │
sentiment_analysis         ← Sentiment scoring (XLM-RoBERTa)
      │
topic_analysis             ← Guided BERTopic + DTM + chart generation
      ├──► export_topics_metrics   → BigQuery: historico_topicos_mei
      └──► export_to_bigquery      → BigQuery: tabela_completa_vFinal
```

#### 5.1 Social Ingestion (`get_reddit_data.py`)

The collector uses `ThreadPoolExecutor` with up to 3 simultaneous workers to cross **19 subreddits** against **17 keywords** (323 task combinations), each task with random jitter (2–5s) to respect API rate limits without explicit sleep blocking.

Monitored subreddits include: `r/brdev`, `r/antitrampo`, `r/investimentos`, `r/empreendedorismo`, `r/farialimabets`, and others. Keywords cover the MEI/PJ ecosystem: `"MEI"`, `"DAS"`, `"pejotização"`, `"Uberização"`, `"CNPJ"`, `"Simples Nacional"`.

Raw data is persisted as individual JSONs per task combination (`subreddit_keyword.json`), preserving nested posts and comments for full auditability.

---

### Pipeline 2 — `gov_data_pipeline` (9 blocks)

Responsible for ingesting official macroeconomic and judicial time series from government APIs and structured files, normalizing them, and loading into BigQuery. All loaders run independently with no shared state.

**Block dependency chains:**

```
get_desocupacao_ibge ──────────────────────────► export_desocupacao_ibge
get_ipca_metrics     ──────────────────────────► export_ipca_metrics
get_dividas_bc       ──────────────────────────► export_divida
get_processos_tst ──► search_tst_processos ────► export_processos_tst
```

#### 5.2 Government Data Ingestion

| Block | Source | Series | Granularity |
|---|---|---|---|
| `get_ipca_metrics.py` | SIDRA/IBGE (Table 1737) | 12-month cumulative inflation | Monthly |
| `get_desocupacao_ibge.py` | SIDRA/IBGE (Table 6381) | Unemployment rate | Monthly |
| `get_dividas_bc.py` | BCB/SGS (Series 29037, 21084) | Household debt & individual default | Monthly |
| `get_processos_tst.py` | TST (Excel, one tab per year) | New cases by subject/year | Annual |

All series are filtered from **January 2018** onward to ensure consistent historical coverage across pre- and post-pandemic periods. The TST data requires an additional transformer (`search_tst_processos`) to filter cases related to employment bond recognition and aggregate by year before export.

---

### 5.3 Transformation — Contextual Regex (`transform_raw_data.py`)

The transformation layer goes beyond standard stopword removal. A **Contextual Regex** system was implemented to disambiguate polysemous terms in informal Brazilian Portuguese.

```python
# Example: "MEI" as slang ("meio" = "half") vs. the tax regime
'mei_context': r'(?:\b(?:o|do|no|pro|meu|um|abrir|sou|virar|ser|pagar|guia|boleto)\s+mei\b|'
               r'\bmei\s+(?:ta|é|atrasado|da|de|pra|cnpj|me)\b)'
```

Six contextual boolean flags are applied (`das_context`, `tax_context`, `pj_context`, `mei_context`, `uberizacao_context`, `precarious_context`), ensuring only records with confirmed fiscal or labor-related vocabulary advance to the NLP layer — acting as a cost-reduction pre-filter before expensive model inference.

---

## 6. NLP Pipeline

### 6.1 Semantic Cleaning — Zero-Shot Classification

**Model:** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

Each record is classified into one of 6 categories derived from the sociological literature on labor precarization:

| Final Category | Description |
|---|---|
| Pejotização and Subordination | CLT-to-PJ substitution, disguised employment bonds |
| Rights Precarization | Absence of vacation pay, FGTS, and social protection |
| Fiscal Risk and Debt | Overdue DAS payments, taxes, tax authority scrutiny |
| Survival and Necessity | Financial vulnerability, gig work, delivery apps |
| Management and Opportunity | Business strategy, positive entrepreneurship narrative |
| Noise/Off-Topic | Unrelated content, generic discussions |

**Confidence threshold:** `0.55` — selected after a sensitivity analysis that balanced retained data volume (~65% of the corpus) against the dataset's average confidence score. Records below the threshold are flagged and discarded before any downstream aggregation.

**Token context management:** The pipeline dynamically calculates a safe input token ceiling, reserving space for hypothesis label tokens and special token buffers, preventing `RuntimeError` from model context overflow during batch inference.

### 6.2 Sentiment Analysis

**Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment` (with local model fallback at `local_models/sentiment_model`)

Selected for being trained specifically on multilingual social media data, offering greater robustness to slang, abbreviations, and informal Brazilian Portuguese neologisms compared to general-purpose sentiment models. Output is normalized to three classes: `Positive`, `Neutral`, `Negative`.

### 6.3 Dynamic Topic Modeling — BERTopic

**Embedding Model:** `paraphrase-multilingual-mpnet-base-v2`

Key configuration decisions:

- **Guided Topic Modeling:** Seed word lists sourced directly from sociological theory (e.g., `["uber", "ifood", "entregador"]` for Uberization; `["pj", "clt", "vínculo"]` for Pejotização), ensuring topics align with the research hypotheses rather than converging on generic clusters.
- **Dynamic Topic Modeling (DTM):** `topics_over_time` with `nr_bins=10` to map the temporal evolution of narratives and enable correlation analysis against macroeconomic series.
- **`min_topic_size=30`** to prevent statistically insignificant micro-clusters from polluting the topic space.
- NLTK stopwords extended with informal Brazilian Portuguese terms (`"pra"`, `"vc"`, `"tá"`, `"pq"`).

Generated artifacts:
- `bertopic_barchart_tcc.png` — Most relevant terms per topic
- `bertopic_over_time_tcc.png` — Normalized temporal evolution of topics
- `topics_over_time_refined.csv` — DTM time series exported to BigQuery for cross-analysis

---

## 7. Design Decisions and Trade-offs

### Why Mage.ai over Apache Airflow?

Airflow requires DAGs to be fully defined upfront and offers no native support for inspecting intermediate outputs, making the development cycle for ML pipelines slow and opaque. Mage treats each block as an individually inspectable artifact, significantly reducing debugging time in pipelines where NLP models have high iteration overhead.

### Why Zero-Shot over Fine-tuning?

The dataset has no manual labels. Fine-tuning would require annotating at least 500–1,000 examples per class, introducing annotator bias and compromising reproducibility. Zero-Shot classification with `mDeBERTa-v3` (trained on XNLI across 15 languages) removes that dependency entirely and allows the analytical categories to be revised without model retraining — a critical advantage for an evolving research domain.

### Why BERTopic over LDA?

LDA assumes documents are mixtures of topics drawn from a Dirichlet distribution, a model that degrades sharply on short texts such as Reddit comments. BERTopic uses dense contextual embeddings for clustering, capturing latent semantic relationships even in sparse, low-token inputs that would confound LDA's word co-occurrence assumptions.

### Confidence Threshold of 0.55

The sensitivity analysis (automatically generated as `grafico_sensibilidade_media_tcc.png`) identified 0.55 as the inflection point where the corpus's average confidence exceeds 70% without the data retention rate dropping below 60%, preserving statistical representativeness while maximizing classification quality.

---

## 8. Prerequisites and Configuration

### System Requirements

- Python **3.10+**
- Docker **20.10+** (recommended for full dependency isolation)
- GPU with **CUDA 11.8+** support (strongly recommended for NLP inference; CPU fallback available with significant performance degradation)
- Google Cloud project with **BigQuery API** enabled and a Service Account granted the `BigQuery Data Editor` role

### Environment Variables

Create a `.env` file at the project root:

```env
# Reddit API (PRAW)
CLIENT_ID=your_client_id
SECRET_KEY=your_secret_key
PASSWORD=your_reddit_password
USER_REDDIT=your_reddit_username

# Google Cloud (alternative to io_config.yaml)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
```

### BigQuery Credentials (Mage.ai)

Configure `io_config.yaml` with the `default` profile pointing to your Service Account:

```yaml
version: 0.1.1
default:
  GOOGLE_SERVICE_ACC_KEY_FILEPATH: "/home/src/service_account.json"
  GOOGLE_LOCATION: US
```

---

## 9. How to Run

### Via Docker (recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/social-sensing-pipeline.git
cd social-sensing-pipeline

# Start the environment with GPU support
docker compose up -d

# Access the Mage.ai UI
open http://localhost:6789
```

### Local Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Start the Mage.ai server
mage start .
```

### Pipeline Execution

The two pipelines are fully independent and can be triggered in parallel from the Mage.ai UI or CLI.

**Pipeline 1 — `reddit_ingestion`** (sequential, GPU-dependent)

```
1. get_reddit_data      → Raw collection (JSONs written to raw_data_reddit/)
2. transform_raw_data   → JSON parse + Contextual Regex → filtered DataFrame
3. semantic_cleaning    → Zero-Shot Classification + 0.55 confidence threshold
4. sentiment_analysis   → Sentiment score per record (Positive / Neutral / Negative)
5. topic_analysis       → Guided BERTopic + DTM + chart and CSV generation
        ├──► export_topics_metrics  → BigQuery: historico_topicos_mei
        └──► export_to_bigquery     → BigQuery: tabela_completa_vFinal
```

> **Note:** Steps 3–5 require GPU for viable execution time. Each step saves a `.parquet` checkpoint to disk, allowing reruns from any point without reprocessing the full chain.

**Pipeline 2 — `gov_data_pipeline`** (parallel loaders, no GPU required)

```
get_desocupacao_ibge ──► export_desocupacao_ibge  → BigQuery: tabela_desocupacao_ibge
get_ipca_metrics     ──► export_ipca_metrics       → BigQuery: tabela_ipca_acumulado
get_dividas_bc       ──► export_divida             → BigQuery: tabela_endividamento_bc
get_processos_tst    ──► search_tst_processos ──► export_processos_tst
                                                   → BigQuery: tabela_processos_tst
```

> **Note:** The four loader chains in `gov_data_pipeline` are independent of each other. Mage.ai will execute them concurrently within the same pipeline run.

---

## 10. Outputs and Artifacts

### BigQuery Tables (`First_Test_Sentiment_Analysis`)

| Table | Description |
|---|---|
| `tabela_completa_vFinal` | Main dataset: text + sentiment + zero-shot category + topic |
| `Analise_Semantica_55_acuracia` | Records approved with confidence threshold ≥ 0.55 |
| `historico_topicos_mei` | DTM time series for macroeconomic correlation analysis |
| `tabela_ipca_acumulado` | 12-month cumulative inflation (IBGE) |
| `tabela_desocupacao_ibge` | Monthly unemployment rate (PNAD Contínua) |
| `tabela_endividamento_bc` | Household debt & individual default rate (BCB) |
| `tabela_processos_tst` | TST new cases by subject and year |

### Visual Artifacts (generated at runtime)

| File | Description |
|---|---|
| `grafico_sensibilidade_media_tcc.png` | Sensitivity analysis: retained volume vs. average confidence by threshold |
| `grafico_distribuicao_categorias_tcc.png` | Confidence score distribution per zero-shot category (boxplot) |
| `radar_contextos_tcc.png` | Boolean context frequency in the filtered corpus (radar chart) |
| `bertopic_barchart_tcc.png` | Most relevant terms per BERTopic topic |
| `bertopic_over_time_tcc.png` | Normalized temporal evolution of topics |

---

## 11. References

- Abilio, L. C. (2019). *Uberização: do empreendedorismo para o autogerenciamento subordinado*. Psicoperspectivas, 18(3).
- Damasceno et al. (2020). *Breves considerações sobre a pejotização e a reforma trabalhista*. Anais Faculdade Processus.
- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*. arXiv:2203.05794.
- Xu, W. W. et al. (2022). *Unmasking the Twitter discourses on masks during the COVID-19 pandemic: BERT topic modeling approach*. JMIR Infodemiology.
- Zhang, B. et al. (2024). *Knowledge-augmented interpretable network for zero-shot stance detection on social media*. IEEE TCSS.

---

*Undergraduate Thesis (TCC) — Data Engineering applied to Computational Social Science*
*CEFET-MG, Department of Computing (DECOM) — Belo Horizonte, Brazil*

FROM mageai/mageai:latest

# Define o diretório de trabalho padrão do Mage
WORKDIR /home/src

# Copia e instala dependências
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Instala uma versão de Chromium para gerar gráficos na fase de análise de Tópicos
RUN apt-get update && apt-get install -y chromium

# Baixa o modelo do Spacy para português
RUN python3 -m spacy download pt_core_news_sm

# Variável para logs do Python aparecerem imediatamente
ENV PYTHONUNBUFFERED=1
FROM python:3.10-bullseye

# 作業ディレクトリ
WORKDIR /app

# 依存ファイルをコピーして先にインストール（キャッシュ効率化）
COPY requirements.txt .

RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
       git \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# アプリケーションのソースコードをコピー
COPY . .

# ポート指定（Streamlit のデフォルト）
EXPOSE 8501

# コンテナ起動時のコマンド
CMD ["streamlit", "run", "app/tsuzumi_RAG_Chat_Demo.py", "--server.port=8501", "--server.address=0.0.0.0"]
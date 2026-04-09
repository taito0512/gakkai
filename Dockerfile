# ベースイメージ: Python 3.11（軽量版）
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# ライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# コードをコピー
COPY . .

# デフォルトコマンド
CMD ["python", "--version"]

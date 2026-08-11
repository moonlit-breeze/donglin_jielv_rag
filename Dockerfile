# ============================================================
# 佛学戒律 RAG 系统 Dockerfile
# ============================================================
# 构建：docker build -t jielv-rag .
# 运行：docker run -p 7860:7860 --env-file .env jielv-rag
# 或使用 docker-compose：docker-compose up
# ============================================================

# 基础镜像：Python 3.10 slim（体积小、兼容性好）
FROM python:3.10-slim

# 工作目录
WORKDIR /app

# 系统依赖：chromadb 需要 gcc 编译，git 用于 FlagEmbedding
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存层
COPY requirements.txt .

# 安装 Python 依赖
# 使用 --no-cache-dir 减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 复制模型文件（如果本地有离线模型）
# 注意：如果模型在 models/ 目录，需要取消下面的注释
# COPY models/ ./models/

# 复制向量数据库（如果已构建）
# 注意：如果向量库在 chroma_db/ 目录，需要取消下面的注释
# COPY chroma_db/ ./chroma_db/

# 设置环境变量
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV PYTHONUNBUFFERED=1

# 暴露 Gradio 端口
EXPOSE 7860

# 启动命令
CMD ["python", "web_app.py"]

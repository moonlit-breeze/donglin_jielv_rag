"""
vector_store.py — 向量化存储模块

职责：
  1. 使用 HuggingFace 本地模型（BAAI/bge-small-zh）将文本转换为语义向量；
  2. 基于 Chroma 向量数据库实现文档的持久化存储与加载；
  3. 强制开启离线模式，避免运行时产生网络请求问题。
"""

import os

# 设置HuggingFace使用离线模式，避免网络请求问题
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "BAAI/bge-small-zh"
DB_DIR = "./chroma_db"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

def create_vectorstore(documents):
    embeddings = get_embeddings()
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return db

def load_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
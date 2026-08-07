"""
vector_store.py — 向量化存储模块

职责：
  1. 使用 HuggingFace 本地模型（BAAI/bge-small-zh）将文本转换为语义向量；
  2. 基于 Chroma 向量数据库实现文档的持久化存储与加载；
  3. 支持按身份分层建库（比丘戒 / 沙弥戒 / 居士戒），各身份独立存储，杜绝跨域混淆；
  4. 强制开启离线模式，避免运行时产生网络请求问题。
"""

import os

# 设置HuggingFace使用离线模式，避免网络请求问题
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "BAAI/bge-small-zh"

# 分层建库：每个身份对应独立的向量数据库目录和 ASCII 集合名
ROLE_DB_MAP = {
    "比丘戒": "./chroma_db/jie_lv",
    "沙弥戒": "./chroma_db/sha_mi",
    "居士戒": "./chroma_db/upasaka",
}

# Chroma collection name 只支持 [a-zA-Z0-9._-]，需用 ASCII 别名
_ROLE_COLLECTION = {
    "比丘戒": "jie_lv",
    "沙弥戒": "sha_mi",
    "居士戒": "upasaka",
}

ALL_ROLES = list(ROLE_DB_MAP.keys())

# 共享 embedding 实例，避免重复加载模型
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"local_files_only": True},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings

def create_vectorstore_for_role(role: str, documents):
    """为指定身份创建独立的向量数据库"""
    if role not in ROLE_DB_MAP:
        raise ValueError(f"未知身份：{role}，支持：{ALL_ROLES}")
    embeddings = get_embeddings()
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=ROLE_DB_MAP[role],
        collection_name=_ROLE_COLLECTION[role],
    )
    return db

def load_vectorstore_for_role(role: str):
    """加载指定身份的向量数据库"""
    if role not in ROLE_DB_MAP:
        raise ValueError(f"未知身份：{role}，支持：{ALL_ROLES}")
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=ROLE_DB_MAP[role],
        embedding_function=embeddings,
        collection_name=_ROLE_COLLECTION[role],
    )

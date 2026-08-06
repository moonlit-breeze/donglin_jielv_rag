"""
retriever.py — 语义检索模块

职责：
  基于向量数据库对用户问题进行相似度检索，返回最相关的戒律文档片段。
  支持按身份标签（居士戒 / 沙弥戒 / 比丘戒）进行过滤检索，
  以便针对不同身份返回精准的戒律参考内容。
"""

from rag.vector_store import load_vectorstore

_db = None

def _get_db():
    """懒加载向量数据库，首次调用时才加载模型权重"""
    global _db
    if _db is None:
        _db = load_vectorstore()
    return _db

def retrieve(question: str, role_filter: str = None, k: int = 3):
    """
    role_filter: "居士戒" / "沙弥戒" / "比丘戒"
    """
    db = _get_db()
    if role_filter:
        results = db.similarity_search(
            question,
            k=k,
            filter={"role": role_filter}
        )
    else:
        results = db.similarity_search(question, k=k)
    return results
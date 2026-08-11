"""
vector_store.py — 向量化存储模块

职责：
  1. 使用 HuggingFace 本地模型（BAAI/bge-small-zh）将文本转换为语义向量；
  2. 基于 Chroma 向量数据库实现文档的持久化存储与加载；
  3. 支持按身份分层建库（比丘戒 / 沙弥戒 / 居士戒），各身份独立存储，杜绝跨域混淆；
  4. 强制开启离线模式，避免运行时产生网络请求问题。

【小白导读】
  这个文件是 RAG 系统的「向量化引擎」，它做的事可以类比为一个「智能图书管理员」：

  1. 嵌入模型（Embedding）= 把文字变成向量
     - 比如“不杀生” → [0.12, -0.35, 0.78, ...]（512 个数字）
     - 语义相近的文字，生成的向量也相近
     - 这样我们就可以通过计算向量距离来找到“意思相近”的内容

  2. Chroma 向量数据库 = 专门存储向量的数据库
     - 普通数据库存文字，向量数据库存向量
     - 它最擅长的事：给一个向量，快速找到库里最相似的几个向量

  3. 分层建库 = 每个身份一个独立的库
     - 居士戒、沙弥戒、比丘戒各自有独立的向量库
     - 好处：问居士的问题绝不会查到比丘的戒律，杜绝跨域混淆
"""

import os

# ============================================================
# 离线模式设置（必须在导入任何 NLP 库之前！）
# ============================================================
# 为什么需要离线模式？
#   HuggingFace 库默认会在启动时联网检查模型更新，
#   在内网/代理环境下会报 SSL 错误或卡死。
#   设置这两个环境变量后，库只会从本地缓存加载模型，不会联网。
# ============================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 嵌入模型名称
# bge-small-zh 是 BAAI 开源的中文语义嵌入模型，
# 专门针对中文优化，模型小（约 100MB），推理快，效果好。
EMBEDDING_MODEL = "BAAI/bge-small-zh"

# ============================================================
# 分层建库配置
# ============================================================
# 为什么每个身份要单独的向量库？
#   如果所有戒律都放在一个库里，问“居士可以喝酒吗”时，
#   可能会检索到比丘戒的内容，这对居士来说是不适用的。
#   分开存储后，检索时只查询对应身份的库，从物理层面杜绝跨域。
#
# ROLE_DB_MAP：每个身份对应的向量库存储目录
# ============================================================
ROLE_DB_MAP = {
    "比丘戒": "./chroma_db/jie_lv",   # 比丘（受具足戒的男性僧人）
    "沙弥戒": "./chroma_db/sha_mi",   # 沙弥（未受具足戒的男性僧人）
    "居士戒": "./chroma_db/upasaka",  # 居士（在家学佛的信众）
}

# Chroma 的 collection name 只支持 ASCII 字符（a-zA-Z0-9._-），
# 所以中文身份需要用 ASCII 别名
_ROLE_COLLECTION = {
    "比丘戒": "jie_lv",
    "沙弥戒": "sha_mi",
    "居士戒": "upasaka",
}

# 所有支持的身份列表，供其他模块使用
ALL_ROLES = list(ROLE_DB_MAP.keys())  # → ["比丘戒", "沙弥戒", "居士戒"]

# ============================================================
# 嵌入模型的延迟加载
# ============================================================
# 为什么要延迟加载？
#   加载嵌入模型需要约 2-5 秒，如果在程序启动时就加载，
#   会拖慢启动速度。用“懒加载”模式，第一次调用时才加载，
#   之后复用已加载的模型实例。
# ============================================================
_embeddings = None


def get_embeddings():
    """
    获取嵌入模型实例（懒加载，全局单例）。

    返回：
      HuggingFaceEmbeddings 实例，可以用来把文本转成向量。

    【小白提示】
    这个函数用了经典的「单例模式」：
    - 第一次调用：创建模型实例并缓存到 _embeddings
    - 后续调用：直接返回已缓存的实例，不重复加载
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            # local_files_only=True：只从本地缓存加载，不联网下载
            model_kwargs={"local_files_only": True},
            # normalize_embeddings=True：生成的向量做归一化处理
            # 归一化后，向量之间的余弦相似度 = 点积，计算更快
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings

def create_vectorstore_for_role(role: str, documents):
    """
    为指定身份创建独立的向量数据库（写入/重建）。

    参数：
      role: 身份名称，如 "居士戒"
      documents: List[Document] —— 要存入的文档列表
    返回：
      Chroma 向量数据库实例

    【小白提示】
    这个函数在「初始化向量库」时使用（python init_db.py），
    它会把所有文档通过嵌入模型转成向量，然后存入 Chroma 数据库。
    注意：如果目标目录已有数据，会被覆盖（重建）。
    """
    if role not in ROLE_DB_MAP:
        raise ValueError(f"未知身份：{role}，支持：{ALL_ROLES}")
    embeddings = get_embeddings()
    # Chroma.from_documents()：
    #   1. 对每个 document 调用嵌入模型，生成向量
    #   2. 将向量和原文一起存入 Chroma 数据库
    #   3. 自动建立向量索引，支持快速相似度搜索
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=ROLE_DB_MAP[role],       # 存储目录
        collection_name=_ROLE_COLLECTION[role],    # 集合名（ASCII）
    )
    return db

def load_vectorstore_for_role(role: str):
    """
    加载指定身份的向量数据库（只读查询）。

    参数：
      role: 身份名称，如 "居士戒"
    返回：
      Chroma 向量数据库实例，可以用来做相似度搜索

    【小白提示】
    与 create 不同，这个函数在「回答问题」时使用。
    它不会创建新数据，只是加载已有的向量库，供检索模块调用。
    """
    if role not in ROLE_DB_MAP:
        raise ValueError(f"未知身份：{role}，支持：{ALL_ROLES}")
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=ROLE_DB_MAP[role],
        embedding_function=embeddings,
        collection_name=_ROLE_COLLECTION[role],
    )

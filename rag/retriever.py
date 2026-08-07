"""
retriever.py — 语义检索模块

职责：
  基于分层向量数据库对用户问题进行相似度检索，返回最相关的戒律文档片段。
  核心机制：
  1. 分层检索：根据身份只查询对应库，绝不会把比丘戒答给居士；
  2. 跨域拦截：metadata 过滤 domain="jielv"，防止非戒律问题得到牵强回答；
  3. 相关度阈值：过滤掉相似度不足的结果，避免硬凑答案。
"""

from rag.vector_store import load_vectorstore_for_role, ALL_ROLES

# 懒加载：按身份缓存向量数据库，首次查询时才加载
_dbs = {}

def _get_db(role: str):
    """懒加载指定身份的向量数据库"""
    if role not in _dbs:
        _dbs[role] = load_vectorstore_for_role(role)
    return _dbs[role]

# 相关度阈值：低于此分数的结果不返回，防止硬凑答案
# 实测：戒律相关问题 top1 分数 ≈ 0.55–0.67，无关问题 ≈ 0.45–0.53
MIN_RELEVANCE = 0.50

def retrieve(question: str, role_filter: str = None, k: int = 3):
    """
    检索与问题最相关的戒律文档。

    role_filter: "居士戒" / "沙弥戒" / "比丘戒"，None 表示全部检索
    返回: List[Document]，按相关度降序
    """
    # 始终附加 domain 过滤，禁止跨域检索
    domain_filter = {"domain": "jielv"}

    # 确定要检索的身份列表
    if role_filter:
        roles_to_search = [role_filter]
    else:
        roles_to_search = ALL_ROLES

    all_results = []
    for role in roles_to_search:
        try:
            db = _get_db(role)
            # 使用带相关度分数的检索，用于阈值过滤
            results_with_scores = db.similarity_search_with_relevance_scores(
                question, k=k, filter=domain_filter
            )
            for doc, score in results_with_scores:
                if score >= MIN_RELEVANCE:
                    all_results.append((doc, score))
        except Exception:
            # 某个身份的库不存在（尚未初始化），跳过
            continue

    # 按相关度降序排序，取前 k 条
    all_results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in all_results[:k]]

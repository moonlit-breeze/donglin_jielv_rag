"""
retriever.py — 语义检索模块

职责：
  基于分层向量数据库对用户问题进行相似度检索，返回最相关的戒律文档片段。
  核心机制：
  1. 分层检索：根据身份只查询对应库，绝不会把比丘戒答给居士；
  2. 跨域拦截：metadata 过滤 domain="jielv"，防止非戒律问题得到牵强回答；
  3. 相关度阈值：过滤掉相似度不足的结果，避免硬凑答案；
  4. 混合召回：语义检索 + 同义词扩展 + 关键词匹配 + RRF 重排，提高召回率。
"""

from rag.vector_store import load_vectorstore_for_role, ALL_ROLES

# 懒加载：按身份缓存向量数据库，首次查询时才加载
_dbs = {}

# 简单停用词集合，关键词匹配时过滤
_STOPWORDS = set(
    "的 是 了 在 和 与 或 可以 能 吗 呢 吧 啊 我 你 他 她 它 们 这 那 有 个 为 之 而 以 及 其 该 请 问 如何 什么 哪些 怎么 不 要 会 都 就 也 很 但 么 着 过 来 去 上 下".split()
)

# 同义词扩展表：口语化表达 → 可能的戒律术语
# 用于关键词召回，解决"换个词就搜不到"的问题
_SYNONYMS = {
    "吃饭": ["食", "非时食", "持午", "过午不食", "斋"],
    "吃肉": ["杀生", "食肉", "三净肉", "荤"],
    "喝酒": ["饮酒", "酒戒", "戒酒"],
    "钱": ["金银", "财宝", "金钱", "持金钱"],
    "衣服": ["三衣", "袈裟", "衣", "着装"],
    "结婚": ["婚姻", "嫁娶", "淫欲", "邪淫"],
    "说谎": ["妄语", "大妄语", "虚诳"],
    "唱歌": ["歌舞", "观听", "伎乐", "音乐"],
    "下午": ["非时", "日暮", "黄昏"],
    "比丘": ["比丘戒", "具足戒", "大戒"],
    "沙弥": ["沙弥戒", "十戒"],
    "居士": ["居士戒", "五戒", "优婆塞", "优婆夷"],
}


def _get_db(role: str):
    """懒加载指定身份的向量数据库"""
    if role not in _dbs:
        _dbs[role] = load_vectorstore_for_role(role)
    return _dbs[role]


# 相关度阈值：低于此分数的结果不返回，防止硬凑答案
# 实测：戒律相关问题 top1 分数 ≈ 0.55–0.67，无关问题 ≈ 0.45–0.53
MIN_RELEVANCE = 0.50


def _expand_terms(question: str):
    """
    从问题中提取关键词，并按同义词表扩展。
    返回：dict，key 为扩展后的词，value 为原始权重。
    """
    # 提取 2 字及以上的有效词（去掉纯停用词组合）
    raw_terms = set()
    for length in (4, 3, 2):
        for i in range(len(question) - length + 1):
            term = question[i:i + length]
            if any(c in _STOPWORDS for c in term):
                continue
            raw_terms.add(term)

    # 加入单字（非停用词）
    for c in question:
        if c not in _STOPWORDS:
            raw_terms.add(c)

    # 同义词扩展
    expanded = {}
    for term in raw_terms:
        expanded[term] = 1.0
        for key, syns in _SYNONYMS.items():
            if key in term or term in key:
                for syn in syns:
                    expanded[syn] = expanded.get(syn, 0.0) + 0.8
                expanded[term] = max(expanded[term], 1.0)
    return expanded


def _keyword_search(question: str, role: str, k: int):
    """
    在指定身份库中进行关键词匹配召回，使用同义词扩展和命中次数评分。
    当语义检索无结果或结果不足时使用，作为补充召回手段。
    """
    terms = _expand_terms(question)
    if not terms:
        return []

    try:
        db = _get_db(role)
        data = db.get()
        docs = data.get("documents", [])
        metadatas = data.get("metadatas", [])
    except Exception:
        return []

    scored = []
    for doc_text, meta in zip(docs, metadatas):
        if not doc_text:
            continue
        score = 0.0
        for term, weight in terms.items():
            if term in doc_text:
                score += weight
        if score > 0:
            from langchain_core.documents import Document
            scored.append((Document(page_content=doc_text, metadata=meta or {}), score))

    # 按命中分数降序，取前 k
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:k]]


def _rrf_fusion(rank_lists: list, k: int = 60):
    """
    Reciprocal Rank Fusion：多路召回结果融合。
    rank_lists: List[List[Document]]，每一路按相关性排序
    返回：融合后排序的 Document 列表
    """
    scores = {}
    doc_key = lambda d: (d.page_content, tuple(sorted(d.metadata.items())))

    for docs in rank_lists:
        for rank, doc in enumerate(docs, start=1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    if not scores:
        return []

    # 按 key 取 doc 对象，按 RRF 分数排序
    doc_map = {}
    for docs in rank_lists:
        for doc in docs:
            key = doc_key(doc)
            doc_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


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

    semantic_results = []
    keyword_results = []

    for role in roles_to_search:
        try:
            db = _get_db(role)
            # 语义检索：使用带相关度分数的检索，用于阈值过滤
            results_with_scores = db.similarity_search_with_relevance_scores(
                question, k=k, filter=domain_filter
            )
            role_semantic = []
            for doc, score in results_with_scores:
                if score >= MIN_RELEVANCE:
                    role_semantic.append(doc)
            semantic_results.extend(role_semantic)

            # 关键词检索：作为补充召回
            role_kw = _keyword_search(question, role, k=k)
            keyword_results.extend(role_kw)
        except Exception:
            # 某个身份的库不存在（尚未初始化），跳过
            continue

    # 若语义检索有结果，与关键词结果做 RRF 融合
    # 若语义检索无结果，仅返回关键词结果
    if semantic_results:
        final_results = _rrf_fusion([semantic_results, keyword_results], k=60)[:k]
    else:
        final_results = keyword_results[:k]

    return final_results

"""
retriever.py — 语义检索模块

职责：
  基于分层向量数据库对用户问题进行相似度检索，返回最相关的戒律文档片段。
  核心机制：
  1. 分层检索：根据身份只查询对应库，绝不会把比丘戒答给居士；
  2. 跨域拦截：metadata 过滤 domain="jielv"，防止非戒律问题得到牵强回答；
  3. 相关度阈值：过滤掉相似度不足的结果，避免硬凑答案；
  4. 混合召回：语义检索 + 同义词扩展 + 关键词匹配 + RRF 重排，提高召回率；
  5. Reranker 精排：使用 cross-encoder 对候选文档进行二次打分排序，提升检索相关性。
"""

from rag.vector_store import load_vectorstore_for_role, ALL_ROLES
from FlagEmbedding import FlagReranker

# 懒加载：按身份缓存向量数据库，首次查询时才加载
_dbs = {}

# 延迟加载，避免启动时拖慢速度
_reranker = None

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

# Reranker 精排相关度阈值：低于此分数的结果不返回
# Reranker 分数范围 [-∞, +∞]（normalize=True 后约 [0, 1]），0.1 为较宽松的兜底线
RERANK_MIN_SCORE = 0.1


def _extract_terms(question: str):
    """
    从问题中提取关键词（2-4 字子串），过滤停用词。
    返回 set。
    """
    raw_terms = set()
    for length in (4, 3, 2):
        for i in range(len(question) - length + 1):
            term = question[i:i + length]
            if any(c in _STOPWORDS for c in term):
                continue
            raw_terms.add(term)
    # 单字
    for c in question:
        if c not in _STOPWORDS:
            raw_terms.add(c)
    return raw_terms


def _expand_terms(question: str):
    """
    从问题中提取关键词，并按同义词表扩展。
    返回：dict，key 为扩展后的词，value 为权重。

    注意：同义词只根据原始问题中是否出现同义词 key 来触发一次，
    避免多个子串重复触发导致通用词（如"五戒"）权重过高。
    """
    raw_terms = _extract_terms(question)
    expanded = {term: 1.0 for term in raw_terms}

    # 基于整个问题做同义词扩展，每个 key 只触发一次
    for key, syns in _SYNONYMS.items():
        if key in question:
            for syn in syns:
                # 同义词权重低于原始词，避免喧宾夺主
                expanded[syn] = max(expanded.get(syn, 0.0), 0.5)

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
            # 忽略单字，避免过于宽泛的匹配
            if len(term) < 2:
                continue
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

def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(
            'BAAI/bge-reranker-v2-m3',
            use_fp16=True  # 用半精度，降低显存/内存占用
        )
    return _reranker
    
def retrieve(question: str, role_filter: str = None, k: int = 3, rerank: bool = False):
    """
    检索与问题最相关的戒律文档。

    role_filter: "居士戒" / "沙弥戒" / "比丘戒"，None 表示全部检索
    rerank: 是否启用 Reranker 精排（cross-encoder 二次打分，提升相关性）
    返回: List[Document]，按相关度降序
    """
    # 始终附加 domain 过滤，禁止跨域检索
    domain_filter = {"domain": "jielv"}

    # 确定要检索的身份列表
    if role_filter:
        roles_to_search = [role_filter]
    else:
        roles_to_search = ALL_ROLES

    # Reranker 开启时扩大候选池，给精排更多选择
    search_k = 15 if rerank else k

    semantic_results = []
    keyword_results = []

    for role in roles_to_search:
        try:
            db = _get_db(role)
            # 语义检索：使用带相关度分数的检索，用于阈值过滤
            results_with_scores = db.similarity_search_with_relevance_scores(
                question, k=search_k, filter=domain_filter
            )
            role_semantic = []
            for doc, score in results_with_scores:
                if score >= MIN_RELEVANCE:
                    role_semantic.append(doc)
            semantic_results.extend(role_semantic)

            # 关键词检索：作为补充召回
            role_kw = _keyword_search(question, role, k=search_k)
            keyword_results.extend(role_kw)
        except Exception:
            # 某个身份的库不存在（尚未初始化），跳过
            continue

    # 若语义检索结果足够（>= k/2 条且分数达标），优先使用语义结果，
    # 仅在语义结果不足时与关键词结果做 RRF 融合，避免关键词噪声拉低排序。
    min_semantic_needed = max(1, search_k // 2)
    if len(semantic_results) >= min_semantic_needed:
        candidates = semantic_results
    elif semantic_results:
        candidates = _rrf_fusion([semantic_results, keyword_results], k=60)
    else:
        candidates = keyword_results

    # --- Reranker 精排 ---
    if rerank and candidates:
        reranker = _get_reranker()
        pairs = [[question, doc.page_content] for doc in candidates]
        scores = reranker.compute_score(pairs, normalize=True)
        # compute_score 对单条输入返回 float，多条返回 list
        if isinstance(scores, (int, float)):
            scores = [scores]
        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        # 过滤低于阈值的精排结果
        candidates = [doc for score, doc in scored if score >= RERANK_MIN_SCORE]

    # 截断到最终 k 条
    final_results = candidates[:k]

    # 去重：同一段原文可能在多个身份库中重复存在（如"通用"内容）
    seen = set()
    deduped = []
    for doc in final_results:
        content = doc.page_content or ""
        source = str(doc.metadata.get("source", ""))
        key = (content.strip(), source.strip())
        if key not in seen:
            seen.add(key)
            deduped.append(doc)

    return deduped


def retrieve_with_scores(question: str, role_filter: str = None, k: int = 3, rerank: bool = False):
    """
    检索并返回 reranker 分数，用于对比实验。
    rerank=True 时返回 (List[Document], List[float])；
    rerank=False 时返回 (List[Document], None)。
    """
    domain_filter = {"domain": "jielv"}
    if role_filter:
        roles_to_search = [role_filter]
    else:
        roles_to_search = ALL_ROLES

    search_k = 15 if rerank else k
    semantic_results = []
    keyword_results = []

    for role in roles_to_search:
        try:
            db = _get_db(role)
            results_with_scores = db.similarity_search_with_relevance_scores(
                question, k=search_k, filter=domain_filter
            )
            for doc, score in results_with_scores:
                if score >= MIN_RELEVANCE:
                    semantic_results.append(doc)
            keyword_results.extend(_keyword_search(question, role, k=search_k))
        except Exception:
            continue

    min_semantic_needed = max(1, search_k // 2)
    if len(semantic_results) >= min_semantic_needed:
        candidates = semantic_results
    elif semantic_results:
        candidates = _rrf_fusion([semantic_results, keyword_results], k=60)
    else:
        candidates = keyword_results

    if not rerank:
        # 去重 + 截断
        seen = set()
        deduped = []
        for doc in candidates[:k]:
            key = (doc.page_content.strip(), str(doc.metadata.get("source", "")).strip())
            if key not in seen:
                seen.add(key)
                deduped.append(doc)
        return deduped, None

    # Reranker 精排
    reranker = _get_reranker()
    pairs = [[question, doc.page_content] for doc in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, (int, float)):
        scores = [scores]
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    result_docs = []
    result_scores = []
    seen = set()
    for score, doc in scored:
        key = (doc.page_content.strip(), str(doc.metadata.get("source", "")).strip())
        if key in seen:
            continue
        seen.add(key)
        result_docs.append(doc)
        result_scores.append(score)
        if len(result_docs) >= k:
            break

    return result_docs, result_scores

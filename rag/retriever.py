"""
retriever.py — 语义检索模块

职责：
  基于分层向量数据库对用户问题进行相似度检索，返回最相关的戒律文档片段。
  核心机制：
  1. 分层检索：根据身份只查询对应库，绝不会把比丘戒答给居士；
  2. 跨域拦截：metadata 过滤 domain="jielv"，防止非戒律问题得到牵强回答；
  3. 相关度阈值：过滤掉相似度不足的结果，避免硬凑答案；
  4. 混合召回：语义检索 + 同义词扩展 + 倒排索引关键词匹配 + RRF 重排，提高召回率；
  5. Reranker 精排：使用 cross-encoder 对候选文档进行二次打分排序，提升检索相关性。

【小白导读】—— 这是整个系统最核心、最复杂的文件！

  这个文件做的事可以用一句话概括：
  “用户问了一个问题，帮他在知识库里找到最相关的几条内容。”

  听起来简单，但要做好很难。本系统用了 5 层策略来提升检索质量：
  其中关键词匹配层使用倒排索引加速，避免每次全量扫描文档。

  第 1 层：语义检索（Bi-encoder）
    - 把问题和知识库都转成向量，算相似度
    - 优点：能理解语义（“喝酒”和“饮酒”意思相近）
    - 缺点：有时不够精准

  第 2 层：关键词匹配（基于倒排索引）
    - 预先建好「词 → 文档」的倒排索引，查询时直接命中，避免全量扫描
    - 直接看哪些文档包含了问题中的关键词
    - 作为语义检索的补充，解决“换个词就搜不到”的问题

  第 3 层：同义词扩展
    - “吃饭” → 自动搜索“食”“非时食”“持午”等相关词
    - 解决口语化提问 vs 书面化知识库的差异

  第 4 层：RRF 融合
    - 把语义检索和关键词检索的结果合并排序
    - 公式：score(d) = Σ 1/(k + rank)，多路都排前面的得分更高

  第 5 层：Reranker 精排（可选）
    - 用更精准的 cross-encoder 模型对候选结果二次打分
    - 慢但准，能把真正相关的排到最前面
"""

import os
# 离线模式：必须在 import FlagEmbedding 之前设置
# FlagReranker 内部使用 huggingface_hub 下载模型，
# 如果这些环境变量没设置，它会尝试联网，在内网/代理环境下会报 SSL 错误。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from rag.vector_store import load_vectorstore_for_role, ALL_ROLES
from FlagEmbedding import FlagReranker
import hashlib

# ============================================================
# 懒加载：按身份缓存向量数据库
# ============================================================
# 为什么不一次性加载所有库？
#   每个向量库加载到内存需要时间和空间，
#   如果用户只问了居士戒的问题，加载比丘戒和沙弥戒的库就是浪费。
#   用字典缓存，第一次查询某个身份时才加载，之后复用。
# ============================================================
_dbs = {}

# 延迟加载，避免启动时拖慢速度
_reranker = None

# ============================================================
# Reranker 结果缓存（LRU）
# ============================================================
# 为什么需要缓存？
#   同一个问题短时间内可能被多次提问（用户重试、多轮对话等），
#   每次都对 15-30 条文档做 cross-encoder 计算很耗时。
#   用 LRU 缓存最近 64 个查询的精排结果，重复提问时直接返回。
# ============================================================
from collections import OrderedDict
import time as _time

_RERANK_CACHE_MAX = 64
_rerank_cache = OrderedDict()  # key: (question, role, k) → value: List[Document]

# ============================================================
# 检索结果缓存层（语义缓存）
# ============================================================
# 与 reranker 缓存不同，这个缓存层覆盖所有检索结果（含非精排）。
# 同一个问题 + 身份 + k 在短时间内重复查询时，直接返回缓存。
# TTL = 5 分钟，防止返回过时结果。
# ============================================================
_RETRIEVE_CACHE_MAX = 128
_RETRIEVE_CACHE_TTL = 300  # 5 分钟
_retrieve_cache = OrderedDict()  # key: hash → (timestamp, List[Document])

# ============================================================
# 停用词集合
# ============================================================
# 什么是停用词？
#   就是“的、是、了、吗”这类在任何文档中都会大量出现的词。
#   如果关键词匹配时包含这些词，几乎每篇文档都会被匹配到，
#   完全没有区分度。所以要过滤掉。
# ============================================================
_STOPWORDS = set(
    "的 是 了 在 和 与 或 可以 能 吗 呢 吧 啊 我 你 他 她 它 们 这 那 有 个 为 之 而 以 及 其 该 请 问 如何 什么 哪些 怎么 不 要 会 都 就 也 很 但 么 着 过 来 去 上 下".split()
)

# ============================================================
# 同义词扩展表
# ============================================================
# 为什么需要同义词扩展？
#   用户问的是口语：“可以吃饭吗？”
#   知识库里写的是书面语：“非时食”“持午”“过午不食”
#   语义检索能处理一部分，但关键词匹配就搜不到了。
#   同义词表就是桥接这个差异的桥梁。
#
# 怎么用？
#   如果问题中出现左边的 key（如“吃饭”），
#   就会自动把右边的同义词也加入搜索范围。
# ============================================================
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
    # 金融/生计类（涉及“正命”“不偷盗”“贪欲”的延伸讨论）
    "炒股": ["正命", "生计", "不偷盗", "贪欲", "投资"],
    "股票": ["正命", "生计", "不偷盗", "贪欲"],
    "基金": ["正命", "生计", "金钱"],
    "期货": ["正命", "生计", "贪欲"],
    "赌博": ["不偷盗", "贪欲", "博弈", "正命"],
    "吸毒": ["饮酒", "不饮酒", "迷醉", "麻醉"],
    "色情": ["邪淫", "不邪淫", "淫欲"],
    "杀人": ["杀生", "不杀生", "断人命"],
    # 身份类
    "比丘": ["比丘戒", "具足戒", "大戒"],
    "沙弥": ["沙弥戒", "十戒"],
    "居士": ["居士戒", "五戒", "优婆塞", "优婆夷"],
}


def _get_db(role: str):
    """懒加载指定身份的向量数据库"""
    if role not in _dbs:
        _dbs[role] = load_vectorstore_for_role(role)
    return _dbs[role]


# ============================================================
# Query 改写（LLM 辅助）
# ============================================================
# 用户口语提问有时太短或太模糊，向量检索效果不好。
# 例如：“吃个饭行不行” → 知识库用的是“非时食”“持午”
# 用 LLM 将口语转换为规范检索词，能显著提升召回率。
#
# 策略：仅对短问题（< 20 字）触发改写，避免浪费 token。
# ============================================================
_REWRITE_CACHE_MAX = 128
_rewrite_cache = OrderedDict()  # key: question → value: rewritten question


def _rewrite_query(question: str, role_filter: str = None) -> str:
    """
    用 LLM 将口语化问题改写为规范检索词。
    仅对短问题触发，长问题本身就足够清晰。
    返回改写后的问题，改写失败则返回原始问题。
    """
    # 长问题不改写
    if len(question) >= 20:
        return question

    # 查缓存
    cache_key = f"{question}||{role_filter or 'ALL'}"
    if cache_key in _rewrite_cache:
        _rewrite_cache.move_to_end(cache_key)
        return _rewrite_cache[cache_key]

    try:
        from rag.llm_client import create_provider, call_with_retry
        provider = create_provider()
        role_hint = f"（当前身份：{role_filter}）" if role_filter else ""
        messages = [
            {"role": "system", "content": (
                "你是一个佛教戒律检索查询改写助手。\n"
                "用户会用口语提问，你需要将其改写为更规范的检索查询，"
                "帮助向量检索系统找到更相关的戒律内容。\n"
                "只输出改写后的查询，不要解释。\n"
                "如果原问题已经足够规范，直接原样输出。\n"
                "改写时考虑戒律领域的术语映射：\n"
                "  吃饭→非时食/持午；喝酒→饮酒；说谎→妄语；"
                "杀人→杀生；偷东西→偷盗；结婚→婚姻/邪淫；"
                "炒股→正命/投资；吸毒→饮酒/迷醉；色情→邪淫"
            )},
            {"role": "user", "content": f"原问题：{question}\n身份：{role_filter or '不限'}\n改写后："},
        ]
        rewritten = call_with_retry(provider, messages, temperature=0.1, timeout=15.0, max_retries=1)
        rewritten = rewritten.strip()
        # 安全校验：改写后不能比原来短太多（可能改写失败）
        if len(rewritten) < 2:
            return question

        # 缓存
        _rewrite_cache[cache_key] = rewritten
        if len(_rewrite_cache) > _REWRITE_CACHE_MAX:
            _rewrite_cache.popitem(last=False)

        return rewritten
    except Exception:
        return question


# 相关度阈值：低于此分数的结果不返回，防止硬凑答案
# 实测：戒律相关问题 top1 分数 ≈ 0.55–0.67，无关问题 ≈ 0.45–0.53
# 0.50 是一个经验值，正好在“相关”和“不相关”之间
MIN_RELEVANCE = 0.50

# Reranker 精排相关度阈值：低于此分数的结果不返回
# Reranker 分数范围 [-∞, +∞]（normalize=True 后约 [0, 1]）
# 0.1 为较宽松的兜底线，主要用来过滤完全不相关的结果
RERANK_MIN_SCORE = 0.1


def _extract_terms(question: str):
    """
    从问题中提取关键词（2-4 字子串），过滤停用词。
    返回 set。

    【小白提示】
    这个函数的思路很简单：
    把问题拆成所有可能的 2-4 字子串，过滤掉含停用词的。
    比如“居士可以喝酒吗”会被拆成：
      - 4字：“居士可以”“士可以喝”“可以喝酒”“以喝酒吗”
      - 3字：“居士可”“士可以”“可以喝”“以喝酒”“喝酒吗”
      - 2字：“居士”“士可”“可以”“以喝”“喝酒”“酒吗”
    然后过滤掉包含“可”“以”“吗”等停用词的子串，
    最终剩下“居士”“喝酒”“酒”等有意义的关键词。

    为什么要从长到短遍历？
    因为更长的子串更具体，比如“不杀生”比“杀”更有区分度。
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

    【小白提示】
    这个函数在 _extract_terms 的基础上做了一步扩展：
    如果问题中出现“吃饭”，就自动把“食”“非时食”“持午”等
    同义词也加入搜索范围，但给它们更低的权重（0.5）。

    为什么同义词权重只有 0.5？
    因为原始关键词（如“吃饭”）肯定比同义词（如“斋”）更相关，
    给同义词较低权重可以防止它们“喧宾夺主”。

    注意：同义词只根据原始问题中是否出现 key 来触发一次，
    避免多个子串重复触发导致通用词（如“五戒”）权重过高。
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


# ============================================================
# 倒排索引（关键词检索加速）
# ============================================================
# 什么是倒排索引？
#   原方案：遍历所有文档，看每篇文档里有没有查询词（正排，O(N)）
#   倒排索引：预先建好「词 → 包含该词的文档列表」映射，
#             查询时直接查映射，跳过无关文档（倒排，O(命中数)）
#
# 类比：正排像「一本一本书翻目录」，倒排像「书末的索引页」——
#       查"饮酒"这个词，索引页直接告诉你它出现在第 3、7、12 页。
# ============================================================

# 每个身份库各一份索引，懒加载构建，构建后缓存
# _inverted_index: role -> {term: [doc_key, ...]}
# _doc_registry:   role -> {doc_key: Document}，与索引平行，用于按 key 取回文档
_inverted_index = {}
_doc_registry = {}


def _build_inverted_index(role: str):
    """
    为指定身份库构建倒排索引（懒加载，只构建一次）。

    流程：
      1. 从向量库拉取该身份的全部文档
      2. 对每篇文档做 n-gram 提取（与查询侧 _extract_terms 完全对称）
      3. 对每个词，记录它出现在哪些文档里
      4. 结果缓存到 _inverted_index，下次查询直接复用
    """
    # 已经构建过，直接返回
    if role in _inverted_index:
        return

    from langchain_core.documents import Document

    db = _get_db(role)
    data = db.get()
    docs = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    index = {}      # term -> [doc_key, ...]
    registry = {}   # doc_key -> Document

    for i, (doc_text, meta) in enumerate(zip(docs, metadatas)):
        if not doc_text:
            continue
        doc = Document(page_content=doc_text, metadata=meta or {})
        doc_key = i          # 用下标做 key，简单稳定
        registry[doc_key] = doc

        # 对文档做和查询侧一样的 n-gram 提取
        # 注意：单字不进索引（查询侧也忽略单字），避免索引爆炸
        for term in _extract_terms(doc_text):
            if len(term) < 2:
                continue
            index.setdefault(term, []).append(doc_key)

    _inverted_index[role] = index
    _doc_registry[role] = registry


def _keyword_search(question: str, role: str, k: int):
    """
    倒排索引版关键词检索。

    与旧版的区别：
      旧版：每次查询都全量拉取文档，逐条 for 循环做 `term in doc_text`
      新版：先查倒排索引，直接拿到「包含该词的文档」，跳过无关文档
    """
    terms = _expand_terms(question)   # 提取查询词 + 同义词扩展
    if not terms:
        return []

    # 懒构建倒排索引（第一次用时构建，之后缓存）
    _build_inverted_index(role)

    index = _inverted_index.get(role, {})
    registry = _doc_registry.get(role, {})

    # 累加每个命中文档的得分
    scored = {}   # doc_key -> 累计分数
    for term, weight in terms.items():
        if len(term) < 2:          # 单字忽略，与旧版一致
            continue
        for doc_key in index.get(term, []):   # 直接查映射，不再扫全部文档
            scored[doc_key] = scored.get(doc_key, 0.0) + weight

    # 按分数降序，取前 k；同分时按 doc_key 升序（等价于原文档顺序，保证稳定）
    result = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return [registry[doc_key] for doc_key, _ in result[:k]]

def _rrf_fusion(rank_lists: list, k: int = 60):
    """
    Reciprocal Rank Fusion：多路召回结果融合。

    【小白提示】
    RRF 是一种简单有效的多路结果融合算法。
    假设我们有两个搜索结果列表：
      - 语义检索：[文档A, 文档B, 文档C]
      - 关键词检索：[文档B, 文档D, 文档A]

    RRF 的公式是：score(d) = Σ 1/(k + rank)
    其中 rank 是文档在某一路结果中的排名（从 1 开始）。

    举个例子（k=60）：
    - 文档A：语义第1名 + 关键词第3名 = 1/61 + 1/63 = 0.032
    - 文档B：语义第2名 + 关键词第1名 = 1/62 + 1/61 = 0.032
    - 文档C：语义第3名 = 1/63 = 0.016
    - 文档D：关键词第2名 = 1/62 = 0.016

    直觉：一个文档在多路召回中都排前面，它就越可能是好结果。
    k=60 是一个常用的平滑参数，防止排名靠前的结果分数差异过大。
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

# ============================================================
# Reranker 模型延迟加载
# ============================================================
# 为什么要延迟加载？
#   Reranker 模型（bge-reranker-v2-m3）约 1.1GB，
#   加载一次需要 2-4 分钟。如果用户没用 Reranker，
#   加载它就是浪费。所以只在第一次开启时才加载。
# ============================================================
def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(
            'BAAI/bge-reranker-v2-m3',
            use_fp16=True  # 用半精度，降低显存/内存占用
        )
    return _reranker


def _preload_reranker():
    """
    在后台线程中预加载 Reranker 模型。
    在 web_app.py 启动时调用，用户提问前就把模型加载好，
    避免首次提问时等待 2-4 分钟。
    """
    _get_reranker()


def _rerank_cache_key(question: str, role_filter: str, k: int) -> str:
    """生成缓存 key：基于问题+身份+条数的哈希。"""
    raw = f"{question}||{role_filter or 'ALL'}||{k}"
    return hashlib.md5(raw.encode()).hexdigest()


def _filter_by_sub_role(docs, sub_role_filter: str):
    """
    按细分身份（sub_role）后过滤候选文档。

    策略：
      - 如果未指定 sub_role_filter，直接返回原文档
      - 如果指定了，优先返回 sub_role 匹配的文档
      - 若没有匹配的，保留 sub_role 为空的文档（兼容旧数据）
      - 这样不会因为数据尚未标注 sub_role 而直接丢结果
    """
    if not sub_role_filter:
        return docs

    matched = []
    fallback = []
    for doc in docs:
        doc_sub = str(doc.metadata.get("sub_role", "")).strip()
        if doc_sub == sub_role_filter:
            matched.append(doc)
        elif not doc_sub:
            fallback.append(doc)

    # 有匹配则只返回匹配；否则返回未标注 sub_role 的文档
    return matched if matched else fallback


def retrieve(question: str, role_filter: str = None, sub_role_filter: str = None,
             k: int = 3, rerank: bool = False, rewrite: bool = False):
    """
    检索与问题最相关的戒律文档。—— 这是整个系统的主检索入口！

    参数：
      question:        用户的问题，如“居士可以喝酒吗”
      role_filter:     身份过滤，“居士戒”/“沙弥戒”/“比丘戒”，None 表示全部检索
      sub_role_filter: 细分身份过滤，如“五戒”/“菩萨戒·十重”，None 表示不过滤
      k:               返回结果数量，默认 3 条
      rerank:          是否启用 Reranker 精排（cross-encoder 二次打分，提升相关性）
    返回：
      List[Document]，按相关度降序

    【小白提示】完整流程：
      1. 确定搜哪些库（根据身份）
      2. 每个库做两路召回（语义 + 倒排索引关键词）
      3. 融合两路结果
      4. 可选：按 sub_role 做后过滤
      5. 可选：Reranker 精排
      6. 截断 + 去重 → 返回
    """
    # ============================================================
    # 检索缓存检查
    # ============================================================
    cache_key = hashlib.md5(f"{question}||{role_filter or 'ALL'}||{sub_role_filter or 'ALL'}||{k}||{rerank}".encode()).hexdigest()
    now = _time.time()
    if cache_key in _retrieve_cache:
        ts, cached_docs = _retrieve_cache[cache_key]
        if now - ts < _RETRIEVE_CACHE_TTL:
            _retrieve_cache.move_to_end(cache_key)
            return cached_docs
        else:
            del _retrieve_cache[cache_key]

    # ============================================================
    # Query 改写：将口语化问题转为规范检索词（可选）
    # ============================================================
    if rewrite:
        rewritten_question = _rewrite_query(question, role_filter)
    else:
        rewritten_question = question

    # 始终附加 domain 过滤，禁止跨域检索
    # 只检索 domain="jielv" 的条目，防止非戒律内容被检索到
    domain_filter = {"domain": "jielv"}

    # Step 1: 确定要检索的身份列表
    # 如果指定了身份，只查对应的库；否则查所有库
    if role_filter:
        roles_to_search = [role_filter]
    else:
        roles_to_search = ALL_ROLES

    # Reranker 开启时扩大候选池，给精排更多选择
    # 为什么？Reranker 比语义检索更精准，给它更多候选才能发挥优势。
    # 比如 k=3 时，语义检索只取 3 条，但 Reranker 会先取 10 条再精排选 3 条。
    # 优化：15→10，候选池依然充裕，但 Reranker 计算量减少 33%。
    search_k = 10 if rerank else k

    # 两路召回的结果分别存放
    semantic_results = []  # 语义检索结果（基于向量相似度）
    keyword_results = []   # 关键词检索结果（基于文本匹配）

    for role in roles_to_search:
        try:
            db = _get_db(role)
            # ----- 语义检索 -----
            # 使用改写后的问题做检索，提升召回率
            results_with_scores = db.similarity_search_with_relevance_scores(
                rewritten_question, k=search_k, filter=domain_filter
            )
            # 阈值过滤：只保留相似度 >= MIN_RELEVANCE 的结果
            # 为什么需要阈值？
            #   向量检索总会返回结果，即使完全不相关。
            #   比如问“今天天气”，库里最相似的可能只有 0.45 分，
            #   远低于正常问题的 0.55-0.67 分。
            #   阈值 0.50 就是用来过滤这种“硬凑”的结果。
            role_semantic = []
            for doc, score in results_with_scores:
                if score >= MIN_RELEVANCE:
                    role_semantic.append(doc)
            semantic_results.extend(role_semantic)

            # ----- 关键词检索（倒排索引）-----
            # 使用改写后的问题做关键词检索
            role_kw = _keyword_search(rewritten_question, role, k=search_k)
            keyword_results.extend(role_kw)
        except Exception:
            # 某个身份的库不存在（尚未初始化），跳过
            continue

    # Step 3: 融合策略
    # ============================================================
    # 为什么不总是融合两路结果？
    #   因为关键词检索有时会引入噪声（不相关但碰巧包含关键词的文档）。
    #   如果语义检索已经足够多（>= search_k/2），就只用语义结果，
    #   避免关键词噪声拉低排序。
    #   只有语义结果不足时，才用 RRF 融合两路结果。
    # ============================================================
    min_semantic_needed = max(1, search_k // 2)
    if len(semantic_results) >= min_semantic_needed:
        candidates = semantic_results
    elif semantic_results:
        candidates = _rrf_fusion([semantic_results, keyword_results], k=60)
    else:
        candidates = keyword_results

    # Step 4: Reranker 精排（可选）
    # ============================================================
    # 送入 Reranker 前先截断候选数量
    # 优化：30→15，实测 15 条候选已足够精排出 top-3，
    # 但 Reranker 计算量减半。
    # ============================================================
    _MAX_RERANK_INPUT = 15
    if rerank and len(candidates) > _MAX_RERANK_INPUT:
        candidates = candidates[:_MAX_RERANK_INPUT]

    # --- Reranker 精排核心逻辑 ---
    if rerank and candidates:
        # 先查缓存：相同问题 + 身份 + k 的结果直接返回
        cache_key = _rerank_cache_key(question, role_filter, k)
        if cache_key in _rerank_cache:
            _rerank_cache.move_to_end(cache_key)  # LRU 标记为最近使用
            cached = _rerank_cache[cache_key]
            # 缓存命中，直接走去重返回
            seen = set()
            deduped = []
            for doc in cached[:k]:
                content = doc.page_content or ""
                source = str(doc.metadata.get("source", ""))
                key = (content.strip(), source.strip())
                if key not in seen:
                    seen.add(key)
                    deduped.append(doc)
            return deduped

        reranker = _get_reranker()
        # 构造 (问题, 文档) 对，送入 cross-encoder
        # 优化：截断文档内容到 256 字，cross-encoder 处理 question+doc 拼接，
        # 文档越长计算越慢，256 字足够保留核心信息。
        _MAX_DOC_LEN = 256
        pairs = [[question, doc.page_content[:_MAX_DOC_LEN]] for doc in candidates]
        # batch_size 控制每次送入模型的样本数，增大会更快但占更多内存
        scores = reranker.compute_score(pairs, normalize=True, batch_size=8)
        # compute_score 对单条输入返回 float，多条返回 list
        # 这里统一转成 list
        if isinstance(scores, (int, float)):
            scores = [scores]
        # 按 Reranker 分数降序排列，过滤低于阈值的
        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        # 过滤低于阈值的精排结果
        candidates = [doc for score, doc in scored if score >= RERANK_MIN_SCORE]

        # 写入缓存
        _rerank_cache[cache_key] = candidates
        if len(_rerank_cache) > _RERANK_CACHE_MAX:
            _rerank_cache.popitem(last=False)  # 淘汰最旧的

    # Step 5: 按 sub_role 后过滤（如果指定了）
    candidates = _filter_by_sub_role(candidates, sub_role_filter)

    # Step 6: 先去重，再截断到最终 k 条
    # 去重：同一段原文可能在多个身份库中重复存在（如“通用”内容）
    # 用 (内容, 出处) 作为去重 key
    # 为什么先去重再截断？
    #   若先截断 top-k 再去重，top-k 内出现重复时结果会不足 k 条；
    #   先对完整候选池去重（保留精排分数靠前的重复项），再截断，
    #   既能保证返回条数充足，又不影响精排排序质量。
    seen = set()
    deduped = []
    for doc in candidates:
        content = doc.page_content or ""
        source = str(doc.metadata.get("source", ""))
        key = (content.strip(), source.strip())
        if key not in seen:
            seen.add(key)
            deduped.append(doc)

    final_results = deduped[:k]

    # ============================================================
    # 写入检索缓存
    # ============================================================
    _retrieve_cache[cache_key] = (now, deduped)
    if len(_retrieve_cache) > _RETRIEVE_CACHE_MAX:
        _retrieve_cache.popitem(last=False)

    return deduped


def retrieve_with_scores(question: str, role_filter: str = None, sub_role_filter: str = None,
                         k: int = 3, rerank: bool = False):
    """
    检索并返回 reranker 分数，用于对比实验。

    【小白提示】
    这个函数和 retrieve() 的逻辑几乎一样，
    唯一的区别是它还会返回 Reranker 的分数。
    主要用于 tests/test_reranker_comparison.py 中的对比实验，
    帮助我们观察 Reranker 的效果。
    实际业务中用的是 retrieve()，不用这个。

    返回：
      rerank=False 时：(List[Document], None)
      rerank=True 时：(List[Document], List[float])，float 是每条的 Reranker 分数
    """
    domain_filter = {"domain": "jielv"}
    if role_filter:
        roles_to_search = [role_filter]
    else:
        roles_to_search = ALL_ROLES

    search_k = 10 if rerank else k
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

    # 按 sub_role 后过滤
    candidates = _filter_by_sub_role(candidates, sub_role_filter)

    if not rerank:
        # 先去重，再截断（与 retrieve() 保持一致）
        # 先对完整候选池去重，再截断到 k 条，避免 top-k 内重复导致结果不足 k 条
        seen = set()
        deduped = []
        for doc in candidates:
            key = (doc.page_content.strip(), str(doc.metadata.get("source", "")).strip())
            if key not in seen:
                seen.add(key)
                deduped.append(doc)
        return deduped[:k], None

    # Reranker 精排（与 retrieve() 相同的优化）
    _MAX_RERANK_INPUT_WS = 15
    if len(candidates) > _MAX_RERANK_INPUT_WS:
        candidates = candidates[:_MAX_RERANK_INPUT_WS]

    reranker = _get_reranker()
    _MAX_DOC_LEN = 256
    pairs = [[question, doc.page_content[:_MAX_DOC_LEN]] for doc in candidates]
    scores = reranker.compute_score(pairs, normalize=True, batch_size=8)
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

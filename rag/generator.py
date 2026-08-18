"""
generator.py — 答案生成模块

职责：
  将检索到的戒律文档片段组装为上下文，调用 DeepSeek 大语言模型，
  根据参考资料生成结构化的佛教戒律问答回复。
  强制标注真实经文出处，跨域问题直接拒答。

【小白导读】
  这个文件是 RAG 流程的“最后一站”。
  前面的检索模块已经找到了相关文档，这里要做的是：

  1. 把检索到的文档拼成“参考资料”
  2. 把参考资料 + 用户问题 + 系统指令一起发给大模型
  3. 大模型根据参考资料生成回答

  关键概念：
  - SYSTEM_PROMPT：给大模型的“指令手册”，告诉它怎么回答
  - format_context()：把检索结果格式化为参考资料
  - generate()：主入口，串联整个生成流程
  - is_retrieval_relevant()：检索结果相关性检查（防止误导）
"""

from dotenv import load_dotenv
import os
import re
import json
from typing import List, Dict, Iterator, Optional

load_dotenv()

from rag.llm_client import create_provider, call_with_retry, stream_with_retry

# 全局 LLM Provider 缓存（按 provider+model 组合缓存，支持用户切换模型）
_providers = {}

def _get_provider(provider_name: str = None, model: str = None):
    """
    懒加载 LLM Provider。

    参数：
      provider_name: 可选，指定 provider
      model:         可选，指定模型名
    """
    key = f"{provider_name or 'env'}::{model or 'default'}"
    if key not in _providers:
        _providers[key] = create_provider(provider_name=provider_name, model=model)
    return _providers[key]

# ============================================================
# 详细程度提示词
# ============================================================
# 用户可以在 Web UI 中选择回答详细程度：简洁/标准/详细
# 这里通过追加不同的指令来控制模型的输出长度和深度
# ============================================================
_DETAIL_PROMPT = {
    "简洁": "回答请尽量简洁，直接给出结论和依据，不做过多展开。",
    "标准": "回答需清晰完整，包含结论、适用条件和依据。",
    "详细": "回答需详尽，除结论和依据外，可适当补充背景、开许、例外和相关说明。",
}

# JSON 模式提示词
# 注意：JSON 示例中的花括号必须用 {{ 和 }} 转义，
# 否则调用 .format() 时会被 Python 误解析为占位符
JSON_SYSTEM_PROMPT = """你是一位严谨的佛教戒律助手，仅回答与佛教戒律相关的问题。

【核心原则】
优先使用提供的参考资料回答；参考资料能完整覆盖问题时，必须严格依据参考资料，不得编造。

【回答原则】
1. 当前用户身份为「{role}」。回答时必须仅围绕该身份的戒条展开。
2. 若用户身份为「未指定」或「不限」，则在回答中区分不同身份的适用条件。
3. 必须区分戒条性质：性戒（根本戒）与遮戒（防范戒）的约束力度不同。

【输出格式】
必须按以下 JSON 格式输出，不要包含任何其他内容：
{{
  "answer": "正文回答",
  "sources": [
    {{"type": "知识库", "role": "居士戒", "severity": "遮戒", "category": "在家生活", "ref": "《增壹阿含经》"}},
    {{"type": "补充", "ref": "《善生经》"}}
  ],
  "severity": "遮戒",
  "confidence": "high|medium|low",
  "note": "可选补充说明"
}}

【confidence 说明】
- high：参考资料直接覆盖，答案确定。
- medium：部分参考或权威经典兜底，基本可信。
- low：知识库未覆盖，主要依赖模型记忆，需谨慎。

【越界处理】
若提问与佛教戒律无关，answer 填写：“此问题超出戒律问答范围，建议向相关领域咨询。”，confidence 为 low。
"""


def _call_llm(system_prompt: str, user_msg: str, max_retries: int = 2,
              chat_history: List[Dict[str, str]] = None,
              provider_name: str = None, model: str = None):
    """
    调用 LLM，支持多轮对话历史注入。

    【小白提示】
    - system_prompt: 系统指令（告诉模型"你是谁""怎么做"）
    - user_msg: 用户的问题 + 参考资料
    - chat_history: 多轮对话历史（可选）
    - temperature=0.1: 控制输出的随机性，越低越确定（戒律问答需要确定性）
    """
    messages = [{"role": "system", "content": system_prompt}]
    # 注入对话历史（最多 4 条，即 2 轮）
    if chat_history:
        for msg in chat_history[-4:]:
            messages.append(msg)
    messages.append({"role": "user", "content": user_msg})

    return call_with_retry(_get_provider(provider_name=provider_name, model=model), messages, max_retries=max_retries)


def _check_format(answer: str) -> bool:
    """
    简单校验回答是否包含必要的格式标识。
    检查回答中是否同时包含【答】和【依据】两个部分。
    如果缺少，说明模型没有遵守指令，会触发重试。
    """
    return "【答】" in answer and "【依据】" in answer


def _truncate_context(context: str, max_chars: int = 3000) -> str:
    """
    上下文过长时截断，避免超出模型处理上限。
    3000 字符 ≈ 约 1500 个汉字，加上系统提示词和用户问题，
    总计不会超出 DeepSeek 的 8K token 上下文窗口。
    """
    if len(context) <= max_chars:
        return context
    return context[:max_chars] + "\n...（上下文已截断）"

# ============================================================
# 主系统提示词（SYSTEM_PROMPT）
# ============================================================
# 这是发给大模型的“指令手册”，是整个生成质量的灵魂。
# 它告诉大模型：
#   - 你是谁（严谨的佛教戒律助手）
#   - 怎么回答（优先用参考资料，不足时用权威经典兜底）
#   - 回答的格式（【答】+ 【依据】）
#   - 不能做什么（跨域问题拒答、不得编造经文）
#
# 注意：{role} 是模板变量，会根据用户选择的身份动态替换
# 比如用户选“居士戒”，这里就会变成“当前用户身份为「居士戒」”
# ============================================================
SYSTEM_PROMPT = """你是一位严谨的佛教戒律助手，仅回答与佛教戒律相关的问题。

【核心原则】
优先使用提供的参考资料回答；参考资料能完整覆盖问题时，必须严格依据参考资料，不得编造。

【参考资料不足时的兜底规则】
若参考资料未能完全覆盖问题，但问题确属佛教戒律范畴，可有限度地引用你记忆中的权威佛教经典和祖师大德开示作为补充。但必须遵守：
1. 只能引用公认权威的佛教典籍或祖师著作：如《阿含经》《四分律》《梵网经》《楞严经》《优婆塞戒经》《善生经》《佛遗教经》《大智度论》《十善业道经》《地藏经》、印光大师文钞、大安法师开示等。
2. 不得虚构经文名称、律典条目或祖师言论；不确知的内容宁可不写，也不得编造。
3. 必须在【依据】中明确区分两类来源：
   - 来自参考资料：标注「知识库：居士戒·在家生活」
   - 来自外部权威经典：标注「补充：《善生经》」

【越界处理】
若提问与佛教戒律无关（如炒股、天气、编程等），请直接回答：
『此问题超出戒律问答范围，建议向相关领域咨询。』
不得从戒律资料中牵强附会地回答。

【回答原则】
1. 当前用户身份为「{role}」。回答时必须仅围绕该身份的戒条展开，不得主动介绍其他身份（如居士、沙弥）的持犯情况，除非参考资料中明确提到。
2. 若用户身份为「未指定」或「不限」，则在回答中区分不同身份的适用条件（居士 / 沙弥 / 比丘），不得将某一身份的戒条笼统推广到所有人。
3. 必须区分戒条性质：性戒（根本戒）与遮戒（防范戒）的约束力度不同，遮戒对未受戒者不构成约束。
4. 若参考资料中包含开许、例外、不同层次的说明（如「未受戒者无罪过」「方便说」等），必须完整体现，不得省略。
5. 不得将「已受戒者的戒条要求」笼统地回答为所有人都必须遵守。

【等级标注规则】
1. 来自知识库的内容，必须按参考资料中的severity字段标注等级（根本戒/重戒/轻戒/遮戒等）。
2. 来自权威经典兆底的内容，只有当该戒条等级在佛教传统中有公认定论时才标注；否则应说明"该戒条等级未在知识库中明确标注"，不得臆测。

{format_section}
"""

# ============================================================
# 深度思考（Chain-of-Thought）提示词
# ============================================================
# 开启“深度思考”后，模型会先分析戒条类型、适用范围，
# 再给出结论。类似于“先想清楚再回答”，比直接崩答案更准确。
# ============================================================
_COT_FORMAT = """
【深度思考模式】
请严格按以下四步结构回答：

【分析】
分析该问题涉及哪些戒条，属于哪类戒（性戒/遮戒/轻戒），约束力度如何。

【辨析】
区分不同身份（居士/沙弥/比丘）的适用差异，说明开许、例外和方便说。

【答】
给出明确结论（正文，需包含适用条件说明）

【依据】
（必须标注来源；若同时使用了参考资料和外部补充，需分别列出：
  ① 知识库来源：身份·等级·类别·原文出处；
  ② 补充来源：权威经典或法师著作名称。）
"""

# 标准格式（非深度思考）
_STANDARD_FORMAT = """
【回答格式】
【答】
（正文，需包含适用条件说明）

【依据】
（必须标注来源；若同时使用了参考资料和外部补充，需分别列出：
  ① 知识库来源：身份·等级·类别·原文出处；
  ② 补充来源：权威经典或法师著作名称。）
"""

def format_context(docs):
    """
    把检索到的文档格式化为参考资料文本。

    【小白提示】
    这个函数做的事很简单：把 Document 列表转成格式化的文本。
    每条文档会被格式化为：
      （居士戒·遮戒·在家生活）不饮酒戒的具体内容...[出处：《增壹阿含经》]

    这样大模型看到参考资料时，就能知道每条来自哪个身份、什么等级、哪本经典。
    """
    parts = []
    for doc in docs:
        role = doc.metadata.get("role", "")
        source = doc.metadata.get("source", "")
        category = doc.metadata.get("category", "")
        severity = doc.metadata.get("severity", "")
        # 构造来源标注：身份 + 类别 + 经文出处
        source_label = f"（{role}"
        if severity and severity != "未标注":
            source_label += f"·{severity}"
        if category:
            source_label += f"·{category}"
        source_label += "）"
        source_ref = f"\n[出处：{source}]" if source else ""
        parts.append(f"{source_label}{doc.page_content}{source_ref}")
    return "\n\n".join(parts)

# ============================================================
# 检索相关性校验（重要！）
# ============================================================
# 简单停用词集合，用于主题相关性校验
_STOPWORDS = set("的 是 了 在 和 与 或 可以 能 吗 呢 吧 啊 我 你 他 她 它 们 这 那 有 个 为 之 而 以 及 其 该 请 问 如何 什么 哪些 怎么 吗 不 要 会 都 就 都 也 很 但 吗 么 呢 吧".split())


def is_retrieval_relevant(question: str, docs) -> bool:
    """
    简单校验检索结果是否与问题主题相关。

    【小白提示】
    为什么需要这个检查？
    向量检索总会返回结果，即使完全不相关。
    比如问“比丘穿什么衣服”，但比丘戒库里只有基础戒条（杀盗淫妄酒），
    检索到的“不偷盗”显然和“衣服”无关。

    这个函数的做法很简单：
    看看问题中的字在参考资料中是否也出现。
    如果问题中的有效字在资料中完全没出现，说明检索失效了。

    注意：这个方法虽然简单，但对“同义词”不敏感。
    比如“衣服”和“三衣”是同义词但字面不同，可能会被误判。
    """
    q_chars = set(question) - _STOPWORDS
    if not q_chars:
        return True  # 无法判断，默认通过

    for doc in docs:
        content_chars = set(doc.page_content)
        if q_chars & content_chars:
            return True
    return False

def _add_confidence_marker(answer: str, confidence: str) -> str:
    """在回答末尾追加置信度标识。"""
    marker_map = {
        "high": "高置信度：知识库直接覆盖",
        "medium": "中置信度：部分参考或权威经典兜底",
        "low": "低置信度：知识库未覆盖，建议进一步核实",
    }
    marker = marker_map.get(confidence, "")
    if marker:
        return answer + f"\n\n*📊 置信度：{marker}*"
    return answer


# 匹配 LLM 可能自行输出的置信度文本（防止与系统追加的置信度重复）
_CONFIDENCE_RE = re.compile(
    r'\n*(?:---\s*\n*)?(?:\*{0,2})?置信度[：:]\s*[^\n]*(?:\*{0,2})?\s*$',
    re.MULTILINE
)

def _strip_llm_confidence(text: str) -> str:
    """剥离 LLM 自行输出的置信度文本，避免与系统追加的置信度重复。"""
    return _CONFIDENCE_RE.sub('', text).rstrip()


def _try_parse_json(answer: str):
    """尝试解析 JSON 输出，失败则返回原字符串。"""
    try:
        # 去掉可能的 markdown 代码块标记
        text = answer.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())
    except Exception:
        return None


def generate(question: str, docs, role: str = "", detail_level: str = "标准",
             json_mode: bool = False, deep_think: bool = False,
             chat_history: List[Dict[str, str]] = None,
             provider_name: str = None, model: str = None):
    """
    主生成函数：将检索结果 + 用户问题发给大模型，生成回答。

    参数：
      question:      用户问题
      docs:          检索到的文档列表
      role:          用户身份（用于注入到 SYSTEM_PROMPT）
      detail_level:  回答详细程度（简洁/标准/详细）
      json_mode:     是否使用结构化 JSON 输出
      deep_think:    是否启用深度思考（CoT）模式
      chat_history:  多轮对话历史（Gradio messages 格式，可选）
      provider_name: 可选，指定 LLM provider
      model:         可选，指定模型名
    """
    # ============================================================
    # Step 1: 检索相关性检查
    # ============================================================
    # 若用户选了具体身份（如“比丘戒”），但检索结果与问题明显不相关，
    # 就清空文档，允许模型用权威经典兜底回答。
    # 这样既避免了误导性的检索结果，又不会直接拒答。
    # ============================================================
    fallback_note = ""
    confidence = "high"
    if role and role not in ("不限", "未指定") and not is_retrieval_relevant(question, docs):
        docs = []
        fallback_note = "（知识库未检索到与问题直接相关的内容，请依据你确知的权威佛教戒律知识谨慎回答，并紧扣上述身份。）"
        confidence = "medium"

    # 空检索时进一步降低置信度
    if not docs and not fallback_note:
        confidence = "low"
    elif not docs and fallback_note:
        confidence = "medium"

    # ============================================================
    # Step 2: 组装上下文 + 截断
    # ============================================================
    context = format_context(docs)
    context = _truncate_context(context)  # 超过 3000 字就截断

    # ============================================================
    # Step 3: 调用大模型
    # ============================================================
    if json_mode:
        system_prompt = JSON_SYSTEM_PROMPT.format(role=role or "未指定")
        user_msg = f"用户身份：{role or '未指定'}\n\n参考资料：\n{context}\n{fallback_note}\n\n问题：{question}"
        answer = _call_llm(system_prompt, user_msg, chat_history=chat_history,
                           provider_name=provider_name, model=model)
        parsed = _try_parse_json(answer)
        if parsed:
            return parsed
        # JSON 解析失败，返回原始文本并标注
        return {"answer": answer, "sources": [], "severity": "未标注", "confidence": "low", "note": "JSON 解析失败，返回原始文本"}

    # Markdown 模式：用主系统提示词
    # 深度思考模式下用 CoT 格式，否则用标准格式
    fmt = _COT_FORMAT.strip() if deep_think else _STANDARD_FORMAT.strip()
    system_prompt = SYSTEM_PROMPT.format(role=role or "未指定", format_section=fmt)
    # 追加详细程度要求
    detail_note = _DETAIL_PROMPT.get(detail_level, _DETAIL_PROMPT["标准"])
    system_prompt += f"\n\n【回答详细程度要求】\n{detail_note}"
    user_msg = f"用户身份：{role or '未指定'}\n\n参考资料：\n{context}\n{fallback_note}\n\n问题：{question}"

    answer = _call_llm(system_prompt, user_msg, chat_history=chat_history,
                       provider_name=provider_name, model=model)

    # ============================================================
    # Step 4: 格式校验
    # ============================================================
    # 检查回答是否包含必要的格式标识（【答】和【依据】）
    # 如果缺少，可能是模型没有遵守指令，重试一次
    # ============================================================
    # 检查回答是否包含必要的格式标识
    # 深度思考模式检查【分析】+【答】+【依据】，标准模式检查【答】+【依据】
    if deep_think:
        has_format = "【分析】" in answer and "【答】" in answer and "【依据】" in answer
    else:
        has_format = _check_format(answer)
    if not has_format:
        correction_prompt = system_prompt + "\n\n注意：上次回答格式不完整，必须包含【答】和【依据】两个部分。"
        answer = _call_llm(correction_prompt, user_msg, chat_history=chat_history,
                           provider_name=provider_name, model=model)

    # ============================================================
    # Step 5: 追加置信度标识并返回
    # ============================================================
    # 先剥离 LLM 可能自行输出的置信度文本，再追加系统的置信度标识
    answer = _strip_llm_confidence(answer)
    return _add_confidence_marker(answer, confidence)


def generate_stream(question: str, docs, role: str = "", detail_level: str = "标准",
                    json_mode: bool = False, deep_think: bool = False,
                    chat_history: List[Dict[str, str]] = None,
                    provider_name: str = None, model: str = None) -> Iterator[str]:
    """
    流式生成函数：逐 token 返回回答，用于 Web 实时展示。

    【小白提示】
    这个函数和 generate() 几乎一样，区别在于：
    - generate() 等模型生成完毕才返回完整回答
    - generate_stream() 每生成一小段就 yield 一次，用户能看到"打字"效果

    这对于等待 5-10 秒才出结果的场景非常重要，
    用户可以实时看到回答在"写出来"，而不是干等白屏。
    """
    # Step 1: 检索相关性检查
    fallback_note = ""
    confidence = "high"
    if role and role not in ("不限", "未指定") and not is_retrieval_relevant(question, docs):
        docs = []
        fallback_note = "（知识库未检索到与问题直接相关的内容，请依据你确知的权威佛教戒律知识谨慎回答，并紧扣上述身份。）"
        confidence = "medium"

    if not docs and not fallback_note:
        confidence = "low"
    elif not docs and fallback_note:
        confidence = "medium"

    # Step 2: 组装上下文
    context = format_context(docs)
    context = _truncate_context(context)

    if json_mode:
        system_prompt = JSON_SYSTEM_PROMPT.format(role=role or "未指定")
        user_msg = f"用户身份：{role or '未指定'}\n\n参考资料：\n{context}\n{fallback_note}\n\n问题：{question}"
    else:
        fmt = _COT_FORMAT.strip() if deep_think else _STANDARD_FORMAT.strip()
        system_prompt = SYSTEM_PROMPT.format(role=role or "未指定", format_section=fmt)
        detail_note = _DETAIL_PROMPT.get(detail_level, _DETAIL_PROMPT["标准"])
        system_prompt += f"\n\n【回答详细程度要求】\n{detail_note}"
        user_msg = f"用户身份：{role or '未指定'}\n\n参考资料：\n{context}\n{fallback_note}\n\n问题：{question}"

    # Step 3: 流式调用 LLM
    accumulated = ""
    try:
        for chunk in stream_with_retry(_get_provider(provider_name=provider_name, model=model),
                                       [{"role": "system", "content": system_prompt}]
                                       + (chat_history or [])[-4:]
                                       + [{"role": "user", "content": user_msg}]):
            accumulated += chunk
            yield accumulated
    except Exception as e:
        yield accumulated + f"\n\n[生成中断：{e}]"
        return

    # 追加置信度标识（先剥离 LLM 可能自行输出的置信度文本）
    accumulated = _strip_llm_confidence(accumulated)
    yield _add_confidence_marker(accumulated, confidence)

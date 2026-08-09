"""
generator.py — 答案生成模块

职责：
  将检索到的戒律文档片段组装为上下文，调用 DeepSeek 大语言模型，
  根据参考资料生成结构化的佛教戒律问答回复。
  强制标注真实经文出处，跨域问题直接拒答。
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

# 详细程度提示词补充
_DETAIL_PROMPT = {
    "简洁": "回答请尽量简洁，直接给出结论和依据，不做过多展开。",
    "标准": "回答需清晰完整，包含结论、适用条件和依据。",
    "详细": "回答需详尽，除结论和依据外，可适当补充背景、开许、例外和相关说明。",
}

# JSON 模式提示词
JSON_SYSTEM_PROMPT = """你是一位严谨的佛教戒律助手，仅回答与佛教戒律相关的问题。

【核心原则】
优先使用提供的参考资料回答；参考资料能完整覆盖问题时，必须严格依据参考资料，不得编造。

【输出格式】
必须按以下 JSON 格式输出，不要包含任何其他内容：
{
  "answer": "正文回答",
  "sources": [
    {"type": "知识库", "role": "居士戒", "severity": "遮戒", "category": "在家生活", "ref": "《增壹阿含经》"},
    {"type": "补充", "ref": "《善生经》"}
  ],
  "severity": "遮戒",
  "confidence": "high|medium|low",
  "note": "可选补充说明"
}

【confidence 说明】
- high：参考资料直接覆盖，答案确定。
- medium：部分参考或权威经典兜底，基本可信。
- low：知识库未覆盖，主要依赖模型记忆，需谨慎。

【越界处理】
若提问与佛教戒律无关，answer 填写："此问题超出戒律问答范围，建议向相关领域咨询。"，confidence 为 low。
"""


def _call_llm(system_prompt: str, user_msg: str, max_retries: int = 2):
    """调用 DeepSeek API，支持失败重试。"""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1,
                timeout=60.0
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
            continue
    raise last_error


def _check_format(answer: str) -> bool:
    """简单校验回答是否包含必要的格式标识。"""
    return "【答】" in answer and "【依据】" in answer


def _truncate_context(context: str, max_chars: int = 3000) -> str:
    """上下文过长时截断，避免超出模型处理上限。"""
    if len(context) <= max_chars:
        return context
    return context[:max_chars] + "\n...（上下文已截断）"
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
2. 来自权威经典兜底的内容，只有当该戒条等级在佛教传统中有公认定论时才标注；否则应说明"该戒条等级未在知识库中明确标注"，不得臆测。

【回答格式】
【答】
（正文，需包含适用条件说明）

【依据】
（必须标注来源；若同时使用了参考资料和外部补充，需分别列出：
  ① 知识库来源：身份·等级·类别·原文出处；
  ② 补充来源：权威经典或法师著作名称。）
"""

def format_context(docs):
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

# 简单停用词集合，用于主题相关性校验
_STOPWORDS = set("的 是 了 在 和 与 或 可以 能 吗 呢 吧 啊 我 你 他 她 它 们 这 那 有 个 为 之 而 以 及 其 该 请 问 如何 什么 哪些 怎么 吗 不 要 会 都 就 都 也 很 但 吗 么 呢 吧".split())

def is_retrieval_relevant(question: str, docs) -> bool:
    """
    简单校验检索结果是否与问题主题相关。
    若问题中的有效用字在参考资料中完全没有出现，认为检索失效。
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
        return answer + f"\n\n---\n\n*置信度：{marker}*"
    return answer


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


def generate(question: str, docs, role: str = "", detail_level: str = "标准", json_mode: bool = False):
    # 若用户指定了具体身份，但检索结果与问题明显不相关，
    # 则清空误导性资料，允许模型从权威经典兜底回答，但仍紧扣当前身份。
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

    context = format_context(docs)
    context = _truncate_context(context)

    if json_mode:
        system_prompt = JSON_SYSTEM_PROMPT.format(role=role or "未指定")
        user_msg = f"用户身份：{role or '未指定'}\n\n参考资料：\n{context}\n{fallback_note}\n\n问题：{question}"
        answer = _call_llm(system_prompt, user_msg)
        parsed = _try_parse_json(answer)
        if parsed:
            return parsed
        # JSON 解析失败，返回原始文本并标注
        return {"answer": answer, "sources": [], "severity": "未标注", "confidence": "low", "note": "JSON 解析失败，返回原始文本"}

    # Markdown 模式
    system_prompt = SYSTEM_PROMPT.format(role=role or "未指定")
    detail_note = _DETAIL_PROMPT.get(detail_level, _DETAIL_PROMPT["标准"])
    system_prompt += f"\n\n【回答详细程度要求】\n{detail_note}"
    user_msg = f"用户身份：{role or '未指定'}\n\n参考资料：\n{context}\n{fallback_note}\n\n问题：{question}"

    answer = _call_llm(system_prompt, user_msg)

    # 格式校验：若缺少【答】或【依据】，尝试再请求一次
    if not _check_format(answer):
        correction_prompt = system_prompt + "\n\n注意：上次回答格式不完整，必须包含【答】和【依据】两个部分。"
        answer = _call_llm(correction_prompt, user_msg)

    return _add_confidence_marker(answer, confidence)

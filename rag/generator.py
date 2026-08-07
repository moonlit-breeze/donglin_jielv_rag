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

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
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
1. 必须区分戒条的适用条件：该戒针对谁（已受戒者 / 未受戒者、居士 / 沙弥 / 比丘）？
2. 必须区分戒条性质：性戒（根本戒）与遮戒（防范戒）的约束力度不同，遮戒对未受戒者不构成约束。
3. 若参考资料中包含开许、例外、不同层次的说明（如「未受戒者无罪过」「方便说」等），必须完整体现，不得省略。
4. 不得将「已受戒者的戒条要求」笼统地回答为所有人都必须遵守。

【回答格式】
【答】
（正文，需包含适用条件说明）

【依据】
（必须标注来源；若同时使用了参考资料和外部补充，需分别列出：
  ① 知识库来源：身份·类别·原文出处；
  ② 补充来源：权威经典或法师著作名称。）
"""

def format_context(docs):
    parts = []
    for doc in docs:
        role = doc.metadata.get("role", "")
        source = doc.metadata.get("source", "")
        category = doc.metadata.get("category", "")
        # 构造来源标注：身份 + 类别 + 经文出处
        source_label = f"（{role}"
        if category:
            source_label += f"·{category}"
        source_label += "）"
        source_ref = f"\n[出处：{source}]" if source else ""
        parts.append(f"{source_label}{doc.page_content}{source_ref}")
    return "\n\n".join(parts)

def generate(question: str, docs):
    context = format_context(docs)
    user_msg = f"参考资料：\n{context}\n\n问题：{question}"

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.1,
        timeout=60.0
    )
    return resp.choices[0].message.content

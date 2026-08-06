"""
generator.py — 答案生成模块

职责：
  将检索到的戒律文档片段组装为上下文，调用 DeepSeek 大语言模型，
  根据参考资料生成结构化的佛教戒律问答回复。
"""

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
SYSTEM_PROMPT = """你是一位严谨的佛教戒律助手。
请严格根据参考资料回答问题，不得编造经文或戒律。
若参考资料中没有相关信息，请回答：『未在现有戒律资料中找到明确依据。』

回答原则（非常重要）：
1. 必须区分戒条的适用条件：该戒针对谁（已受戒者 / 未受戒者、居士 / 沙弥 / 比丘）？
2. 必须区分戒条性质：性戒（根本戒）与遮戒（防范戒）的约束力度不同，遮戒对未受戒者不构成约束。
3. 若参考资料中包含开许、例外、不同层次的说明（如「未受戒者无罪过」「方便说」等），必须在回答中完整体现，不得省略。
4. 不得将「已受戒者的戒条要求」笼统地回答为所有人都必须遵守，应明确告知提问者该戒的前提条件。

回答格式：
【答】
（正文，需包含适用条件说明）

【依据】
（必须标注出处：来自哪部戒律的第几条，例如「出自《居士戒》第12条」。若参考资料中包含条目编号，请务必引用。）
"""

def format_context(docs):
    parts = []
    for doc in docs:
        role = doc.metadata.get("role", "")
        line_no = doc.metadata.get("line", "")
        label = f"（{role} 第{line_no}条）" if line_no != "" else f"（{role}）"
        parts.append(f"{label}{doc.page_content}")
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
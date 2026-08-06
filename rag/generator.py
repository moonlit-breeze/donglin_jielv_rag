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

回答格式：
【答】
（正文）

【依据】
（列出参考资料来源）
"""

def format_context(docs):
    parts = []
    for doc in docs:
        role = doc.metadata.get("role", "")
        parts.append(f"（{role}）{doc.page_content}")
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
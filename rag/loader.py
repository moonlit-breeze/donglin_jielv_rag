"""
loader.py — 知识库加载模块

职责：
  从 data/knowledge_base.json 加载结构化戒律数据，转换为 LangChain Document 对象，
  携带 role、domain、source、category 等元数据，为向量检索提供丰富过滤维度。
"""

import json
from langchain_core.documents import Document

# 领域标识，用于检索时过滤，防止跨域回答
DOMAIN = "jielv"

def load_knowledge_base(file_path: str = "data/knowledge_base.json"):
    """加载 JSON 知识库，返回 Document 列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    docs = []
    for idx, entry in enumerate(entries):
        content = entry.get("content", "")
        if not content:
            continue
        doc = Document(
            page_content=content,
            metadata={
                "role": entry.get("role", "通用"),
                "domain": DOMAIN,
                "source": entry.get("source", ""),
                "category": entry.get("category", ""),
                "index": idx,
            }
        )
        docs.append(doc)

    return docs

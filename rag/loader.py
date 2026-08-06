"""
loader.py — 文本加载与切分模块

职责：
  1. 读取原始戒律文本文件，解析每行的身份标签（如比丘戒、沙弥戒、居士戒）与正文内容；
  2. 将文本封装为 LangChain Document 对象，并携带来源、行号、身份等元数据；
  3. 使用 RecursiveCharacterTextSplitter 对过长文本进行切分，为后续向量化做准备。
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

def parse_line(line: str) -> dict:
    """从文本行提取身份和正文"""
    m = re.match(r'【(.+?)】(.*)', line.strip())
    if m:
        role = m.group(1)      # 比丘戒 / 沙弥戒 / 居士戒
        content = m.group(2)
    else:
        role = "未知"
        content = line.strip()
    return {"role": role, "content": content}

def load_and_split(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    docs = []
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        meta = parse_line(line)
        doc = Document(
            page_content=meta["content"],
            metadata={
                "source": file_path,
                "line": idx,
                "role": meta["role"]  # 关键：身份标签
            }
        )
        docs.append(doc)

    # 切分（戒律本来是一条条，这里几乎不切，只防太长）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n", "。", "；"]
    )

    return splitter.split_documents(docs)
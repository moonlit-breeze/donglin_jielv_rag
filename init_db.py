"""
初始化向量数据库。

从 data/knowledge_base.json 加载结构化数据，按身份分层建库。
"通用"条目（如戒律总览、佛教戒律传承）会被写入所有身份库，
确保任何身份下都能检索到通用知识。
"""

import os

# 必须在导入任何 NLP 库之前设置离线模式，避免联网请求
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from rag.loader import load_knowledge_base
from rag.vector_store import create_vectorstore_for_role, ROLE_DB_MAP

if __name__ == "__main__":
    print("正在加载 data/knowledge_base.json ...")
    docs = load_knowledge_base()
    print(f"共加载 {len(docs)} 条知识条目")

    # 按 role 分组，"通用" 条目写入所有身份库
    groups = {}
    for doc in docs:
        role = doc.metadata.get("role", "通用")
        if role == "通用":
            for r in ROLE_DB_MAP:
                groups.setdefault(r, []).append(doc)
        else:
            groups.setdefault(role, []).append(doc)

    for role, role_docs in groups.items():
        if role in ROLE_DB_MAP:
            print(f"正在为「{role}」创建向量库（{len(role_docs)} 条）→ {ROLE_DB_MAP[role]}")
            create_vectorstore_for_role(role, role_docs)
        else:
            print(f"跳过未知身份「{role}」，不在 ROLE_DB_MAP 中")

    print("\n向量库创建完成！")
    print("各身份库目录：")
    for role, path in ROLE_DB_MAP.items():
        count = len(groups.get(role, []))
        print(f"  {role} → {path}（{count} 条）")

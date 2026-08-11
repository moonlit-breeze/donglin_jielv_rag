"""
初始化向量数据库。

从 data/knowledge_base.json 加载结构化数据，按身份分层建库。
“通用”条目（如戒律总览、佛教戒律传承）会被写入所有身份库，
确保任何身份下都能检索到通用知识。

【小白导读】
  这个脚本是「建库工具」，运行一次就能把所有知识存入向量数据库。

  执行流程：
    1. 读取 knowledge_base.json（由 ingest.py 生成的结构化数据）
    2. 按 role 分组（居士戒 / 沙弥戒 / 比丘戒 / 通用）
    3. “通用”条目（如“什么是五戒”）写入所有身份库，这样不管哪个身份问都能查到
    4. 每个身份单独建库，存入对应条目

  运行方式：
    python init_db.py
"""

import os

# ============================================================
# 离线模式设置（必须在导入任何 NLP 库之前！）
# 原理同 vector_store.py，确保 HuggingFace 不会联网
# ============================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from rag.loader import load_knowledge_base
from rag.vector_store import create_vectorstore_for_role, ROLE_DB_MAP

if __name__ == "__main__":
    # Step 1: 加载知识库
    print("正在加载 data/knowledge_base.json ...")
    docs = load_knowledge_base()
    print(f"共加载 {len(docs)} 条知识条目")

    # Step 2: 按 role 分组
    # groups 是一个字典，key 是身份，value 是该身份的文档列表
    # 特殊处理：“通用”条目会被复制到所有身份库中
    groups = {}
    for doc in docs:
        role = doc.metadata.get("role", "通用")
        if role == "通用":
            # “通用”条目（如“什么是五戒”）对所有身份都有参考价值，
            # 所以写入每个身份的向量库
            for r in ROLE_DB_MAP:
                groups.setdefault(r, []).append(doc)
        else:
            # 特定身份的条目只写入对应的库
            groups.setdefault(role, []).append(doc)

    # Step 3: 为每个身份创建向量库
    for role, role_docs in groups.items():
        if role in ROLE_DB_MAP:
            print(f"正在为「{role}」创建向量库（{len(role_docs)} 条）→ {ROLE_DB_MAP[role]}")
            # create_vectorstore_for_role 会把所有文档转成向量并存入 Chroma
            create_vectorstore_for_role(role, role_docs)
        else:
            print(f"跳过未知身份「{role}」，不在 ROLE_DB_MAP 中")

    # 完成，打印汇总信息
    print("\n向量库创建完成！")
    print("各身份库目录：")
    for role, path in ROLE_DB_MAP.items():
        count = len(groups.get(role, []))
        print(f"  {role} → {path}（{count} 条）")

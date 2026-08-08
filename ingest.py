"""
ingest.py — PDF 戒律文档导入脚本

使用方式：
  python -m rag.ingest                        # 默认导入 data/jielv.pdf
  python -m rag.ingest data/新戒律文档.pdf     # 指定 PDF 文件路径

流程：
  Step 1: 读取 PDF，提取文本并转换为 JSON 格式，覆盖写入 data/knowledge_base.json
  Step 2: 加载 JSON，按身份分组，清空旧向量库后重建
"""

import os

# 必须在导入任何 NLP 库之前设置离线模式，避免联网请求
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import shutil
from pathlib import Path

from rag.pdf_loader import load_pdf_to_json, save_json_data
from rag.loader import load_knowledge_base
from rag.vector_store import create_vectorstore_for_role, ROLE_DB_MAP

# 知识库 JSON 输出路径（覆盖写入）
KNOWLEDGE_BASE_PATH = "data/knowledge_base.json"

def main():
    # 解析命令行参数，支持指定 PDF 路径
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = Path("./data/jielv.pdf")

    if not pdf_path.exists():
        print(f"[错误] PDF 文件不存在：{pdf_path}")
        print("用法：python -m rag.ingest <PDF文件路径>")
        sys.exit(1)

    print(f"目标 PDF：{pdf_path}")
    print(f"知识库输出：{KNOWLEDGE_BASE_PATH}")

    # ── Step 1: PDF → JSON ──
    print("\n" + "=" * 40)
    print("Step 1: 从 PDF 提取文本并转换为 JSON 格式...")
    json_data = load_pdf_to_json(str(pdf_path))

    if not json_data:
        print("[错误] PDF 未提取到任何有效内容，终止导入")
        sys.exit(1)

    # 覆盖写入知识库 JSON
    save_json_data(json_data, KNOWLEDGE_BASE_PATH)

    # ── Step 2: JSON → 向量库（覆盖重建）──
    print("\n" + "=" * 40)
    print("Step 2: 加载 JSON 并重建向量库...")

    # 清空旧的向量库目录，避免残留数据
    for role, db_path in ROLE_DB_MAP.items():
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print(f"  已清空旧库：{db_path}")

    # 加载合并后的 JSON（此时 knowledge_base.json 已被覆盖）
    docs = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    print(f"  共加载 {len(docs)} 条知识条目")

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
            print(f"  正在为「{role}」创建向量库（{len(role_docs)} 条）→ {ROLE_DB_MAP[role]}")
            create_vectorstore_for_role(role, role_docs)
        else:
            print(f"  跳过未知身份「{role}」，不在 ROLE_DB_MAP 中")

    # ── 完成 ──
    print("\n" + "=" * 40)
    print("导入完成！各身份库：")
    for role, path in ROLE_DB_MAP.items():
        count = len(groups.get(role, []))
        print(f"  {role} → {path}（{count} 条）")


if __name__ == "__main__":
    main()

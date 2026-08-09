"""
ingest.py — PDF 戒律文档导入脚本

使用方式：
  python ingest.py                              # 默认导入 data/jielv.pdf，覆盖重建
  python ingest.py data/新戒律文档.pdf           # 指定 PDF 文件路径，覆盖重建
  python ingest.py --merge data/新戒律文档.pdf   # 与现有 knowledge_base.json 合并追加
  python ingest.py --preview data/新戒律文档.pdf # 预览前 3 条提取结果，不保存不建库
  python ingest.py --no-chunk data/新戒律文档.pdf # 关闭智能分块

流程：
  Step 1: 读取 PDF，提取文本并转换为 JSON 格式
  Step 2: 根据模式选择：
          - 覆盖模式（默认）：覆盖写入 data/knowledge_base.json，备份旧文件
          - 合并模式：与现有 knowledge_base.json 合并，去重后保存
          - 预览模式：只打印前 3 条，不保存不建库
  Step 3: 加载 JSON，按身份分组，清空旧向量库后重建
"""

import os

# 必须在导入任何 NLP 库之前设置离线模式，避免联网请求
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import shutil
from pathlib import Path

from rag.pdf_loader import load_pdf_to_json, save_json_data, load_json_data, validate_records
from rag.loader import load_knowledge_base
from rag.vector_store import create_vectorstore_for_role, ROLE_DB_MAP

# 知识库 JSON 输出路径
KNOWLEDGE_BASE_PATH = Path("data/knowledge_base.json")


def _backup_existing_json():
    """如果已有知识库 JSON，则备份为 .bak 文件。"""
    if KNOWLEDGE_BASE_PATH.exists():
        backup_path = KNOWLEDGE_BASE_PATH.with_suffix(".json.bak")
        shutil.copy(KNOWLEDGE_BASE_PATH, backup_path)
        print(f"[INFO] 已备份旧知识库：{backup_path}")


def _merge_with_existing(new_data: list) -> list:
    """将新数据与现有 knowledge_base.json 合并，并按 content 去重。"""
    if not KNOWLEDGE_BASE_PATH.exists():
        return new_data

    existing = load_json_data(str(KNOWLEDGE_BASE_PATH))
    print(f"[INFO] 现有知识库共 {len(existing)} 条，新导入 {len(new_data)} 条")

    seen = set()
    merged = []
    for record in existing + new_data:
        key = (record.get("role"), record.get("content", "").strip())
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)

    print(f"[INFO] 合并去重后共 {len(merged)} 条")
    return merged


def main():
    # 解析命令行参数
    merge_mode = False
    preview_mode = False
    smart_chunk = True
    args = sys.argv[1:]
    if "--merge" in args:
        merge_mode = True
        args.remove("--merge")
    if "--preview" in args:
        preview_mode = True
        args.remove("--preview")
    if "--no-chunk" in args:
        smart_chunk = False
        args.remove("--no-chunk")

    if args:
        pdf_path = Path(args[0])
    else:
        pdf_path = Path("./data/jielv.pdf")

    if not pdf_path.exists():
        print(f"[错误] PDF 文件不存在：{pdf_path}")
        print("用法：python ingest.py [--merge] [--preview] [--no-chunk] <PDF文件路径>")
        sys.exit(1)

    print(f"目标 PDF：{pdf_path}")
    print(f"知识库输出：{KNOWLEDGE_BASE_PATH}")
    print(f"导入模式：{'合并追加' if merge_mode else '覆盖重建'}")
    print(f"智能分块：{'开启' if smart_chunk else '关闭'}")

    # ── Step 1: PDF → JSON ──
    print("\n" + "=" * 40)
    print("Step 1: 从 PDF 提取文本并转换为 JSON 格式...")
    json_data = load_pdf_to_json(str(pdf_path), smart_chunk=smart_chunk, preview=preview_mode)

    if not json_data:
        print("[错误] PDF 未提取到任何有效内容，终止导入")
        sys.exit(1)

    # 预览模式直接结束
    if preview_mode:
        print("\n" + "=" * 40)
        print("预览结果：")
        for idx, record in enumerate(json_data, start=1):
            print(f"\n[{idx}] role={record.get('role')} source={record.get('source')}")
            print(record.get('content', '')[:300] + "...")
        print("\n预览结束，未保存未建库。")
        return

    # 校验新数据
    errors = validate_records(json_data)
    if errors:
        print("[错误] 新数据 Schema 校验失败：")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    # ── Step 2: 保存知识库 JSON ──
    print("\n" + "=" * 40)
    print("Step 2: 保存知识库 JSON...")

    if merge_mode:
        final_data = _merge_with_existing(json_data)
    else:
        _backup_existing_json()
        final_data = json_data

    # 最终校验并保存
    errors = validate_records(final_data)
    if errors:
        print("[错误] 最终知识库 Schema 校验失败：")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    save_json_data(final_data, str(KNOWLEDGE_BASE_PATH))

    # ── Step 3: JSON → 向量库（覆盖重建）──
    print("\n" + "=" * 40)
    print("Step 3: 加载 JSON 并重建向量库...")

    # 清空旧的向量库目录，避免残留数据
    for role, db_path in ROLE_DB_MAP.items():
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print(f"  已清空旧库：{db_path}")

    # 加载合并后的 JSON
    docs = load_knowledge_base(str(KNOWLEDGE_BASE_PATH))
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

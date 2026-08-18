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

【小白导读】
  这个脚本是系统的「数据管道」，负责把原始文档变成向量库。
  你可以把它理解为一个工厂流水线：

  PDF/TXT 原始文件  →  文本提取  →  结构化 JSON  →  向量化  →  Chroma 数据库

  重点关注的几个问题：
  1. 怎么从 PDF 中判断哪段文字属于“居士戒”？（看 _extract_role_from_text）
  2. 为什么长文本需要切分成小块？（看 _chunk_by_sentence，太长会导致检索发散）
  3. 怎么从文本中自动提取出处？（看 _extract_source_from_text）
"""

import os

# 必须在导入任何 NLP 库之前设置离线模式，避免联网请求
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import shutil
import re
from pathlib import Path

from rag.pdf_loader import load_pdf_to_json, save_json_data, load_json_data, validate_records
from rag.loader import load_knowledge_base
from rag.vector_store import create_vectorstore_for_role, ROLE_DB_MAP
from rag.retriever import invalidate_all_caches

# 知识库 JSON 输出路径
KNOWLEDGE_BASE_PATH = Path("data/knowledge_base.json")

# 合法身份标签
VALID_ROLES = {"比丘戒", "沙弥戒", "居士戒", "比丘尼戒", "通用"}


def _extract_role_from_text(text: str) -> str:
    """
    从文本开头提取身份标签，如【比丘戒】。

    【小白提示】
    PDF 中的每条记录可能以【居士戒】【比丘戒】等标签开头，
    这个函数就是用正则表达式去匹配这个标签。
    如果没匹配到，就默认归为“通用”。
    """
    match = re.match(r"【\s*(比丘戒|沙弥戒|居士戒|比丘尼戒|通用)\s*】", text)
    if match:
        return match.group(1)
    return "通用"


def _extract_source_from_text(text: str, role: str) -> str:
    """
    从文本中启发式提取出处。
    优先级：书名号《...》 > 法师/大师开示 > 身份默认。
    """
    # 书名号
    m = re.search(r"《([^》]+)》", text)
    if m:
        return f"《{m.group(1)}》"

    # 法师/大师开示
    m = re.search(r"(大安法师|印光大师|慧远大师|智者大师|道宣律师)[^，。]*开示", text)
    if m:
        return m.group(0)

    # 身份默认
    if role in {"比丘戒", "沙弥戒", "居士戒", "比丘尼戒"}:
        return f"《四分律》·{role}本"

    return "未知典籍"


def _chunk_by_sentence(text: str, role: str, source: str, category: str,
                       min_size: int = 40, max_size: int = 160) -> list:
    """
    按句末标点对长文本做语义分块。
    目标：每条 chunk 只覆盖 1-3 个完整句子，长度在 [min_size, max_size] 之间，
    既保证句子完整，又避免 overview 式长段落稀释检索信号。

    【小白提示】为什么要分块？
    想象有一段 500 字的概述，里面提到了杀生、偷盗、饮酒等多个话题。
    如果不分块，用户问“可以喝酒吗”时，整段 500 字都会被检索出来，
    但其中只有很少一部分和饮酒相关，这会“稀释”检索信号。
    分块后，每块只含 1-3 个完整句子，检索时更容易命中最相关的部分。
    """
    # 按。！？切分，保留标点
    parts = re.split(r"([。！？])", text)
    sentences = []
    i = 0
    while i < len(parts):
        s = parts[i]
        if i + 1 < len(parts) and parts[i + 1] in "。！？":
            s += parts[i + 1]
            i += 2
        else:
            i += 1
        s = s.strip()
        if s:
            sentences.append(s)

    if not sentences:
        return []

    chunks = []
    current = sentences[0]
    for s in sentences[1:]:
        # 如果当前 chunk 已够长，或加入后会太长，则先保存当前 chunk
        if len(current) >= max_size or len(current) + len(s) > max_size:
            chunks.append({
                "role": role,
                "domain": "jielv",
                "source": source,
                "content": current,
                "category": category,
                "severity": "未标注",
            })
            current = s
        else:
            current += s

    # 处理最后一段
    if current:
        if len(current) < min_size and chunks:
            # 太短则合并到前一段，但不超过上限
            if len(chunks[-1]["content"]) + len(current) <= max_size + 20:
                chunks[-1]["content"] += current
            else:
                chunks.append({
                    "role": role,
                    "domain": "jielv",
                    "source": source,
                    "content": current,
                    "category": category,
                    "severity": "未标注",
                })
        else:
            chunks.append({
                "role": role,
                "domain": "jielv",
                "source": source,
                "content": current,
                "category": category,
                "severity": "未标注",
            })

    return chunks


def _is_general_content(text: str) -> bool:
    """
    判断一段内容是否属于通用/概述性内容，适合写入所有身份库。
    启发式规则：同时提到多个身份，或出现明显的体系/概述性关键词。
    """
    identity_keywords = ["比丘", "沙弥", "居士", "比丘尼"]
    matched = sum(1 for kw in identity_keywords if kw in text)
    if matched >= 2:
        return True

    overview_markers = ["戒律体系", "由浅入深", "佛教戒律总览", "南山律学"]
    if any(marker in text for marker in overview_markers):
        return True

    return False


def load_txt_to_json(txt_path: str) -> list:
    """
    读取 jielv.txt 格式的文本文件，每行一条或多条戒律记录，转换为 JSON 格式。
    行首必须有【身份】标签，如：
      【居士戒】不杀生，不起杀害之心。
    同一行内若出现多个【身份】标签，会按标签拆分为多条独立记录，
    长段落还会按句末标点进一步切分，既保留完整句子，又避免 overview 条目过长导致检索发散。
    """
    records = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 按行内【身份】标签拆分，既保留完整句子，又避免 overview 过长
            segments = re.split(r"(?=【\s*(?:比丘戒|沙弥戒|居士戒|比丘尼戒|通用)\s*】)", line)
            current_role = None
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue
                # 提取这一段的身份
                role_match = re.match(r"【\s*(比丘戒|沙弥戒|居士戒|比丘尼戒|通用)\s*】", seg)
                if role_match:
                    current_role = role_match.group(1)
                    content = re.sub(r"^【\s*(?:比丘戒|沙弥戒|居士戒|比丘尼戒|通用)\s*】", "", seg).strip()
                else:
                    if current_role is None:
                        current_role = "通用"
                    content = seg

                if not content:
                    continue

                # 判断是否为通用/概述性内容，是则写入所有身份库
                effective_role = "通用" if _is_general_content(content) else current_role
                source = _extract_source_from_text(content, effective_role)

                # 长段落按句子切分，保证检索聚焦且句子完整
                if len(content) > 160:
                    chunks = _chunk_by_sentence(content, effective_role, source, effective_role)
                    records.extend(chunks)
                else:
                    records.append({
                        "role": effective_role,
                        "domain": "jielv",
                        "source": source,
                        "content": content,
                        "category": effective_role,
                        "severity": "未标注",
                    })
    print(f"[INFO] 从 {txt_path} 读取了 {len(records)} 条记录")
    return records


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
    # ============================================================
    # 命令行参数解析
    # ============================================================
    # 支持 3 种模式：
    #   --merge   : 合并追加（新数据和旧数据合并去重）
    #   --preview : 预览模式（只看不存，用于检查提取效果）
    #   --no-chunk: 关闭智能分块（整段作为一个条目）
    # ============================================================
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
        file_path = Path(args[0])
    else:
        file_path = Path("./data/jielv.pdf")

    if not file_path.exists():
        print(f"[错误] 文件不存在：{file_path}")
        print("用法：python ingest.py [--merge] [--preview] [--no-chunk] <PDF/TXT文件路径>")
        sys.exit(1)

    print(f"目标文件：{file_path}")
    print(f"知识库输出：{KNOWLEDGE_BASE_PATH}")
    print(f"导入模式：{'合并追加' if merge_mode else '覆盖重建'}")
    print(f"智能分块：{'开启' if smart_chunk else '关闭'}")

    # ── Step 1: 读取文件 → JSON ──
    print("\n" + "=" * 40)
    if file_path.suffix.lower() == ".txt":
        print("Step 1: 从 TXT 文本提取记录...")
        json_data = load_txt_to_json(str(file_path))
    else:
        print("Step 1: 从 PDF 提取文本并转换为 JSON 格式...")
        json_data = load_pdf_to_json(str(file_path), smart_chunk=smart_chunk, preview=preview_mode)

    if not json_data:
        print("[错误] PDF 未提取到任何有效内容，终止导入")
        sys.exit(1)

    # 预览模式直接结束
    if preview_mode:
        print("\n" + "=" * 40)
        print("预览结果：")
        for idx, record in enumerate(json_data, start=1):
            try:
                print(f"\n[{idx}] role={record.get('role')} source={record.get('source')}")
                print(record.get('content', '')[:300] + "...")
            except UnicodeEncodeError:
                print(f"\n[{idx}] role={record.get('role')} source={record.get('source')}")
                print(record.get('content', '')[:300].encode("utf-8", errors="ignore").decode("utf-8") + "...")
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

    # 知识库已重建，清空检索缓存，避免旧数据脏读
    invalidate_all_caches()
    print("已清空检索缓存（结果缓存/精排缓存/改写缓存/倒排索引）")

    # ── 完成 ──
    print("\n" + "=" * 40)
    print("导入完成！各身份库：")
    for role, path in ROLE_DB_MAP.items():
        count = len(groups.get(role, []))
        print(f"  {role} → {path}（{count} 条）")


if __name__ == "__main__":
    main()

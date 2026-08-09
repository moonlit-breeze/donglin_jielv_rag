"""
pdf_loader.py — PDF 戒律文档转换模块

职责：
  将 PDF 格式的戒律文档转换为项目统一使用的 JSON 结构化数据。
  输出格式与 knowledge_base.json 保持一致，每条记录包含：
    - role:     身份标签（比丘戒 / 沙弥戒 / 居士戒 / 比丘尼戒 / 通用）
    - domain:   领域标签，固定为 "jielv"
    - content:  正文内容
    - source:   出处/典籍名称
    - category: 内容分类
    - severity: 戒律等级（根本戒 / 遮戒 / 轻戒 / 重戒 / 未标注）

  实现要点：
    1. 按页读取 PDF 文本；
    2. 根据身份标签把连续页面归并为同一章节；
    3. 按语义边界（戒条编号、标题、空行）对大段内容做智能分块；
    4. 自动提取戒条编号、出处等元数据；
    5. 对输出记录做 Schema 校验，确保字段完整。
"""

import json
import re
from typing import List, Dict, Tuple

from pypdf import PdfReader

# 知识库单条记录的必填字段
REQUIRED_FIELDS = {"role", "domain", "content", "source", "category", "severity"}

# 合法身份标签
VALID_ROLES = {"比丘戒", "沙弥戒", "居士戒", "比丘尼戒", "通用"}

# 戒条编号模式：如 "第一条"、"第1条"、"（一）"、"1."
RULE_NUMBER_PATTERNS = [
    r"第[一二三四五六七八九十百千万0-9]+条",
    r"[（(][一二三四五六七八九十]+[）)]",
    r"^\s*[0-9]+[\.、]\s*",
]

# 标题模式：如 "【饮酒戒】"、"一、不杀生"
HEADING_PATTERNS = [
    r"【[^】]+】",
    r"^[一二三四五六七八九十]+[、.]\s*[^\n]{2,20}$",
]


def validate_records(records: List[Dict]) -> List[str]:
    """
    校验 JSON 记录是否符合知识库 Schema。
    返回错误信息列表；为空表示校验通过。
    """
    errors = []
    for idx, record in enumerate(records):
        missing = REQUIRED_FIELDS - set(record.keys())
        if missing:
            errors.append(f"第 {idx+1} 条记录缺少字段：{', '.join(sorted(missing))}")
        role = record.get("role")
        if role and role not in VALID_ROLES:
            errors.append(f"第 {idx+1} 条记录身份 '{role}' 不合法，支持：{', '.join(sorted(VALID_ROLES))}")
        content = record.get("content", "")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"第 {idx+1} 条记录 content 为空或不是字符串")
    return errors


def normalize_record(record: Dict) -> Dict:
    """补全记录中可能缺失的默认字段。"""
    defaults = {
        "domain": "jielv",
        "source": "未知典籍",
        "category": record.get("role", "通用"),
        "severity": "未标注",
    }
    for key, value in defaults.items():
        if key not in record or not record[key]:
            record[key] = value
    return record


def extract_metadata_from_text(text: str) -> Tuple[str, str]:
    """
    从文本中提取角色（role）和出处（source）。

    注意：这里是基于示例规则做简单匹配，实际使用时应根据 PDF 的排版调整：
      - role:  匹配 【比丘戒】【沙弥戒】【居士戒】【比丘尼戒】 等身份标签
      - source: 匹配 《书名》 形式的书名号
    """
    role = "未知"
    source = "未知典籍"

    # 在前 200 个字符内搜索身份标签，方括号可有可无
    role_match = re.search(r'【?(比丘戒|沙弥戒|居士戒|比丘尼戒)】?', text[:200])
    if role_match:
        role = role_match.group(1)

    # 在前 500 个字符内搜索书名号《...》
    source_match = re.search(r'《([^》]+)》', text[:500])
    if source_match:
        source = f"《{source_match.group(1)}》"

    return role, source


def _extract_rule_number(text: str) -> str:
    """从文本开头提取戒条编号，如'第一条'、'（一）'。"""
    for pattern in RULE_NUMBER_PATTERNS:
        match = re.search(pattern, text[:50])
        if match:
            return match.group(0).strip()
    return ""


def _is_heading(line: str) -> bool:
    """判断一行是否是标题/分块边界。"""
    line = line.strip()
    if not line:
        return False
    for pattern in HEADING_PATTERNS + RULE_NUMBER_PATTERNS:
        if re.match(pattern, line):
            return True
    return False


def _semantic_chunk(content: str, role: str, source: str, category: str, max_chunk_size: int = 800):
    """
    对大段内容按语义边界切分。
    返回 List[Dict]，每个元素是一条知识库记录。
    """
    lines = content.split("\n")
    chunks = []
    current_lines = []
    current_heading = ""

    def flush():
        if current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                rule_number = _extract_rule_number(current_heading + text)
                chunk_source = source
                if rule_number:
                    chunk_source = f"{source} {rule_number}" if source != "未知典籍" else rule_number
                chunks.append({
                    "role": role,
                    "domain": "jielv",
                    "source": chunk_source,
                    "content": text,
                    "category": category,
                    "severity": "未标注",
                })
        current_lines.clear()

    for line in lines:
        if _is_heading(line):
            flush()
            current_heading = line.strip()
            current_lines.append(line)
        else:
            # 如果当前块已较长，且遇到空行，则尝试切分
            if len("\n".join(current_lines)) > max_chunk_size and not line.strip():
                flush()
                current_heading = ""
            current_lines.append(line)

    flush()
    return chunks


def load_pdf_to_json(pdf_path: str, smart_chunk: bool = True, preview: bool = False) -> List[Dict]:
    """
    读取 PDF，提取文本，并转换为统一的 JSON 格式。

    参数：
      smart_chunk: 是否按语义边界对大段内容做智能分块
      preview: 是否只返回前 3 条预览，不保存

    流程：
      1. 逐页读取 PDF 文本；
      2. 每页提取身份（role）和出处（source）；
      3. 若当前页身份与正在累积的身份不同，则先保存上一身份的内容，
         再以当前页开始新身份的累积；
      4. 循环结束后保存最后累积的章节；
      5. 对大段内容做语义分块（可选）；
      6. 对结果做 Schema 校验并补全默认字段。
    """
    reader = PdfReader(pdf_path)
    documents: List[Dict] = []

    current_role = "通用"        # 当前正在累积的身份
    current_source = "未知典籍"  # 当前正在累积的出处
    current_page_content = ""    # 当前身份下已累积的文本

    for page_num, page in enumerate(reader.pages, start=1):
        # 提取当前页文本
        text = page.extract_text()
        if not text:
            continue

        # 提取当前页的身份和出处
        new_role, new_source = extract_metadata_from_text(text)
        if new_role == "未知":
            new_role = current_role

        # 情况1：检测到新身份，且缓冲区里已有上一身份的内容
        #        → 保存上一身份的内容，然后以当前页开启新身份
        if new_role != current_role and current_page_content:
            documents.append({
                "role": current_role,
                "domain": "jielv",
                "source": current_source,
                "content": current_page_content.strip(),
                "category": current_role,
                "severity": "未标注"
            })
            # 重置缓冲区，用当前页开始新章节
            current_page_content = text
            current_role = new_role
            if new_source != "未知典籍":
                current_source = new_source

        # 情况2：检测到身份，且缓冲区为空（通常是第一页，或上一章刚好被保存完）
        #        → 直接用当前页开启新身份
        elif not current_page_content:
            current_page_content = text
            current_role = new_role
            if new_source != "未知典籍":
                current_source = new_source

        # 情况3：当前页没有身份标记，或身份与当前累积身份相同
        #        → 继续追加到当前章节
        else:
            current_page_content += "\n" + text
            # 如果当前页本身有出处信息，也更新一下
            if new_source != "未知典籍":
                current_source = new_source

    # 循环结束后，若缓冲区还有内容，保存为最后一个章节
    if current_page_content:
        documents.append({
            "role": current_role,
            "domain": "jielv",
            "source": current_source,
            "content": current_page_content.strip(),
            "category": current_role,
            "severity": "未标注"
        })

    # 智能分块：把大段内容按标题/戒条编号切小
    if smart_chunk:
        chunked_documents = []
        for doc in documents:
            content = doc["content"]
            # 只有内容较长时才分块
            if len(content) > 800:
                chunks = _semantic_chunk(
                    content,
                    role=doc["role"],
                    source=doc["source"],
                    category=doc["category"],
                )
                chunked_documents.extend(chunks)
            else:
                chunked_documents.append(doc)
        documents = chunked_documents

    # 校验并补全默认字段
    errors = validate_records(documents)
    if errors:
        raise ValueError("PDF 转换结果 Schema 校验失败：\n" + "\n".join(errors))

    documents = [normalize_record(r) for r in documents]

    if preview:
        preview_docs = documents[:3]
        print(f"[INFO] 预览模式：从 {pdf_path} 提取了 {len(documents)} 条数据，展示前 {len(preview_docs)} 条")
        return preview_docs

    print(f"[INFO] 从 {pdf_path} 读取并转换了 {len(documents)} 条 JSON 数据")
    return documents


def save_json_data(data: List[Dict], output_path: str):
    """将数据保存为 JSON 文件。"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 保存 JSON 数据到 {output_path}")


def load_json_data(json_path: str) -> List[Dict]:
    """读取 JSON 文件并返回数据。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[INFO] 从 {json_path} 读取了 {len(data)} 条 JSON 数据")
    return data

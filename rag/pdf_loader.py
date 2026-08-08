"""
pdf_loader.py — PDF 戒律文档转换模块

职责：
  将 PDF 格式的戒律文档转换为项目统一使用的 JSON 结构化数据。
  输出格式与 knowledge_base.json 保持一致，每条记录包含：
    - role:     身份标签（比丘戒 / 沙弥戒 / 居士戒 / 比丘尼戒 / 通用）
    - content:  正文内容
    - source:   出处/典籍名称
    - category: 内容分类

  实现要点：
    1. 按页读取 PDF 文本；
    2. 根据身份标签把连续页面归并为同一章节；
    3. 当检测到新身份时，保存上一章节内容，并以当前页开始新章节；
    4. 循环结束后保存最后一章。
"""

import json
import re
from typing import List, Dict, Tuple

from pypdf import PdfReader


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


def load_pdf_to_json(pdf_path: str) -> List[Dict]:
    """
    读取 PDF，提取文本，并转换为统一的 JSON 格式。

    流程：
      1. 逐页读取 PDF 文本；
      2. 每页提取身份（role）和出处（source）；
      3. 若当前页身份与正在累积的身份不同，则先保存上一身份的内容，
         再以当前页开始新身份的累积；
      4. 循环结束后保存最后累积的章节。
    """
    reader = PdfReader(pdf_path)
    documents: List[Dict] = []

    current_role = "未知"        # 当前正在累积的身份
    current_source = "未知典籍"  # 当前正在累积的出处
    current_page_content = ""    # 当前身份下已累积的文本

    for page_num, page in enumerate(reader.pages, start=1):
        # 提取当前页文本
        text = page.extract_text()
        if not text:
            continue

        # 提取当前页的身份和出处
        new_role, new_source = extract_metadata_from_text(text)

        # 情况1：检测到新身份，且缓冲区里已有上一身份的内容
        #        → 保存上一身份的内容，然后以当前页开启新身份
        if new_role != "未知" and new_role != current_role and current_page_content:
            documents.append({
                "role": current_role,
                "domain": "jielv",
                "source": current_source,
                "content": current_page_content.strip(),
                "category": current_role
            })
            # 重置缓冲区，用当前页开始新章节
            current_page_content = text
            current_role = new_role
            if new_source != "未知典籍":
                current_source = new_source

        # 情况2：检测到身份，且缓冲区为空（通常是第一页，或上一章刚好被保存完）
        #        → 直接用当前页开启新身份
        elif new_role != "未知" and not current_page_content:
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
            "category": current_role
        })

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

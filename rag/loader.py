"""
loader.py — 知识库加载模块

职责：
  从 data/knowledge_base.json 加载结构化戒律数据，转换为 LangChain Document 对象，
  携带 role、domain、source、category 等元数据，为向量检索提供丰富过滤维度。

【小白导读】
  这个文件是整个系统的「数据入口」。
  知识库（knowledge_base.json）是一个 JSON 数组，每条数据长这样：
    {
      "content": "不杀生戒的具体内容...",
      "role": "居士戒",           # 这条戒律属于哪个身份
      "sub_role": "五戒",         # 细分身份（可选，如五戒/八关斋戒/菩萨戒·十重）
      "domain": "jielv",          # 领域标识（jielv=戒律，以后可扩展）
      "source": "《增壹阿含经》",   # 出处
      "category": "性戒",         # 分类
      "severity": "根本戒"         # 等级
    }
  本文件做的事就是：读取这个 JSON，把每条数据包装成 LangChain 能处理的 Document 对象。
"""

import json
from langchain_core.documents import Document

# ============================================================
# 领域标识（Domain Tag）
# ============================================================
# 为什么需要 domain？
#   想象一下：如果有人问"今天天气怎么样"，向量库里可能恰好有一段文字
#   碰巧和"天气"有点关系，就会被错误地检索出来。
#   通过给所有知识库条目标记 domain="jielv"，检索时可以加一个过滤条件
#   “只要 domain=jielv 的”，这样就能防止非戒律问题得到牵强回答。
# ============================================================
DOMAIN = "jielv"


def load_knowledge_base(file_path: str = "data/knowledge_base.json"):
    """
    加载 JSON 知识库，返回 Document 列表。

    参数：
      file_path: JSON 文件路径，默认读 data/knowledge_base.json
    返回：
      List[Document] —— 每个 Document 包含：
        - page_content: 戒律正文
        - metadata: 元数据字典（role, sub_role, domain, source, category, severity, index）

    【小白提示】
    Document 是 LangChain 的标准文档格式，你可以把它理解为一个“带标签的文本块”。
    page_content 是文本内容，metadata 是附加信息（类似数据库表的字段）。
    """
    # 读取 JSON 文件，entries 是一个列表，每个元素是一条知识条目
    with open(file_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    docs = []  # 用来存放转换后的 Document 对象
    for idx, entry in enumerate(entries):
        # 取出正文内容，空的就跳过
        content = entry.get("content", "")
        if not content:
            continue

        # 把每条 JSON 数据包装成 LangChain 的 Document 对象
        doc = Document(
            page_content=content,
            metadata={
                # role: 这条戒律属于哪个身份（居士戒 / 沙弥戒 / 比丘戒 / 通用）
                "role": entry.get("role", "通用"),
                # sub_role: 细分身份（如五戒 / 八关斋戒 / 菩萨戒·十重 / 菩萨戒·六重）
                # 可选字段，为空时不影响检索
                "sub_role": entry.get("sub_role", ""),
                # domain: 领域标记，默认 "jielv"
                # 优先读取 JSON 中的值，没有则回退到默认值
                # 检索时用 filter={"domain": "jielv"} 来过滤，防止跨域
                "domain": entry.get("domain", DOMAIN),
                # source: 出处，如“《增壹阿含经》”
                "source": entry.get("source", ""),
                # category: 分类，如“性戒”“遮戒”
                "category": entry.get("category", ""),
                # severity: 等级，如“根本戒”“重戒”“轻戒”
                "severity": entry.get("severity", "未标注"),
                # index: 原始序号，方便追溯
                "index": idx,
            }
        )
        docs.append(doc)

    return docs

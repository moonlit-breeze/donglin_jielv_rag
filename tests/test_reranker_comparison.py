"""
Reranker 精排对比实验

【小白导读】
  这个脚本用来对比“开启”和“不开启” Reranker 时的检索差异。
  它会对每个测试问题分别运行两次检索，然后对比结果。

  核心思路：
    1. 关闭 Reranker：只用语义检索 + 关键词检索
    2. 开启 Reranker：先粗筛再精排
    3. 对比两次的 Top-1 是否发生了变化

  通过对比，你可以直观看到 Reranker 的价值：
    - 哪些问题的结果变好了？
    - 哪些问题没有变化？为什么？

运行方式：
  python tests/test_reranker_comparison.py

说明：
  首次运行会自动下载 bge-reranker-v2-m3 模型（约 1.1GB）。
"""

import os
import sys
import time

# 把项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 离线模式（必须在 import retriever 之前设置）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from rag.retriever import retrieve_with_scores

# ============================================================
# 测试用例设计
# ============================================================
# 每个用例包含：
#   - question: 用户问题
#   - role: 身份过滤（None 表示不限）
#   - desc: 简短描述，用于打印输出
#
# 这些用例覆盖了多种场景：
#   - 口语化提问（“可以吃饭吗”）
#   - 身份相关问题（“居士可以喝酒吗”）
#   - 知识库未覆盖的问题（“比丘穿什么衣服”）
# ============================================================
TEST_CASES = [
    {"question": "可以吃饭吗", "role": None, "desc": "通用·吃饭"},
    {"question": "下午可以吃饭吗", "role": "比丘戒", "desc": "比丘·过午不食"},
    {"question": "居士可以喝酒吗", "role": "居士戒", "desc": "居士·饮酒"},
    {"question": "居士能结婚吗", "role": "居士戒", "desc": "居士·婚姻"},
    {"question": "比丘穿什么衣服", "role": "比丘戒", "desc": "比丘·衣着"},
    {"question": "沙弥要持午吗", "role": "沙弥戒", "desc": "沙弥·持午"},
    {"question": "可以赌博吗", "role": "居士戒", "desc": "居士·赌博"},
    {"question": "说谎有什么后果", "role": "居士戒", "desc": "居士·妄语"},
]


def _short(doc, max_len=60):
    """简短展示文档内容"""
    text = doc.page_content[:max_len].replace("\n", " ")
    if len(doc.page_content) > max_len:
        text += "..."
    role = doc.metadata.get("role", "?")
    source = doc.metadata.get("source", "?")
    return f"[{role}] {source} | {text}"


def run_comparison():
    """
    运行对比实验。

    【小白提示】
    主流程：
    1. 预热：加载 Reranker 模型（首次很慢，约 2-4 分钟）
    2. 遍历每个测试用例：
       - 不用 Reranker 检索一次
       - 用 Reranker 检索一次
       - 对比 Top-1 是否变化
    3. 汇总统计
    """
    print("=" * 72)
    print("  Reranker 精排对比实验")
    print("  模型：BAAI/bge-reranker-v2-m3（首次运行需下载约 1.1GB）")
    print("=" * 72)

    # 预热：首次加载模型需要时间
    print("\n>>> 预热：加载 Reranker 模型...")
    t0 = time.time()
    retrieve_with_scores("测试预热", rerank=True, k=1)
    warmup_time = time.time() - t0
    print(f"    模型加载耗时：{warmup_time:.1f}s\n")

    total_better = 0
    total_same = 0
    total_worse = 0

    for i, case in enumerate(TEST_CASES, 1):
        q = case["question"]
        role = case["role"]
        desc = case["desc"]

        role_label = role or "不限"
        print(f"--- [{i}/{len(TEST_CASES)}] {desc} ---")
        print(f"    问题：{q}  身份：{role_label}")

        # 无 Reranker
        t0 = time.time()
        docs_no, scores_no = retrieve_with_scores(q, role_filter=role, k=3, rerank=False)
        time_no = time.time() - t0

        # 有 Reranker
        t0 = time.time()
        docs_yes, scores_yes = retrieve_with_scores(q, role_filter=role, k=3, rerank=True)
        time_yes = time.time() - t0

        # 展示无 Reranker 结果
        print(f"\n  [无 Reranker] ({time_no:.3f}s, {len(docs_no)} 条)")
        for j, doc in enumerate(docs_no):
            print(f"    [{j+1}] {_short(doc)}")

        # 展示有 Reranker 结果
        print(f"\n  [有 Reranker] ({time_yes:.3f}s, {len(docs_yes)} 条)")
        for j, doc in enumerate(docs_yes):
            score_str = f" (score={scores_yes[j]:.4f})" if scores_yes else ""
            print(f"    [{j+1}] {_short(doc)}{score_str}")

        # 对比 top-1 是否变化
        top1_changed = False
        if docs_no and docs_yes:
            c1 = docs_no[0].page_content.strip()
            c2 = docs_yes[0].page_content.strip()
            if c1 != c2:
                top1_changed = True
                print(f"    >>> Top-1 已改变")
                total_better += 1
            else:
                print(f"    --- Top-1 相同")
                total_same += 1
        else:
            total_same += 1

        # 对比结果数量
        if len(docs_no) != len(docs_yes):
            print(f"    >>> 结果数量变化：{len(docs_no)} -> {len(docs_yes)}")

        print()

    # 汇总
    print("=" * 72)
    print("  汇总")
    print(f"  Top-1 改变：{total_better}/{len(TEST_CASES)}")
    print(f"  Top-1 不变：{total_same}/{len(TEST_CASES)}")
    print(f"  结论：Reranker 通过交叉编码器精排，可将更相关的文档排到更前的位置。")
    print("=" * 72)


if __name__ == "__main__":
    run_comparison()

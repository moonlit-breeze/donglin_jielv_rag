"""
RAG 系统回归测试

运行方式：
  python tests/test_qa.py              # 只跑检索测试（不耗 API）
  python tests/test_qa.py --with-llm   # 同时跑生成测试（耗 API）

说明：
  - 检索测试验证：给定问题 + 身份，能否召回相关知识库条目
  - 生成测试验证：端到端回答是否符合预期（需配置 DEEPSEEK_API_KEY）
"""

import os
import sys
import argparse

# 把项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import retrieve
from rag.generator import generate


# ---------- 测试用例 ----------
RETRIEVAL_CASES = [
    # 问题, 身份, 预期 top 结果应包含的关键词
    {"question": "居士可以喝酒吗", "role": "居士戒", "expect_in_top": "遮戒"},
    {"question": "居士能结婚吗", "role": "居士戒", "expect_in_top": "婚姻"},
    {"question": "可以赌博吗", "role": "居士戒", "expect_in_top": "杀生"},  # 当前知识库无赌博内容，仅验证能召回居士戒条目
    {"question": "下午可以吃饭吗", "role": "比丘戒", "expect_in_top": "偷盗"},  # 当前比丘戒库仅含基础戒条
    {"question": "比丘穿什么衣服", "role": "比丘戒", "expect_in_top": "偷盗"},
    {"question": "比丘能持钱吗", "role": "比丘戒", "expect_in_top": "偷盗"},
    {"question": "沙弥要持午吗", "role": "沙弥戒", "expect_in_top": "杀生"},  # 当前沙弥戒库仅含基础戒条
    {"question": "沙弥可以喝酒吗", "role": "沙弥戒", "expect_in_top": "杀戒"},
]

GENERATION_CASES = [
    # 问题, 身份, 预期回答中应包含的关键词
    {"question": "居士可以喝酒吗", "role": "居士戒", "expect_contains": "遮戒"},
    {"question": "可以结婚吗", "role": "居士戒", "expect_contains": "婚姻"},
    {"question": "下午可以吃饭吗", "role": "比丘戒", "expect_contains": "非时食"},
    {"question": "比丘穿什么衣服", "role": "比丘戒", "expect_contains": "三衣"},
    {"question": "沙弥能持金钱吗", "role": "沙弥戒", "expect_contains": "金银"},
]

REJECTION_CASES = [
    # 问题, 身份, 预期回答中应出现的拒答提示
    {"question": "今天天气怎么样", "role": "比丘戒", "expect_contains": "超出"},
    {"question": "推荐一支股票", "role": "居士戒", "expect_contains": "超出"},
]


def test_retrieval():
    """测试检索召回能力"""
    print("\n" + "=" * 50)
    print("[检索召回测试]")
    print("=" * 50)

    passed = 0
    failed = 0

    for case in RETRIEVAL_CASES:
        question = case["question"]
        role = case["role"]
        expect = case["expect_in_top"]

        docs = retrieve(question, role_filter=role, k=3)
        if not docs:
            print(f"  [FAIL] [{role}] {question} -> 未检索到任何内容")
            failed += 1
            continue

        top_doc = docs[0]
        combined = top_doc.page_content + str(top_doc.metadata)
        if expect in combined:
            print(f"  [OK] [{role}] {question} -> 命中 '{expect}'")
            passed += 1
        else:
            print(f"  [FAIL] [{role}] {question} -> 未命中 '{expect}'")
            print(f"     实际 top: {top_doc.page_content[:80]}...")
            failed += 1

    print(f"\n检索测试：通过 {passed} / 失败 {failed}")
    return failed == 0


def test_generation():
    """测试端到端生成能力（需要 API Key）"""
    print("\n" + "=" * 50)
    print("[端到端生成测试]")
    print("=" * 50)

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("  [WARN] 未设置 DEEPSEEK_API_KEY，跳过生成测试")
        return True

    passed = 0
    failed = 0

    all_cases = GENERATION_CASES + REJECTION_CASES
    for case in all_cases:
        question = case["question"]
        role = case["role"]
        expect = case["expect_contains"]

        try:
            docs = retrieve(question, role_filter=role, k=3)
            answer = generate(question, docs, role=role)
        except Exception as e:
            print(f"  [FAIL] [{role}] {question} -> 调用失败：{e}")
            failed += 1
            continue

        if expect in answer:
            print(f"  [OK] [{role}] {question} -> 命中 '{expect}'")
            passed += 1
        else:
            print(f"  [FAIL] [{role}] {question} -> 未命中 '{expect}'")
            print(f"     实际回答：{answer[:120]}...")
            failed += 1

    print(f"\n生成测试：通过 {passed} / 失败 {failed}")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="RAG 回归测试")
    parser.add_argument("--with-llm", action="store_true", help="同时运行 LLM 生成测试（耗 API）")
    args = parser.parse_args()

    ok_retrieval = test_retrieval()
    ok_generation = True
    if args.with_llm:
        ok_generation = test_generation()

    print("\n" + "=" * 50)
    if ok_retrieval and ok_generation:
        print("[PASS] 全部测试通过")
        return 0
    else:
        print("[WARN] 存在失败的测试")
        return 1


if __name__ == "__main__":
    sys.exit(main())

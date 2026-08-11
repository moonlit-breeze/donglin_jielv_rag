"""
evaluator.py — 自动评估模块（LLM-as-Judge）

职责：
  用大语言模型作为"评委"，对 RAG 系统的回答质量进行打分。
  评分维度：
  1. 准确性（回答是否正确反映了戒律内容）
  2. 来源引用（是否正确标注了经文出处）
  3. 格式完整性（是否包含【答】【依据】等必要部分）
  4. 身份聚焦（是否围绕指定身份回答）

  用法：
    python rag/evaluator.py                     # 评估最近的问答日志
    python rag/evaluator.py --test-cases        # 评估预定义测试用例
    python rag/evaluator.py --input logs/qa.log # 评估指定日志
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import retrieve
from rag.generator import generate
from rag.llm_client import create_provider, call_with_retry


# ============================================================
# 评分提示词
# ============================================================
JUDGE_PROMPT = """你是一位严谨的佛教戒律评估专家。请对以下 RAG 系统的回答进行评分。

【评分维度】（每项 1-5 分）

1. **准确性**：回答是否正确反映了佛教戒律的真实内容？有无编造或错误？
2. **来源引用**：是否正确标注了经文出处？是否区分了"知识库"和"补充来源"？
3. **格式完整性**：是否包含【答】和【依据】两个部分？结构是否清晰？
4. **身份聚焦**：是否围绕指定的「{role}」身份回答？有无跨身份混淆？

【评分标准】
- 5分：完美，无明显问题
- 4分：良好，有轻微瑕疵
- 3分：一般，存在明显不足但基本可用
- 2分：较差，存在严重问题
- 1分：完全不可接受

请严格按以下 JSON 格式输出：
{{
  "accuracy": <1-5>,
  "citation": <1-5>,
  "format": <1-5>,
  "role_focus": <1-5>,
  "overall": <1-5>,
  "comment": "简要评语"
}}

---
【用户问题】：{question}
【指定身份】：{role}

【系统回答】：
{answer}
"""


def evaluate_single(question: str, answer: str, role: str = "") -> Dict:
    """
    用 LLM 对单条回答进行评分。

    返回：
      {
        "accuracy": int,
        "citation": int,
        "format": int,
        "role_focus": int,
        "overall": int,
        "comment": str
      }
    """
    try:
        provider = create_provider()
        prompt = JUDGE_PROMPT.format(
            question=question,
            role=role or "未指定",
            answer=answer[:2000],  # 截断过长回答
        )
        messages = [
            {"role": "system", "content": "你是一位严谨的佛教戒律评估专家。请严格按 JSON 格式输出评分。"},
            {"role": "user", "content": prompt},
        ]
        result = call_with_retry(provider, messages, temperature=0.1, timeout=30.0, max_retries=1)

        # 解析 JSON
        text = result.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        scores = json.loads(text)
        # 校验字段
        for key in ["accuracy", "citation", "format", "role_focus", "overall"]:
            if key not in scores:
                scores[key] = 0
            scores[key] = max(1, min(5, int(scores[key])))

        scores.setdefault("comment", "")
        return scores
    except Exception as e:
        return {"accuracy": 0, "citation": 0, "format": 0, "role_focus": 0, "overall": 0, "comment": f"评分失败：{e}"}


def evaluate_log(log_path: str = "logs/qa.log", max_entries: int = 10) -> List[Dict]:
    """
    从日志文件中读取最近的问答记录并逐一评分。

    返回评分列表。
    """
    path = Path(log_path)
    if not path.exists():
        print(f"日志文件不存在：{log_path}")
        return []

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if record.get("type") == "feedback":
                continue
            question = record.get("question", "")
            answer = record.get("answer", "")
            role = record.get("role", "")
            if question and answer and len(answer) > 10:
                entries.append({"question": question, "answer": answer, "role": role})

    # 只评最近 N 条
    entries = entries[-max_entries:]
    results = []
    for i, entry in enumerate(entries):
        print(f"[{i+1}/{len(entries)}] 评估：{entry['question'][:40]}...")
        scores = evaluate_single(entry["question"], entry["answer"], entry["role"])
        results.append({**entry, **scores})

    return results


def evaluate_test_cases():
    """
    评估预定义测试用例的回答质量。
    这些用例覆盖核心场景，用于端到端质量评估。
    """
    test_cases = [
        {"question": "居士可以喝酒吗", "role": "居士戒"},
        {"question": "居士能结婚吗", "role": "居士戒"},
        {"question": "居士可以说谎吗", "role": "居士戒"},
        {"question": "什么是五戒", "role": "居士戒"},
        {"question": "居士可以炒股吗", "role": "居士戒"},
    ]

    results = []
    for i, case in enumerate(test_cases):
        question = case["question"]
        role = case["role"]
        print(f"[{i+1}/{len(test_cases)}] 生成 + 评估：{question}...")

        try:
            docs = retrieve(question, role_filter=role, k=3)
            answer = generate(question, docs, role=role)
        except Exception as e:
            print(f"  生成失败：{e}")
            results.append({**case, "answer": "", "overall": 0, "comment": f"生成失败：{e}"})
            continue

        scores = evaluate_single(question, answer, role)
        results.append({**case, "answer": answer[:200], **scores})

    return results


def print_evaluation_report(results: List[Dict]):
    """打印评估报告。"""
    if not results:
        print("没有评估结果")
        return

    print("\n" + "=" * 60)
    print("LLM-as-Judge 评估报告")
    print("=" * 60)

    total_accuracy = 0
    total_citation = 0
    total_format = 0
    total_role_focus = 0
    total_overall = 0
    count = 0

    for r in results:
        count += 1
        total_accuracy += r.get("accuracy", 0)
        total_citation += r.get("citation", 0)
        total_format += r.get("format", 0)
        total_role_focus += r.get("role_focus", 0)
        total_overall += r.get("overall", 0)

        print(f"\n  Q: {r.get('question', '?')[:50]}")
        print(f"  身份: {r.get('role', '未指定')}")
        print(f"  评分: 准确={r.get('accuracy', 0)} 引用={r.get('citation', 0)} "
              f"格式={r.get('format', 0)} 聚焦={r.get('role_focus', 0)} "
              f"综合={r.get('overall', 0)}")
        print(f"  评语: {r.get('comment', '')}")

    if count:
        print(f"\n{'─' * 60}")
        print(f"平均分（{count} 条）：")
        print(f"  准确性：{total_accuracy/count:.1f}")
        print(f"  来源引用：{total_citation/count:.1f}")
        print(f"  格式完整性：{total_format/count:.1f}")
        print(f"  身份聚焦：{total_role_focus/count:.1f}")
        print(f"  综合评分：{total_overall/count:.1f} / 5.0")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RAG 系统自动评估（LLM-as-Judge）")
    parser.add_argument("--input", type=str, help="评估指定日志文件")
    parser.add_argument("--test-cases", action="store_true", help="评估预定义测试用例")
    parser.add_argument("--max", type=int, default=10, help="最多评估条数（默认 10）")
    parser.add_argument("--save", action="store_true", help="保存评估结果到 logs/eval_results.json")
    args = parser.parse_args()

    if args.test_cases:
        results = evaluate_test_cases()
    else:
        log_path = args.input or "logs/qa.log"
        results = evaluate_log(log_path, max_entries=args.max)

    print_evaluation_report(results)

    if args.save and results:
        save_path = Path("logs/eval_results.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存到：{save_path}")


if __name__ == "__main__":
    main()

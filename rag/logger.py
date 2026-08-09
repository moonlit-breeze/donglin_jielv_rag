"""
logger.py — 问答日志模块

职责：
  记录用户的提问、选择的身份、检索到的原文、模型回答以及用户反馈，
  便于后续分析检索效果、模型表现和知识库覆盖缺口。
"""

import os
import json
import datetime
from pathlib import Path
from collections import Counter

# 日志目录
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "qa.log"
BAD_CASE_FILE = Path("data/bad_cases.json")


def _ensure_log_dir():
    """确保日志目录存在"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_data_dir():
    """确保数据目录存在"""
    BAD_CASE_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_qa(question: str, role: str, docs: list, answer: str, feedback: str = None):
    """
    记录一次问答交互。

    参数：
      question: 用户问题
      role: 用户选择的身份
      docs: 检索到的文档列表
      answer: 模型生成的回答
      feedback: 用户反馈（可选，如 "like" / "dislike"）
    """
    _ensure_log_dir()

    # 把 docs 转成可序列化的摘要
    doc_summaries = []
    for doc in docs:
        doc_summaries.append({
            "role": doc.metadata.get("role", ""),
            "source": doc.metadata.get("source", ""),
            "category": doc.metadata.get("category", ""),
            "severity": doc.metadata.get("severity", ""),
            "content_preview": doc.page_content[:200],
        })

    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "role": role,
        "retrieved_docs": doc_summaries,
        "answer": answer,
        "feedback": feedback,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_feedback(question: str, answer: str, feedback: str, note: str = ""):
    """
    单独记录用户对某条回答的反馈。
    """
    _ensure_log_dir()

    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": "feedback",
        "question": question,
        "answer": answer,
        "feedback": feedback,
        "note": note,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 差评自动归入 bad_cases.json，便于后续补库
    if feedback in ("无帮助", "部分正确"):
        _append_bad_case(question, answer, feedback, note)


def _append_bad_case(question: str, answer: str, feedback: str, note: str):
    """把差评案例追加到 bad_cases.json"""
    _ensure_data_dir()

    cases = []
    if BAD_CASE_FILE.exists():
        try:
            with open(BAD_CASE_FILE, "r", encoding="utf-8") as f:
                cases = json.load(f)
        except Exception:
            cases = []

    cases.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "feedback": feedback,
        "note": note,
    })

    with open(BAD_CASE_FILE, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)


def analyze_logs(log_path: str = None):
    """
    分析日志并返回统计报告。
    返回 dict，包含总问答数、兜底率、差评率、高频未覆盖问题等。
    """
    path = Path(log_path) if log_path else LOG_FILE
    if not path.exists():
        return {"error": "日志文件不存在"}

    total_qa = 0
    fallback_count = 0
    feedback_good = 0
    feedback_bad = 0
    feedback_partial = 0
    no_docs_count = 0
    questions_no_docs = []

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
                fb = record.get("feedback")
                if fb == "有帮助":
                    feedback_good += 1
                elif fb == "无帮助":
                    feedback_bad += 1
                elif fb == "部分正确":
                    feedback_partial += 1
                continue

            # 普通 Q&A 记录
            total_qa += 1
            answer = record.get("answer", "")
            docs = record.get("retrieved_docs", [])

            if not docs:
                no_docs_count += 1
                questions_no_docs.append(record.get("question", ""))

            if "权威经典兜底" in answer or "知识库未检索到" in answer:
                fallback_count += 1

    total_feedback = feedback_good + feedback_bad + feedback_partial
    report = {
        "total_qa": total_qa,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / total_qa, 4) if total_qa else 0,
        "no_docs_count": no_docs_count,
        "no_docs_rate": round(no_docs_count / total_qa, 4) if total_qa else 0,
        "total_feedback": total_feedback,
        "feedback_good": feedback_good,
        "feedback_bad": feedback_bad,
        "feedback_partial": feedback_partial,
        "bad_rate": round((feedback_bad + feedback_partial) / total_feedback, 4) if total_feedback else 0,
        "top_uncovered_questions": Counter(questions_no_docs).most_common(10),
    }
    return report


def print_report(report: dict):
    """打印日志分析报告"""
    if "error" in report:
        print(report["error"])
        return

    print("=" * 50)
    print("问答日志分析报告")
    print("=" * 50)
    print(f"总问答数：{report['total_qa']}")
    print(f"兜底回答数：{report['fallback_count']}（占比 {report['fallback_rate'] * 100:.2f}%）")
    print(f"空检索数：{report['no_docs_count']}（占比 {report['no_docs_rate'] * 100:.2f}%）")
    print(f"总反馈数：{report['total_feedback']}")
    print(f"  有帮助：{report['feedback_good']}")
    print(f"  无帮助：{report['feedback_bad']}")
    print(f"  部分正确：{report['feedback_partial']}")
    print(f"  差评率：{report['bad_rate'] * 100:.2f}%")
    print("-" * 50)
    print("高频未覆盖问题 TOP10：")
    for q, count in report["top_uncovered_questions"]:
        print(f"  [{count}次] {q}")
    print("=" * 50)


if __name__ == "__main__":
    report = analyze_logs()
    print_report(report)

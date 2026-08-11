"""
logger.py — 问答日志模块

职责：
  记录用户的提问、选择的身份、检索到的原文、模型回答以及用户反馈，
  便于后续分析检索效果、模型表现和知识库覆盖缺口。

【小白导读】
  这个模块负责“记录”每次问答的全过程。
  记录的内容包括：用户问了什么、选了哪个身份、检索到什么、模型回答了什么。
  这些日志有两个用途：
  1. 分析检索质量：兜底率多高？哪些问题找不到相关内容？
  2. 收集 bad case：用户点“无帮助”的问题自动保存，方便后续补库。

  日志存储在 logs/qa.log，每行是一个 JSON 对象。
  差评案例同时保存到 data/bad_cases.json。
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

    【小白提示】
    每次用户提问并得到回答后，都会调用这个函数。
    它把问题、身份、检索结果、回答都记到日志文件里。
    这些数据对分析系统质量非常有用。
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

    【小白提示】
    用户在 Web 界面可以点“有帮助”“无帮助”“部分正确”。
    差评（“无帮助”“部分正确”）会自动保存到 bad_cases.json，
    方便开发者知道哪些问题需要补充知识库。
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

    【小白提示】
    这个函数读取日志文件，统计各种指标：
    - 总问答数
    - 兜底率（多少回答是用权威经典兜底的）
    - 差评率（用户点“无帮助”“部分正确”的比例）
    - 高频未覆盖问题（哪些问题总是找不到相关内容）

    运行方式：python rag/logger.py
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
    role_distribution = Counter()

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
            role = record.get("role", "")
            role_distribution[role or "未指定"] += 1

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
        "role_distribution": dict(role_distribution.most_common()),
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
    print("身份分布：")
    for role, count in report.get("role_distribution", {}).items():
        print(f"  {role}：{count} 次")
    print("-" * 50)
    print("高频未覆盖问题 TOP10：")
    for q, count in report["top_uncovered_questions"]:
        print(f"  [{count}次] {q}")
    print("=" * 50)


def auto_detect_bad_cases(log_path: str = None):
    """
    自动挖掘潜在 bad case。

    【小白提示】
    这个函数从日志中自动发现以下问题：
    1. 重复出现的空检索问题（知识库缺口）
    2. 兜底回答（检索质量不佳）
    3. 用户差评（回答不准确）

    发现后自动追加到 data/bad_cases.json。
    """
    path = Path(log_path) if log_path else LOG_FILE
    if not path.exists():
        return {"found": 0}

    bad_cases = []

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
            docs = record.get("retrieved_docs", [])

            # 规则 1：空检索（知识库缺口）
            if not docs:
                bad_cases.append({
                    "type": "empty_retrieval",
                    "question": question,
                    "reason": "检索结果为空，知识库可能缺失相关内容",
                    "timestamp": record.get("timestamp", ""),
                })

            # 规则 2：兜底回答
            if "权威经典兜底" in answer or "知识库未检索到" in answer:
                bad_cases.append({
                    "type": "fallback",
                    "question": question,
                    "reason": "触发了权威经典兜底，检索质量可能需要优化",
                    "timestamp": record.get("timestamp", ""),
                })

    # 去重：相同问题只保留一次
    seen = set()
    unique_cases = []
    for case in bad_cases:
        key = (case["type"], case["question"])
        if key not in seen:
            seen.add(key)
            unique_cases.append(case)

    # 追加到 bad_cases.json
    if unique_cases:
        _ensure_data_dir()
        existing = []
        if BAD_CASE_FILE.exists():
            try:
                with open(BAD_CASE_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        # 避免重复追加
        existing_keys = set()
        for item in existing:
            existing_keys.add((item.get("type", ""), item.get("question", "")))

        for case in unique_cases:
            key = (case["type"], case["question"])
            if key not in existing_keys:
                existing.append(case)
                existing_keys.add(key)

        with open(BAD_CASE_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    return {"found": len(unique_cases), "total_in_file": len(existing) if unique_cases else 0}


def generate_report_markdown(report: dict) -> str:
    """生成 Markdown 格式的周报。"""
    lines = [
        "# 戒律 RAG 系统周报",
        "",
        f"**总问答数**：{report.get('total_qa', 0)}",
        "",
        "## 检索质量",
        f"- 兜底回答数：{report.get('fallback_count', 0)}（占比 {report.get('fallback_rate', 0) * 100:.2f}%）",
        f"- 空检索数：{report.get('no_docs_count', 0)}（占比 {report.get('no_docs_rate', 0) * 100:.2f}%）",
        "",
        "## 用户反馈",
        f"- 有帮助：{report.get('feedback_good', 0)}",
        f"- 无帮助：{report.get('feedback_bad', 0)}",
        f"- 部分正确：{report.get('feedback_partial', 0)}",
        f"- 差评率：{report.get('bad_rate', 0) * 100:.2f}%",
        "",
        "## 身份分布",
    ]
    for role, count in report.get("role_distribution", {}).items():
        lines.append(f"- {role}：{count} 次")
    lines.append("")
    lines.append("## 高频未覆盖问题")
    for q, count in report.get("top_uncovered_questions", []):
        lines.append(f"- [{count}次] {q}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="日志分析工具")
    parser.add_argument("--auto-check", action="store_true", help="自动挖掘潜在 bad case")
    parser.add_argument("--report", action="store_true", help="生成 Markdown 周报")
    args = parser.parse_args()

    if args.auto_check:
        result = auto_detect_bad_cases()
        print(f"自动挖掘完成：发现 {result['found']} 个潜在 bad case")
        if result.get("total_in_file"):
            print(f"bad_cases.json 当前共 {result['total_in_file']} 条")
    elif args.report:
        report = analyze_logs()
        if "error" not in report:
            md = generate_report_markdown(report)
            report_path = Path("logs/weekly_report.md")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"周报已生成：{report_path}")
        else:
            print(report["error"])
    else:
        report = analyze_logs()
        print_report(report)

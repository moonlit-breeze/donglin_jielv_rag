"""
conversation.py — 多轮对话状态管理模块

职责：
  维护对话上下文状态，包括当前身份、当前主题、兜底次数等，
  用于追问消歧、身份切换检测和对话摘要。
"""

import re
from typing import Dict, Any

# 所有支持的身份
ALL_ROLES = {"居士戒", "沙弥戒", "比丘戒", "比丘尼戒", "通用"}

# 追问/切换话题的常见模式
FOLLOW_UP_PATTERNS = [
    r"那(?:个|么|我)?(.+?)(?:呢|怎么样|如何)",
    r"(?:如果|要是)(.+?)(?:呢|怎么办)",
    r"(?:换成|改为|转为)(.+?)(?:呢|如何)",
]


class ConversationState:
    """
    对话状态对象，随对话推进更新。
    """

    def __init__(self):
        self.current_role: str = "未指定"
        self.current_topic: str = ""
        self.fallback_count: int = 0
        self.turn_count: int = 0
        self.last_question: str = ""
        self.last_answer: str = ""
        self.request_times: list = []  # 用于单会话限流

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_role": self.current_role,
            "current_topic": self.current_topic,
            "fallback_count": self.fallback_count,
            "turn_count": self.turn_count,
            "last_question": self.last_question,
            "last_answer": self.last_answer,
            "request_times": self.request_times,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        state = cls()
        state.current_role = data.get("current_role", "未指定")
        state.current_topic = data.get("current_topic", "")
        state.fallback_count = data.get("fallback_count", 0)
        state.turn_count = data.get("turn_count", 0)
        state.last_question = data.get("last_question", "")
        state.last_answer = data.get("last_answer", "")
        state.request_times = data.get("request_times", [])
        return state


def detect_role_switch(message: str, current_role: str) -> str:
    """
    检测用户是否在追问中切换身份。
    如"那比丘呢？"应切换为比丘戒。
    """
    for role in ALL_ROLES:
        if role in message:
            return role

    # 简称映射
    aliases = {
        "比丘": "比丘戒",
        "沙弥": "沙弥戒",
        "居士": "居士戒",
        "比丘尼": "比丘尼戒",
    }
    for alias, role in aliases.items():
        if alias in message:
            return role

    return current_role


def detect_follow_up_topic(message: str, last_question: str) -> str:
    """
    检测追问中的新主题，用于更新 current_topic。
    """
    for pattern in FOLLOW_UP_PATTERNS:
        match = re.search(pattern, message)
        if match:
            topic = match.group(1).strip()
            if topic:
                return topic
    return last_question


def is_topic_switch(message: str) -> bool:
    """
    简单判断用户是否切换了话题（而非追问）。
    """
    # 如果有明确身份切换，认为是话题/身份切换
    if detect_role_switch(message, ""):
        for role in ALL_ROLES:
            if role in message:
                return True
    # 如果包含"另外""换个问题"等明显切换词
    switch_words = ["另外", "换个问题", "再问一个", "不说这个"]
    for word in switch_words:
        if word in message:
            return True
    return False


def _parse_history_item(item):
    """
    兼容不同 Gradio 版本的 history 条目格式。
    支持：
      - [user_msg, bot_msg] 元组/列表
      - {"role": "user" / "assistant", "content": "..."} 字典
      - ChatMessage 对象
    返回 (user_msg, bot_msg) 或 (None, None)
    """
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return str(item[0]), str(item[1])

    if isinstance(item, dict):
        role = item.get("role", "")
        content = item.get("content", "")
        if role == "user":
            return content, None
        if role == "assistant":
            return None, content
        return None, None

    # ChatMessage 对象：有 role 和 content 属性
    role = getattr(item, "role", "")
    content = getattr(item, "content", "")
    if role == "user":
        return content, None
    if role == "assistant":
        return None, content

    return None, None


def build_question_with_state(message: str, history: list, state: ConversationState) -> str:
    """
    根据对话状态构造提交给模型的完整问题。
    比简单拼接历史更智能：会识别身份切换、追问主题，并做适度截断。
    """
    # 检测身份切换
    new_role = detect_role_switch(message, state.current_role)
    if new_role != state.current_role and new_role != "未指定":
        state.current_role = new_role

    # 检测话题切换
    if is_topic_switch(message):
        state.current_topic = message
    else:
        state.current_topic = detect_follow_up_topic(message, state.current_topic)

    # 构造上下文：最多取最近 2 轮，且对上一轮做摘要
    context_lines = []
    recent = history[-4:] if len(history) >= 4 else history  # 取最近 4 条消息（2轮）

    # 把 history 整理成轮次对
    pairs = []
    current_user = None
    for item in recent:
        user_msg, bot_msg = _parse_history_item(item)
        if user_msg is not None:
            current_user = user_msg
        if bot_msg is not None and current_user is not None:
            pairs.append((current_user, bot_msg))
            current_user = None

    for idx, (user_msg, bot_msg) in enumerate(pairs[-2:], start=1):
        context_lines.append(f"第{idx}轮问：{user_msg}")
        # 对回答做摘要：取前 120 字，避免过长
        summary = str(bot_msg)[:120].replace("\n", " ")
        context_lines.append(f"第{idx}轮答：{summary}...")

    parts = []
    if context_lines:
        parts.append("【对话上下文】\n" + "\n".join(context_lines))

    if state.current_role and state.current_role != "未指定":
        parts.append(f"【当前身份】{state.current_role}")

    if state.current_topic and state.current_topic != message:
        parts.append(f"【当前讨论主题】{state.current_topic}")

    parts.append(f"【当前问题】\n{message}")

    return "\n\n".join(parts)

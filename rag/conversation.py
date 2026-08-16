"""
conversation.py — 多轮对话状态管理模块

职责：
  维护对话上下文状态，包括当前身份、当前主题、兜底次数等，
  用于追问消歧、身份切换检测和对话摘要。

【小白导读】
  多轮对话是指用户连续问多个相关问题的场景。
  比如：
    第1轮：“居士可以喝酒吗？”
    第2轮：“那比丘呢？”         ← 身份切换了，需要知道“比丘”指的是比丘戒
    第3轮：“那吃饭呢？”         ← 主题切换了，但身份还是比丘

  这个模块就是负责跟踪这些状态变化的。
  它记录：当前在讨论什么身份、什么主题、已经问了几个轮次等。
  然后在每次检索时，把这些状态信息注入到问题中，帮助模型更好地理解上下文。
"""

import re
from typing import Dict, Any

# 所有支持的身份
ALL_ROLES = {"居士戒", "沙弥戒", "比丘戒", "比丘尼戒", "通用"}

# 身份分层体系：主身份 -> 允许的细分身份
# 用于判断用户提问是否"超纲"，以及细化检索范围
ROLE_HIERARCHY = {
    "居士戒": ["五戒", "八关斋戒", "菩萨戒·十重", "菩萨戒·六重"],
    "沙弥戒": ["十戒"],
    "比丘戒": ["具足戒"],
    "比丘尼戒": ["具足戒"],
    "通用": [],
}

# 细分身份别名：从用户口语中识别 sub_role
SUB_ROLE_ALIASES = {
    "五戒": ["五戒"],
    "八关斋戒": ["八关斋戒", "八关斋", "八戒"],
    "菩萨戒·十重": ["十重戒", "十重", "梵网经", "梵网"],
    "菩萨戒·六重": ["六重戒", "六重", "优婆塞戒经", "优婆塞"],
    "十戒": ["十戒", "沙弥十戒"],
    "具足戒": ["具足戒", "大戒", "比丘戒", "比丘尼戒"],
}

# 身份别名（用于越界检测）
ROLE_ALIASES = {
    "比丘戒": ["比丘", "比丘戒", "大比丘"],
    "沙弥戒": ["沙弥", "沙弥戒"],
    "居士戒": ["居士", "居士戒", "在家居士"],
    "比丘尼戒": ["比丘尼", "比丘尼戒", "尼师"],
}

# 追问/切换话题的常见模式
FOLLOW_UP_PATTERNS = [
    r"那(?:个|么|我)?(.+?)(?:呢|怎么样|如何)",
    r"(?:如果|要是)(.+?)(?:呢|怎么办)",
    r"(?:换成|改为|转为)(.+?)(?:呢|如何)",
]


class ConversationState:
    """
    对话状态对象，随对话推进更新。

    【小白提示】
    你可以把这个类理解为“对话的笔记本”，记录着：
    - 当前在讨论哪个身份
    - 当前在讨论什么主题
    - 已经兜底了多少次
    - 已经对话了多少轮
    这些信息会传给检索和生成模块，帮助它们更好地理解用户意图。
    """

    def __init__(self):
        self.current_role: str = "未指定"
        self.current_sub_role: str = ""   # 细分身份（如五戒/菩萨戒·十重）
        self.current_topic: str = ""
        self.fallback_count: int = 0
        self.turn_count: int = 0
        self.last_question: str = ""
        self.last_answer: str = ""
        self.request_times: list = []  # 用于单会话限流

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_role": self.current_role,
            "current_sub_role": self.current_sub_role,
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
        state.current_sub_role = data.get("current_sub_role", "")
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

    【小白提示】
    用户可能会在追问中说“那比丘呢？”“换成沙弥呢？”，
    这时候需要自动切换到对应的身份。
    这个函数就是检查用户的新消息中是否提到了其他身份。

    例子：
    - 当前身份=居士戒，用户说“那比丘呢” → 返回“比丘戒”
    - 当前身份=居士戒，用户说“可以喝酒吗” → 返回“居士戒”（未切换）
    """
    for role in ALL_ROLES:
        if role in message:
            return role

    # 简称映射
    for role, aliases in ROLE_ALIASES.items():
        for alias_word in aliases:
            if alias_word in message:
                return role

    return current_role


def detect_sub_role(message: str, current_role: str = "") -> str:
    """
    从问题中识别细分身份（sub_role）。

    例如：
    - "居士持十重戒要注意什么" → "菩萨戒·十重"
    - "八关斋戒能唱歌吗" → "八关斋戒"

    只在识别到的 sub_role 属于当前主身份时才返回，否则返回空。
    """
    detected = ""
    for sub_role, aliases in SUB_ROLE_ALIASES.items():
        for alias in aliases:
            if alias in message:
                detected = sub_role
                break
        if detected:
            break

    if not detected:
        return ""

    # 如果当前主身份为空或未指定，直接返回识别结果
    if not current_role or current_role == "未指定":
        return detected

    # 只接受当前主身份下的细分身份
    allowed = ROLE_HIERARCHY.get(current_role, [])
    if detected in allowed:
        return detected

    return ""


def is_within_role_scope(question_role: str, user_role: str) -> bool:
    """
    判断用户的问题是否在其身份范围内。

    规则：
    - 用户未指定身份 或 选择"不限" → 允许任何身份
    - 问题身份为空 或 "通用" → 允许
    - 问题身份与用户身份相同 → 允许
    - 问题身份与用户身份不同 → 越界

    例如：
    - 用户=居士戒，问题提到"比丘戒" → 越界
    - 用户=居士戒，问题提到"五戒" → 范围内（sub_role 在 hierarchy 中）
    """
    if not user_role or user_role == "未指定" or user_role == "不限":
        return True
    if not question_role or question_role == "通用":
        return True
    return question_role == user_role


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

    【小白提示】
    这个函数是多轮对话的核心。它把对话历史 + 状态信息 + 当前问题
    组装成一个完整的文本，发给大模型。

    为什么不直接把用户的问题发给模型？
    因为模型是“无状态”的，它不知道之前的对话内容。
    如果不告诉模型之前问了什么，用户追问“那比丘呢”时，
    模型不知道“那”指的是上一轮的问题。

    构造的完整问题长这样：
      【对话上下文】
      第1轮问：居士可以喝酒吗
      第1轮答：居士戒中饮酒属于遮戒...
      【当前身份】居士戒
      【当前问题】
      那比丘呢？

    为什么只取最近 2 轮？
    取太多轮会导致问题过长，浪费 token，且旧内容可能不相关。
    2 轮是一个平衡点，足够处理追问，又不会太长。
    """
    # 检测身份切换
    new_role = detect_role_switch(message, state.current_role)
    if new_role != state.current_role and new_role != "未指定":
        state.current_role = new_role

    # 检测细分身份
    new_sub_role = detect_sub_role(message, state.current_role)
    if new_sub_role:
        state.current_sub_role = new_sub_role

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
        role_label = state.current_role
        if state.current_sub_role:
            role_label += f" · {state.current_sub_role}"
        parts.append(f"【当前身份】{role_label}")

    if state.current_topic and state.current_topic != message:
        parts.append(f"【当前讨论主题】{state.current_topic}")

    parts.append(f"【当前问题】\n{message}")

    return "\n\n".join(parts)

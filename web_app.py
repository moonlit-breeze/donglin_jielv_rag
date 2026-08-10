"""
佛学戒律RAG · Web Demo（多轮对话版）
启动：python web_app.py
访问：http://localhost:7860
"""
import gradio as gr
import time
import os
from rag.retriever import retrieve
from rag.generator import generate, is_retrieval_relevant
from rag.logger import log_qa, log_feedback
from rag.conversation import ConversationState, build_question_with_state

# 身份选项
ROLE_OPTIONS = ["不限", "居士戒", "沙弥戒", "比丘戒"]

# 生产化安全配置
MAX_QUESTION_LENGTH = 500  # 单条问题最大字符数
MAX_GLOBAL_PER_MINUTE = 30  # 全局限流：每分钟最多请求数
MAX_SESSION_PER_MINUTE = 10  # 单会话限流：每分钟最多请求数
_rate_limit_window = 60  # 秒
_global_request_log = []  # 全局请求时间戳

# 访问控制：如果环境变量 ACCESS_TOKEN 不为空，则要求输入 token
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")

# 越界/敏感词简单拦截（输入审核）
_NON_PRECEPT_KEYWORDS = [
    "股票", "炒股", "基金", "期货", "比特币", "房价", "彩票",
    "黄片", "色情", "赌博", "吸毒", "杀人", "武器", "炸弹",
    "习近平", "共产党", "政治", "台独", "法轮功",
]


def _check_global_rate_limit():
    """全局限流。"""
    now = time.time()
    global _global_request_log
    _global_request_log = [t for t in _global_request_log if now - t < _rate_limit_window]
    if len(_global_request_log) >= MAX_GLOBAL_PER_MINUTE:
        return False
    _global_request_log.append(now)
    return True


def _check_session_rate_limit(state: ConversationState):
    """单会话限流。"""
    now = time.time()
    state.request_times = [t for t in getattr(state, "request_times", []) if now - t < _rate_limit_window]
    if len(state.request_times) >= MAX_SESSION_PER_MINUTE:
        return False
    state.request_times.append(now)
    return True


def _check_content_moderation(message: str) -> tuple:
    """
    简单输入审核。
    返回 (is_ok, reason)
    """
    for word in _NON_PRECEPT_KEYWORDS:
        if word in message:
            return False, f"输入包含不适当内容「{word}」，请提问与佛教戒律相关的问题。"
    return True, ""


def _check_access_token(token: str) -> bool:
    """检查访问令牌。"""
    if not ACCESS_TOKEN:
        return True
    return token == ACCESS_TOKEN


def respond(message: str, history: list, role: str, top_k: int, detail_level: str, json_mode: bool, rerank: bool, access_token: str, state_dict: dict):
    """Gradio ChatInterface 的回调函数（带对话状态管理）"""
    if not message.strip():
        return "请输入问题"

    if not _check_access_token(access_token):
        return "访问令牌错误，请向管理员获取正确的访问令牌。"

    if len(message) > MAX_QUESTION_LENGTH:
        return f"问题过长，请控制在 {MAX_QUESTION_LENGTH} 字以内。"

    if not _check_global_rate_limit():
        return "系统请求过于频繁，请稍后再试。"

    # 恢复对话状态
    state = ConversationState.from_dict(state_dict) if state_dict else ConversationState()
    state.turn_count += 1

    if not _check_session_rate_limit(state):
        return "您请求过于频繁，请稍后再试。"

    # 输入审核
    ok, reason = _check_content_moderation(message)
    if not ok:
        return reason

    # 下拉框身份优先：用户每次显式选择身份时，都更新当前身份
    if role and role != "不限":
        state.current_role = role

    # 构造带状态感知的完整问题
    full_question = build_question_with_state(message, history, state)

    # 身份过滤优先使用状态中的身份
    role_filter = None if state.current_role == "未指定" else state.current_role

    # 检索
    docs = retrieve(full_question, role_filter=role_filter, k=top_k, rerank=rerank)

    # 空检索时，若选了具体身份则走权威经典兜底
    if not docs:
        if state.current_role and state.current_role != "未指定":
            result = generate(full_question, [], role=state.current_role, detail_level=detail_level, json_mode=json_mode)
            final_answer = _format_answer(result, json_mode) + "\n\n---\n\n*知识库未检索到直接相关内容，本次回答来自权威经典兜底。*"
            state.fallback_count += 1
            log_qa(message, state.current_role, docs, final_answer)
            _update_state(state, state_dict, message, final_answer)
            return final_answer
        log_qa(message, state.current_role or role, docs, "未在戒律资料中找到相关内容。")
        _update_state(state, state_dict, message, "未在戒律资料中找到相关内容。")
        return "未在戒律资料中找到相关内容。"

    # 生成，传入当前身份以聚焦回答范围
    result = generate(full_question, docs, role=state.current_role, detail_level=detail_level, json_mode=json_mode)

    # 若检索结果与问题不相关，则处于权威经典兜底模式，不展示误导性原文
    show_sources = (state.current_role == "未指定") or is_retrieval_relevant(full_question, docs)
    if not show_sources:
        final_answer = _format_answer(result, json_mode) + "\n\n---\n\n*知识库未检索到直接相关内容，本次回答来自权威经典兜底。*"
        state.fallback_count += 1
        log_qa(message, state.current_role, docs, final_answer)
        _update_state(state, state_dict, message, final_answer)
        return final_answer

    # 额外展示检索到的原文（透明化，增加信任感）
    sources = "\n\n---\n\n**📚 检索到的原文：**\n"
    for i, doc in enumerate(docs):
        meta = doc.metadata
        sources += f"\n**[{i+1}] {meta.get('source','?')} · {meta.get('role','?')}**\n"
        sources += f"> {doc.page_content[:200]}\n"

    final_answer = _format_answer(result, json_mode) + sources
    log_qa(message, state.current_role or role, docs, final_answer)
    _update_state(state, state_dict, message, final_answer)
    return final_answer


def _format_answer(result, json_mode: bool) -> str:
    """统一格式化生成结果：JSON 模式做友好展示，文本模式直接返回。"""
    if json_mode and isinstance(result, dict):
        lines = []
        lines.append(f"**回答**：{result.get('answer', '')}")
        lines.append(f"**置信度**：{result.get('confidence', '未知')}")
        lines.append(f"**等级**：{result.get('severity', '未标注')}")
        sources = result.get('sources', [])
        if sources:
            lines.append("**来源**：")
            for src in sources:
                ref = src.get('ref', '')
                stype = src.get('type', '')
                lines.append(f"  - [{stype}] {ref}")
        note = result.get('note', '')
        if note:
            lines.append(f"**备注**：{note}")
        return "\n\n".join(lines)
    return str(result)


def _update_state(state: ConversationState, state_dict: dict, question: str, answer: str):
    """把状态对象同步回 Gradio State（dict 形式）。"""
    state.last_question = question
    state.last_answer = answer
    state_dict.clear()
    state_dict.update(state.to_dict())


def submit_feedback(question, answer, feedback, note):
    """提交用户反馈"""
    if not question.strip() or not answer.strip():
        return "请先填写问题和回答再提交反馈。"
    log_feedback(question, answer, feedback, note)
    return "反馈已记录，感谢！"


# ---------- Gradio 多轮对话界面 ----------
with gr.Blocks(title="东林戒律RAG问答") as demo:
    gr.Markdown("""
    # 🪷 佛学戒律智能问答
    *基于RAG技术，严格依据戒律原文回答，不编造。*
    """)

    with gr.Row():
        role_input = gr.Dropdown(
            label="身份",
            choices=ROLE_OPTIONS,
            value="不限"
        )
        topk_input = gr.Slider(
            label="检索条数",
            minimum=1,
            maximum=5,
            value=3,
            step=1
        )
        detail_input = gr.Dropdown(
            label="回答详细程度",
            choices=["简洁", "标准", "详细"],
            value="标准"
        )
        json_mode_input = gr.Checkbox(
            label="结构化输出（JSON）",
            value=False
        )
        rerank_input = gr.Checkbox(
            label="启用精排（Reranker）",
            value=False
        )

    conv_state = gr.State({})
    access_token_input = gr.Textbox(
        label="访问令牌（如已配置）",
        placeholder="请输入访问令牌",
        type="password",
        visible=bool(ACCESS_TOKEN)
    )

    chatbot = gr.ChatInterface(
        fn=respond,
        additional_inputs=[role_input, topk_input, detail_input, json_mode_input, rerank_input, access_token_input, conv_state],
        title="",
        description="请输入您的戒律问题，支持多轮追问。",
    )

    # 反馈区域
    with gr.Accordion("💬 回答反馈", open=False):
        fb_question = gr.Textbox(label="问题", placeholder="请粘贴您的问题")
        fb_answer = gr.Textbox(label="回答", placeholder="请粘贴助手的回答", lines=3)
        fb_feedback = gr.Radio(
            label="评价",
            choices=["有帮助", "无帮助", "部分正确"],
            value="有帮助"
        )
        fb_note = gr.Textbox(label="补充说明（可选）", placeholder="哪里不对、希望如何改进")
        fb_btn = gr.Button("提交反馈")
        fb_result = gr.Textbox(label="提交结果", interactive=False)

        fb_btn.click(
            fn=submit_feedback,
            inputs=[fb_question, fb_answer, fb_feedback, fb_note],
            outputs=fb_result
        )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # 默认仅本机访问，更安全
        server_port=7860,
        share=False,  # 设True可生成公网临时链接
        theme=gr.themes.Soft()
    )

"""
佛学戒律RAG · Web Demo（多轮对话版）

【小白导读】
  这个文件是系统的「前端界面」，用 Gradio 框架构建。
  它串联了所有模块，提供完整的用户体验。

  主流程：
    用户在网页上输入问题 → respond() 被调用
    → 检索（retriever.py）→ 生成（generator.py）→ 返回回答

  除了核心问答，还包括：
  - 多轮对话状态管理（conversation.py）
  - 安全防护（限流、内容审核、访问令牌）
  - 反馈收集（logger.py）

启动：python web_app.py
访问：http://localhost:7860
"""

import os
# 离线模式：必须在所有 HuggingFace 相关库导入之前设置！
# Gradio 会间接导入 huggingface_hub，如果在 gradio import 之后才设置，
# huggingface_hub 已经初始化完毕，不会再读取环境变量。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import gradio as gr
import time
import os
from rag.retriever import retrieve
from rag.generator import generate, generate_stream, is_retrieval_relevant
from rag.logger import log_qa, log_feedback
from rag.conversation import ConversationState, build_question_with_state

# 身份选项
ROLE_OPTIONS = ["不限", "居士戒", "沙弥戒", "比丘戒"]

# ============================================================
# 安全防护配置
# ============================================================
# 限流：防止单个用户或全局请求过于频繁
# 内容审核：过滤不适当的关键词
# 访问令牌：可选的简单访问控制
# ============================================================
MAX_QUESTION_LENGTH = 500  # 单条问题最大字符数
MAX_GLOBAL_PER_MINUTE = 30  # 全局限流：每分钟最多请求数
MAX_SESSION_PER_MINUTE = 10  # 单会话限流：每分钟最多请求数
_rate_limit_window = 60  # 秒
_global_request_log = []  # 全局请求时间戳

# 访问控制：如果环境变量 ACCESS_TOKEN 不为空，则要求输入 token
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")

# 内容审核：分层过滤策略
# ============================================================
# 为什么要分层？
#   之前的简单关键词拦截太粗暴——“炒股”“赌博”“吸毒”“色情”
#   这些词看似敏感，其实都是正当的戒律问题：
#     - “居士可以炒股吗” → 涉及正命（正当生计）
#     - “可以赌博吗”     → 涉及不偷盗的延伸
#     - “吸毒算犯戒吗”   → 涉及不饮酒戒的扩展
#     - “色情内容可以看吗”→ 涉及不邪淫
#   所以把这些词拆成两类：
#     1. 硬拦截：政治敏感 + 暴力危险（与戒律无关，必须拦截）
#     2. 放行：金融 / 生活类（都是合法的戒律问题，交给 RAG 处理）
# ============================================================

# 硬拦截词表：与戒律完全无关的政治/暴力/危险内容
_HARD_BLOCKED = [
    # 政治敏感
    "习近平", "共产党", "台独", "法轮功",
    # 暴力/危险
    "武器", "炸弹", "恐怖袭击",
]

# 以下词曾经被拦截，但现在确认为正当戒律问题，予以放行：
#   股票、炒股、基金、期货、比特币、房价、彩票
#   → 涉及“正命”（正当生计）和“不偷盗”的延伸讨论
#   黄片、色情
#   → 涉及“不邪淫”
#   赌博
#   → 涉及“不偷盗”的延伸
#   吸毒
#   → 涉及“不饮酒”戒的扩展（一切迷醉性物质）
#   杀人
#   → 涉及“不杀生”（戒律核心戒条）


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
    分层输入审核。

    【小白提示】
    只拦截与戒律完全无关的内容（政治敏感、暴力危险）。
    金融、生活类词汇（炒股、赌博、吸毒等）都是正当的戒律问题，
    放行交给 RAG 系统正常检索回答。

    返回：(is_ok: bool, reason: str)
    """
    for word in _HARD_BLOCKED:
        if word in message:
            return False, f"输入包含与戒律无关的内容「{word}」，请提问与佛教戒律相关的问题。"
    return True, ""


def _check_access_token(token: str) -> bool:
    """检查访问令牌。"""
    if not ACCESS_TOKEN:
        return True
    return token == ACCESS_TOKEN


def _build_chat_history(history: list) -> list:
    """
    将 Gradio history 转为 LLM messages 格式。
    返回: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]
    只取最近 4 条消息（2 轮对话），避免 token 过长。
    """
    if not history:
        return []
    messages = []
    for item in history[-6:]:  # 取最近 6 条（3 轮），后续还会再截到 4 条
        user_msg, bot_msg = None, None
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = str(item[0]), str(item[1])
        elif isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            if role == "user":
                messages.append({"role": "user", "content": content[:300]})
                continue
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content[:300]})
                continue
        else:
            role_attr = getattr(item, "role", "")
            content_attr = getattr(item, "content", "")
            if role_attr == "user":
                messages.append({"role": "user", "content": str(content_attr)[:300]})
                continue
            elif role_attr == "assistant":
                messages.append({"role": "assistant", "content": str(content_attr)[:300]})
                continue
            continue
        if user_msg is not None:
            messages.append({"role": "user", "content": user_msg[:300]})
        if bot_msg is not None:
            messages.append({"role": "assistant", "content": bot_msg[:300]})
    return messages[-4:]  # 最多 4 条（2 轮）


def respond(message: str, history: list, role: str, top_k: int, detail_level: str, json_mode: bool, rerank: bool, deep_think: bool, rewrite: bool, streaming: bool, access_token: str, state_dict: dict):
    """
    Gradio ChatInterface 的回调函数（流式 + 非流式双模式）

    【小白提示】
    这是整个 Web 应用的核心函数！
    每次用户在网页上发送消息，Gradio 就会调用这个函数。
    当 streaming=True 时，它是一个 generator，逐块 yield 回答。
    """
    if not message.strip():
        yield "请输入问题"
        return

    if not _check_access_token(access_token):
        yield "访问令牌错误，请向管理员获取正确的访问令牌。"
        return

    if len(message) > MAX_QUESTION_LENGTH:
        yield f"问题过长，请控制在 {MAX_QUESTION_LENGTH} 字以内。"
        return

    if not _check_global_rate_limit():
        yield "系统请求过于频繁，请稍后再试。"
        return

    # 恢复对话状态
    state = ConversationState.from_dict(state_dict) if state_dict else ConversationState()
    state.turn_count += 1

    if not _check_session_rate_limit(state):
        yield "您请求过于频繁，请稍后再试。"
        return

    # 输入审核
    ok, reason = _check_content_moderation(message)
    if not ok:
        yield reason
        return

    # 下拉框身份优先
    if role and role != "不限":
        state.current_role = role

    # 构造带状态感知的完整问题
    full_question = build_question_with_state(message, history, state)

    # 身份过滤
    role_filter = None if state.current_role == "未指定" else state.current_role

    # 多轮对话历史（传给 LLM 作为 messages）
    chat_history = _build_chat_history(history)

    # 检索
    docs = retrieve(full_question, role_filter=role_filter, k=top_k, rerank=rerank, rewrite=rewrite)

    # 空检索处理
    if not docs:
        if state.current_role and state.current_role != "未指定":
            fallback_suffix = "\n\n---\n\n*知识库未检索到直接相关内容，本次回答来自权威经典兜底。*"
            state.fallback_count += 1
            use_stream = streaming and not json_mode
            if use_stream:
                for partial in generate_stream(full_question, [], role=state.current_role, detail_level=detail_level, json_mode=json_mode, deep_think=deep_think, chat_history=chat_history):
                    yield partial + fallback_suffix
            else:
                result = generate(full_question, [], role=state.current_role, detail_level=detail_level, json_mode=json_mode, deep_think=deep_think, chat_history=chat_history)
                final_answer = _format_answer(result, json_mode) + fallback_suffix
                yield final_answer
            log_qa(message, state.current_role, docs, "权威经典兜底")
            _update_state(state, state_dict, message, "权威经典兜底")
            return
        log_qa(message, state.current_role or role, docs, "未在戒律资料中找到相关内容。")
        _update_state(state, state_dict, message, "未在戒律资料中找到相关内容。")
        yield "未在戒律资料中找到相关内容。"
        return

    # 判断是否权威经典兜底模式
    show_sources = (state.current_role == "未指定") or is_retrieval_relevant(full_question, docs)
    fallback_suffix = "\n\n---\n\n*知识库未检索到直接相关内容，本次回答来自权威经典兜底。*" if not show_sources else ""
    if not show_sources:
        state.fallback_count += 1

    # 检索到的原文（仅相关时展示）
    sources_text = ""
    if show_sources:
        sources_text = "\n\n---\n\n**📚 检索到的原文：**\n"
        for i, doc in enumerate(docs):
            meta = doc.metadata
            sources_text += f"\n**[{i+1}] {meta.get('source','?')} · {meta.get('role','?')}**\n"
            sources_text += f"> {doc.page_content[:200]}\n"

    # 生成
    # JSON 模式强制非流式：流式返回原始文本，无法做友好格式化
    use_stream = streaming and not json_mode
    if use_stream:
        for partial in generate_stream(full_question, docs, role=state.current_role, detail_level=detail_level, json_mode=json_mode, deep_think=deep_think, chat_history=chat_history):
            yield partial + fallback_suffix + sources_text
    else:
        result = generate(full_question, docs, role=state.current_role, detail_level=detail_level, json_mode=json_mode, deep_think=deep_think, chat_history=chat_history)
        final_answer = _format_answer(result, json_mode) + fallback_suffix + sources_text
        yield final_answer

    log_qa(message, state.current_role or role, docs, "已生成")
    _update_state(state, state_dict, message, "已生成")


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
# ============================================================
# UI 构建（Gradio Blocks）
# ============================================================
# Gradio 是一个快速构建 Web UI 的 Python 框架。
# 这里用 gr.Blocks 来自由布局界面，而不是用默认的模板。
#
# 界面结构：
#   - 顶部：标题 + 描述
#   - 控制区：身份下拉框、检索条数、详细程度、JSON模式、Reranker开关
#   - 对话区：ChatInterface（自动提供输入框和消息列表）
#   - 底部：反馈收集区（Accordion 可折叠）
# ============================================================
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
        deep_think_input = gr.Checkbox(
            label="深度思考",
            value=False
        )
        rewrite_input = gr.Checkbox(
            label="智能改写",
            value=False
        )
        streaming_input = gr.Checkbox(
            label="流式输出",
            value=True
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
        additional_inputs=[role_input, topk_input, detail_input, json_mode_input, rerank_input, deep_think_input, rewrite_input, streaming_input, access_token_input, conv_state],
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
    # ============================================================
    # 后台预加载 Reranker 模型
    # ============================================================
    # 在 Gradio 启动前用后台线程加载模型，
    # 这样用户首次勾选“启用精排”时不用再等 2-4 分钟。
    # 模型加载不影响 Web UI 的启动，用户可以同时看到界面。
    # ============================================================
    import threading
    from rag.retriever import _preload_reranker
    print("正在后台预加载 Reranker 模型（不影响界面启动）...")
    _preload_thread = threading.Thread(target=_preload_reranker, daemon=True)
    _preload_thread.start()

    # 端口冲突自动重试：如果 7860 被占用，依次尝试 7861、7862
    for port in [7860, 7861, 7862, 7863]:
        try:
            demo.launch(
                server_name="127.0.0.1",  # 默认仅本机访问，更安全
                server_port=port,
                share=False,  # 设True可生成公网临时链接
                theme=gr.themes.Soft()
            )
            break
        except OSError as e:
            if "Cannot find empty port" in str(e) and port < 7863:
                print(f"端口 {port} 被占用，尝试使用 {port+1}...")
            else:
                raise

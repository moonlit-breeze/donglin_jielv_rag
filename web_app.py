"""
佛学戒律RAG · Web Demo
启动：python web_app.py
访问：http://localhost:7860
"""
import gradio as gr
from rag.retriever import retrieve
from rag.generator import generate

# 身份选项
ROLE_OPTIONS = ["不限", "居士戒", "沙弥戒", "比丘戒"]

def answer(question: str, role: str, top_k: int):
    """Gradio的回调函数"""
    if not question.strip():
        return "请输入问题"

    # 转换角色参数
    role_filter = None if role == "不限" else role

    # 检索
    docs = retrieve(question, role_filter=role_filter, k=top_k)

    if not docs:
        return "未在戒律资料中找到相关内容。"

    # 生成
    answer_text = generate(question, docs)

    # 额外展示检索到的原文（透明化，增加信任感）
    sources = "\n\n---\n\n**📚 检索到的原文：**\n"
    for i, doc in enumerate(docs):
        meta = doc.metadata
        sources += f"\n**[{i+1}] {meta.get('source','?')} · {meta.get('role','?')}**\n"
        sources += f"> {doc.page_content[:200]}\n"

    return answer_text + sources

# ---------- Gradio界面 ----------
with gr.Blocks(title="东林戒律RAG问答", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🪷 佛学戒律智能问答
    *基于RAG技术，严格依据戒律原文回答，不编造。*
    """)

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="请输入您的问题",
                placeholder="例如：居士可以喝酒吗？",
                lines=2
            )
        with gr.Column(scale=1):
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

    submit_btn = gr.Button("🔍 提问", variant="primary", size="lg")

    output = gr.Markdown(label="回答")

    # 绑定事件
    submit_btn.click(
        fn=answer,
        inputs=[question_input, role_input, topk_input],
        outputs=output
    )
    question_input.submit(
        fn=answer,
        inputs=[question_input, role_input, topk_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 允许局域网访问
        server_port=7860,
        share=False  # 设True可生成公网临时链接
    )
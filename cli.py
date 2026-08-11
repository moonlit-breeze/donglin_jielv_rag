"""
cli.py — 命令行问答入口

【小白导读】
  这是系统的最简单的使用方式：在终端里输入问题，得到回答。
  相比 Web 界面，它没有多轮对话状态管理、无限流、无反馈收集，
  但代码流程非常清晰，适合理解 RAG 的基本流程。

  主流程：
    用户输入身份 → 输入问题 → 检索 → 生成回答 → 打印结果

  运行方式：
    python cli.py
"""

from rich.console import Console
from rag.retriever import retrieve
from rag.generator import generate, generate_stream
import openai

console = Console()

# 超过该长度的提问会触发成本/质量警告
# DeepSeek API 按 token 计费，长输入 = 更贵 + 可能超出向量模型处理上限
MAX_QUESTION_CHARS = 300

# ============================================================
# 主循环：不断接收用户输入并回答
# ============================================================
console.print("[bold cyan]佛学戒律问答系统[/bold cyan]")
# ============================================================
# 身份选择：1-居士 2-沙弥 3-比丘，回车=全部检索
# ============================================================
console.print("输入 exit 退出\n")

while True:
    try:
        role_input = console.input("[yellow]身份> [/yellow]").strip()
    except (KeyboardInterrupt, EOFError):
        break

    if role_input.lower() == "exit":
        break

    role_map = {
        "1": "居士戒",
        "2": "沙弥戒",
        "3": "比丘戒"
    }

    if role_input and role_input not in role_map:
        console.print("[red]无效身份，请输入 1、2、3 或回车[/red]")
        continue

    role_filter = role_map.get(role_input)

    question = console.input("[green]问> [/green]").strip()
    if not question:
        continue

    if len(question) > MAX_QUESTION_CHARS:
        console.print(
            f"[yellow]提示：输入过长（{len(question)} 字符）。"
            "API 按 token 计费，长输入会增加调用成本，"
            "且可能超出向量模型处理上限、降低检索质量。[/yellow]"
        )
        confirm = console.input("[yellow]继续请按 y，放弃请按 n> [/yellow]").strip().lower()
        if confirm != "y":
            continue

    # ============================================================
    # 检索：调用 retriever 模块在向量库中查找相关文档
    # ============================================================
    console.print("[yellow]正在检索戒律...[/yellow]")

    try:
        docs = retrieve(question, role_filter=role_filter, k=3)
    except Exception as e:
        console.print(f"[red]检索失败：{e}[/red]")
        continue

    # ============================================================
    # 空检索处理：如果指定了身份，允许权威经典兜底
    # ============================================================
    if not docs:
        if role_filter:
            # 指定了身份但知识库未命中，允许权威经典兜底
            try:
                answer = generate(question, [], role=role_filter)
                console.print("\n" + answer + "\n")
                console.print("[dim]— 知识库未检索到直接相关内容，本次回答来自权威经典兜底。[/dim]\n")
            except Exception as e:
                console.print(f"[red]生成回答时出错：{e}[/red]")
            continue
        console.print(
            "[red]未检索到相关戒律条文。[/red]\n"
            "[dim]该问题可能超出戒律知识库范围，或相关度不足已被过滤。[/dim]"
        )
        continue

    # ============================================================
    # 生成回答：调用 generator 模块把检索结果 + 问题发给大模型
    # ============================================================
    # CLI 默认使用流式输出，用户体验更好
    try:
        console.print("\n[green]答>[/green] ", end="")
        answer = ""
        for partial in generate_stream(question, docs, role=role_filter or "未指定"):
            answer = partial
        console.print(answer)
    except KeyboardInterrupt:
        console.print("\n[red]已取消本次请求[/red]")
        continue
    except openai.AuthenticationError:
        console.print("[red]API Key 无效，请检查 .env 中的 DEEPSEEK_API_KEY[/red]")
        continue
    except (openai.APIConnectionError, openai.APITimeoutError):
        console.print("[red]无法连接 DeepSeek API，请检查网络后重试[/red]")
        continue
    except openai.RateLimitError:
        console.print("[red]请求过于频繁或账户额度不足，请稍后重试[/red]")
        continue
    except openai.APIError as e:
        console.print(f"[red]API 调用失败：{e}[/red]")
        continue
    except Exception as e:
        console.print(f"[red]生成回答时出错：{e}[/red]")
        continue

    # 自动追加免责声明
    disclaimer = "\n\n[dim]— 内容仅供参考，以丛林规约为准。[/dim]"
    console.print("\n" + (answer or "（模型未返回内容）") + disclaimer + "\n")
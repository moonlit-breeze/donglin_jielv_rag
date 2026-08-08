from rich.console import Console
from rag.retriever import retrieve
from rag.generator import generate
import openai

console = Console()

# 超过该长度的提问触发成本/质量警告（DeepSeek API 按 token 计费）
MAX_QUESTION_CHARS = 300

console.print("[bold cyan]佛学戒律问答系统[/bold cyan]")
console.print("输入身份：1-居士 2-沙弥 3-比丘，回车=全部检索")
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

    console.print("[yellow]正在检索戒律...[/yellow]")

    try:
        docs = retrieve(question, role_filter=role_filter, k=3)
    except Exception as e:
        console.print(f"[red]检索失败：{e}[/red]")
        continue

    if not docs:
        console.print(
            "[red]未检索到相关戒律条文。[/red]\n"
            "[dim]该问题可能超出戒律知识库范围，或相关度不足已被过滤。[/dim]"
        )
        continue

    try:
        answer = generate(question, docs, role=role_filter or "未指定")
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
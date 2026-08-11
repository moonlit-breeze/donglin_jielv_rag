# 佛学戒律 RAG 问答系统

基于 **RAG（Retrieval-Augmented Generation）** 技术的佛教戒律智能问答系统。

以居士"净林"的身份，依据戒律经典文本，为学佛同修解答戒律相关问题。

## 技术架构

| 模块 | 技术选型 | 说明 |
|------|---------|------|
| 文本处理 | LangChain RecursiveCharacterTextSplitter | chunk_size=800, chunk_overlap=150 |
| 向量嵌入 | HuggingFace BAAI/bge-small-zh（本地） | 离线运行，无需联网 |
| 向量数据库 | Chroma（按身份分层存储） | 比丘戒 / 沙弥戒 / 居士戒 各自独立 |
| 检索策略 | 语义检索 + 同义词扩展 + 关键词召回 + RRF 融合 | 多路召回，语义优先 |
| 精排模型 | BAAI/bge-reranker-v2-m3（可选） | Cross-encoder 二次打分，提升相关性 |
| 生成模型 | DeepSeek Chat API | 支持结构化 JSON / Markdown 两种输出 |
| 前端界面 | Gradio 6.0 (ChatInterface) | 多轮对话、身份切换、反馈收集 |

## 核心功能

- **身份分层检索**：根据用户身份（居士/沙弥/比丘）只查询对应知识库，杜绝跨域混淆
- **多轮对话管理**：自动追踪身份切换、追问主题，上下文感知
- **Reranker 精排**（可选）：cross-encoder 对候选文档二次打分，将更相关内容排到前面
- **权威经典兜底**：知识库未覆盖时，引用公认权威佛教典籍谨慎回答，并明确标注来源
- **跨域拦截**：非戒律问题（炒股、天气等）直接拒答
- **用户反馈收集**：差评自动归入 `bad_cases.json`，便于后续补库
- **问答日志分析**：记录所有 Q&A，支持统计兜底率、差评率、高频未覆盖问题
- **访问控制**：可选配置 ACCESS_TOKEN，限制访问权限
- **输入审核**：敏感词过滤 + 限流保护

## 项目结构

```
donglin_jielv_rag/
├── .env                       # 环境变量（DEEPSEEK_API_KEY, ACCESS_TOKEN）
├── .env.example               # 环境变量示例
├── .gitignore
├── requirements.txt           # Python 依赖
├── data/
│   ├── jielv.txt              # 原始戒律文本
│   └── knowledge_base.json    # 结构化知识库（由 ingest.py 生成）
├── models/
│   └── bge-small-zh/          # 本地嵌入模型
├── rag/
│   ├── __init__.py
│   ├── loader.py              # 知识库加载（JSON → LangChain Document）
│   ├── vector_store.py        # 向量库构建与加载（Chroma + bge-small-zh）
│   ├── retriever.py           # 检索模块（语义 + 关键词 + RRF + Reranker）
│   ├── generator.py           # 答案生成（DeepSeek API + 权威经典兜底）
│   ├── conversation.py        # 多轮对话状态管理（身份追踪、追问消歧）
│   ├── logger.py              # 问答日志 + 反馈记录 + 日志分析
│   └── pdf_loader.py          # PDF 文档解析（用于 ingest.py）
├── ingest.py                  # PDF 戒律文档导入脚本（提取 → JSON → 建库）
├── init_db.py                 # 从 knowledge_base.json 初始化向量数据库
├── web_app.py                 # Gradio Web 界面（多轮对话版）
├── cli.py                     # 命令行问答入口
├── tests/
│   ├── test_qa.py             # 检索 + 生成回归测试
│   └── test_reranker_comparison.py  # Reranker 精排对比实验
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**：Reranker 模型（bge-reranker-v2-m3，约 1.1GB）在首次使用时自动下载。
> 如果处于内网/代理环境，可参考 `_download_reranker.py` 中的离线下载方案。

### 2. 配置 API Key

编辑 `.env` 文件：

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
# 可选：设置访问令牌
# ACCESS_TOKEN=your_token_here
```

### 3. 导入戒律文本 & 构建向量索引

**方式一：从 PDF 导入（推荐）**

将戒律 PDF 放入项目目录，执行：

```bash
python ingest.py data/jielv.pdf           # 默认覆盖重建
python ingest.py --merge data/jielv.pdf   # 与现有知识库合并追加
python ingest.py --preview data/jielv.pdf # 预览提取结果，不保存
```

**方式二：从已有 JSON 重建向量库**

```bash
python init_db.py
```

### 4. 启动问答

**Web 界面（推荐）：**

```bash
python web_app.py
```

启动后访问 `http://localhost:7860`，支持：
- 身份选择（不限 / 居士戒 / 沙弥戒 / 比丘戒）
- 多轮追问与身份切换
- 检索条数、回答详细程度调节
- 结构化 JSON 输出模式
- Reranker 精排开关
- 回答反馈（有帮助 / 无帮助 / 部分正确）

**命令行：**

```bash
python cli.py
```

按提示输入身份（1-居士 2-沙弥 3-比丘，回车=全部检索），然后输入问题即可。

## 测试

```bash
# 只跑检索测试（不耗 API）
python tests/test_qa.py

# 同时跑生成测试（需 DEEPSEEK_API_KEY）
python tests/test_qa.py --with-llm

# Reranker 精排对比实验
python tests/test_reranker_comparison.py
```

## 日志分析

系统自动记录所有问答到 `logs/qa.log`，可运行日志分析：

```bash
python rag/logger.py
```

输出兜底率、差评率、高频未覆盖问题 TOP10 等统计报告。

## 注意事项

- 请确保 `.env` 中的 `DEEPSEEK_API_KEY` 有效且有可用额度。
- 修改知识库文本后，需重新执行 `python init_db.py` 以更新向量索引。
- 向量数据库存储在 `chroma_db/` 目录（已加入 .gitignore）。
- 嵌入模型（bge-small-zh）和精排模型（bge-reranker-v2-m3）均在本地离线运行。
- Reranker 首次加载耗时较长（约 2-4 分钟），后续查询复用已加载模型。

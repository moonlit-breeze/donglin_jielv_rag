# 佛学戒律 RAG 问答系统

基于 **RAG（Retrieval-Augmented Generation）** 技术的佛教戒律智能问答系统。

以居士“净林”的身份，依据戒律经典文本，为学佛同修解答戒律相关问题。

## 技术架构

| 模块 | 技术选型 | 说明 |
|------|---------|------|
| 文本处理 | LangChain RecursiveCharacterTextSplitter | chunk_size=800, chunk_overlap=150 |
| 向量嵌入 | HuggingFace BAAI/bge-small-zh（本地） | 离线运行，无需联网 |
| 向量数据库 | Chroma（按身份分层存储） | 比丘戒 / 沙弥戒 / 居士戒 各自独立 |
| 检索策略 | 语义检索 + 同义词扩展 + 关键词召回 + RRF 融合 + 检索缓存 | 多路召回，语义优先，缓存加速 |
| 精排模型 | BAAI/bge-reranker-v2-m3（可选） | Cross-encoder 二次打分，LRU 缓存 |
| LLM 调用层 | 抽象 Provider 模式 | 支持 DeepSeek / OpenAI / SiliconFlow，流式 + 非流式 |
| 生成模型 | DeepSeek Chat API（默认） | 流式输出、深度思考、JSON 结构化输出 |
| 前端界面 | Gradio 6.0 (ChatInterface) | 多轮对话、流式打字、身份切换、反馈收集 |
| 部署方式 | Docker + docker-compose | 一键部署，volume 持久化 |
| 质量评估 | LLM-as-Judge 自动评估 | 四维打分（准确性、来源、格式、聚焦） |

## 核心功能

- **身份分层检索**：根据用户身份（居士/沙弥/比丘）只查询对应知识库，杜绝跨域混淆
- **流式输出**：回答逐字“打字”展示，消除 5-10 秒白屏等待
- **多模型切换**：支持 DeepSeek / OpenAI / SiliconFlow，切换模型只需改 `.env`
- **深度思考（CoT）**：先分析戒条类型和适用范围，再给出结论，更准确
- **多轮对话管理**：对话历史作为 messages 注入 LLM，追问更自然
- **Reranker 精排**（可选）：cross-encoder 对候选文档二次打分，LRU 缓存加速
- **智能改写**（可选）：LLM 将口语转为规范检索词，提升召回率
- **检索结果缓存**：相同问题 5 分钟内秒回，无需重复检索
- **权威经典兜底**：知识库未覆盖时，引用公认权威佛教典籍谨慎回答
- **跨域拦截**：非戒律问题（天气等）直接拒答；生活类词汇（炒股、赌博等）作为正当戒律问题放行
- **用户反馈收集**：差评自动归入 `bad_cases.json`，便于后续补库
- **日志自动分析**：自动挖掘 bad case，生成 Markdown 周报
- **LLM-as-Judge 评估**：四维自动评分（准确性、来源引用、格式、身份聚焦）
- **访问控制**：可选 ACCESS_TOKEN + 双重限流 + 输入审核
- **Docker 部署**：`docker-compose up` 一键启动

## 项目结构

```
donglin_jielv_rag/
├── .env                       # 环境变量（DEEPSEEK_API_KEY / OPENAI_API_KEY / SILICONFLOW_API_KEY）
├── .env.example               # 环境变量示例（含三种 LLM 提供商配置）
├── .gitignore
├── .dockerignore              # Docker 构建排除规则
├── requirements.txt           # Python 依赖
├── Dockerfile                 # Docker 容器化构建
├── docker-compose.yml         # Docker 一键部署
├── REVIEW.md                  # 项目复盘与演进记录
├── data/
│   ├── jielv.txt              # 原始戒律文本
│   └── knowledge_base.json    # 结构化知识库（由 ingest.py 生成）
├── models/
│   └── bge-small-zh/          # 本地嵌入模型
├── rag/
│   ├── __init__.py
│   ├── loader.py              # 知识库加载（JSON → LangChain Document）
│   ├── vector_store.py        # 向量库构建与加载（Chroma + bge-small-zh）
│   ├── retriever.py           # 检索模块（语义 + 关键词 + RRF + Reranker + 缓存 + 改写）
│   ├── llm_client.py          # LLM 统一调用层（Provider 抽象 + 流式 + 重试）
│   ├── generator.py           # 答案生成（流式/非流式 + 深度思考 + 多轮上下文）
│   ├── conversation.py        # 多轮对话状态管理（身份追踪、追问消歧）
│   ├── logger.py              # 问答日志 + 自动 bad case 挖掘 + 周报生成
│   ├── evaluator.py           # LLM-as-Judge 自动评估（四维打分）
│   └── pdf_loader.py          # PDF 文档解析（用于 ingest.py）
├── ingest.py                  # PDF/TXT 戒律文档导入脚本
├── init_db.py                 # 从 knowledge_base.json 初始化向量数据库
├── web_app.py                 # Gradio Web 界面（流式 + 多轮对话）
├── cli.py                     # 命令行问答入口（流式输出）
├── tests/
│   ├── test_qa.py             # 回归测试（32 项：检索 + 阈值 + 去重 + 缓存 + Provider + 生成 + 流式）
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

编辑 `.env` 文件（支持三种 LLM 提供商，任选其一）：

```ini
# 方式一：DeepSeek（默认推荐）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx

# 方式二：通用 OpenAI 兼容 API（适合各种中转站）
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4o-mini

# 方式三：SiliconFlow（国内多模型聚合平台）
# SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxx
# SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V3

# 可选：访问令牌
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
- 流式输出（默认开启，逐字“打字”效果）
- 深度思考（CoT 模式，先分析再回答）
- 智能改写（LLM 将口语转规范检索词）
- 多轮追问与身份切换
- 检索条数、回答详细程度调节
- 结构化 JSON 输出模式
- Reranker 精排开关
- 回答反馈（有帮助 / 无帮助 / 部分正确）

**命令行：**

```bash
python cli.py
```

按提示输入身份（1-居士 2-沙弥 3-比丘，回车=全部检索），然后输入问题即可。支持流式输出。

**Docker 部署：**

```bash
docker-compose up -d
```

自动构建镜像并启动服务，访问 `http://localhost:7860`。

## 测试

```bash
# 只跑免费测试（检索 23 + 阈值 4 + 去重 1 + 缓存 3 + Provider 1 = 32 项）
python tests/test_qa.py

# 同时跑生成 + 流式测试（需 API Key）
python tests/test_qa.py --with-llm

# Reranker 精排对比实验
python tests/test_reranker_comparison.py
```

## 日志分析

系统自动记录所有问答到 `logs/qa.log`，支持多种分析模式：

```bash
# 基础统计报告（兜底率、差评率、高频未覆盖问题 TOP10）
python rag/logger.py

# 自动挖掘 bad case（空检索 + 兜底回答 → data/bad_cases.json）
python rag/logger.py --auto-check

# 生成 Markdown 周报
python rag/logger.py --report
```

## 自动评估（LLM-as-Judge）

用大语言模型作为“评委”，从准确性、来源引用、格式完整性、身份聚焦四个维度对回答质量打分：

```bash
# 评估预定义测试用例（端到端）
python rag/evaluator.py --test-cases

# 评估最近的问答日志
python rag/evaluator.py

# 评估并保存结果
python rag/evaluator.py --test-cases --save
```

## 注意事项

- 请确保 `.env` 中的 API Key 有效且有可用额度（支持 DeepSeek / OpenAI / SiliconFlow 三种提供商）。
- 修改知识库文本后，需重新执行 `python init_db.py` 以更新向量索引。
- 向量数据库存储在 `chroma_db/` 目录（已加入 .gitignore）。
- 嵌入模型（bge-small-zh）和精排模型（bge-reranker-v2-m3）均在本地离线运行。
- Reranker 首次加载耗时较长（约 2-4 分钟），系统会在启动时后台预加载。
- 检索结果缓存有效期 5 分钟，相同问题重复查询秒回。

## 知识库覆盖说明

本系统的知识库以居士戒（在家众）为核心，覆盖五戒、八戒、十善、菩萨戒等内容，并引用《四分律》《梵网经》《楞严经》及印光大师、大安法师等公开开示。

沙弥戒与比丘戒（出家众）仅收录最基础的公开条目。汉传佛教传统中，比丘具足戒的戒本不对在家人公开，因此完整的出家戒内容不适合以公开形态呈现在代码仓库中。系统已预留以下本地补充接口，实际使用者可按需自行扩充：

- `data/jielv.txt` — 按「【身份】正文」格式追加条文
- `python ingest.py data/xxx.pdf` — 从 PDF 导入
- `python init_db.py` — 重建向量索引

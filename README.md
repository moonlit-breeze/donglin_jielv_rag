# 东林戒律 RAG 问答系统

基于 **RAG（Retrieval-Augmented Generation）** 技术的佛教戒律智能问答系统。

以居士"净林"的身份，依据戒律经典文本，为学佛同修解答戒律相关问题。

## 技术架构

- **文本处理**：LangChain RecursiveCharacterTextSplitter（chunk_size=800, chunk_overlap=150）
- **向量嵌入**：DeepSeek Embedding（text-embedding-3-small）
- **向量数据库**：Chroma
- **生成模型**：DeepSeek Chat API

## 项目结构

```
donglin_jielv_rag/
├── .env                  # 环境变量（DEEPSEEK_API_KEY）
├── .env.example          # 环境变量示例
├── .gitignore
├── requirements.txt      # Python 依赖
├── data/
│   └── jielv.txt         # 戒律文本（待放入）
├── rag/
│   ├── __init__.py
│   ├── loader.py         # 文本加载与分割
│   ├── vector_store.py   # 向量库构建与加载
│   ├── retriever.py      # 相似度检索
│   └── generator.py      # DeepSeek 生成回答
├── cli.py                # 命令行入口
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `.env` 文件，填入你的 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 3. 放入戒律文本

将佛教戒律文本放入 `data/jielv.txt`。

### 4. 构建向量索引

```bash
python cli.py build-index
```

### 5. 开始问答

单次问答：

```bash
python cli.py ask "不杀生戒的具体内容是什么？"
```

交互式对话：

```bash
python cli.py interactive
```

## 使用说明

| 命令 | 说明 |
|------|------|
| `python cli.py ask "问题"` | 单次问答 |
| `python cli.py interactive` | 进入交互式对话，输入 exit 退出 |
| `python cli.py build-index` | 构建向量索引（修改文本后需重新执行） |

## 注意事项

- 请确保 `.env` 中的 `DEEPSEEK_API_KEY` 有效且有可用额度。
- 修改 `data/jielv.txt` 后，需要重新执行 `build-index` 以更新索引。
- 向量数据库存储在 `vector_db/` 目录（已加入 .gitignore）。

"""临时脚本：下载 bge-reranker-v2-m3 模型（绕过 SSL 验证）"""
import httpx

# Monkey-patch httpx.Client 以禁用 SSL 验证
_orig_client = httpx.Client
class _NoVerifyClient(_orig_client):
    def __init__(self, *a, **kw):
        kw["verify"] = False
        super().__init__(*a, **kw)
httpx.Client = _NoVerifyClient

_orig_async = httpx.AsyncClient
class _NoVerifyAsync(_orig_async):
    def __init__(self, *a, **kw):
        kw["verify"] = False
        super().__init__(*a, **kw)
httpx.AsyncClient = _NoVerifyAsync

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

print("正在下载 BAAI/bge-reranker-v2-m3（约 1.1GB，请耐心等待）...")

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir=None,  # 下载到默认缓存目录
)
print("模型文件下载完成！")

from FlagEmbedding import FlagReranker
r = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
print("模型加载成功！")

# 验证
scores = r.compute_score([["可以吃饭吗", "佛教戒律不禁止正常饮食"]], normalize=True)
print(f"验证分数：{scores}")
print("一切正常。")

"""
llm_client.py — LLM 统一调用层

职责：
  抽象不同大语言模型提供商的 API 调用，支持：
  1. 多模型切换（DeepSeek / 通用 OpenAI / SiliconFlow）
  2. 流式输出（逐 token 返回，用户不必等完整回答）
  3. 多轮对话历史注入（chat history 作为 messages 传给 API）
  4. 失败重试与超时控制

【小白导读】
  这个文件把"调用大模型"这件事封装成一个统一接口。
  不管底层用的是 DeepSeek、OpenAI 还是其他模型，
  上层代码（generator.py）只需要调用相同的接口。

  好处：
  - 换模型不用改业务代码，只需改 .env 配置
  - 流式/非流式用同一套代码
  - 重试逻辑集中管理
"""

import os
import time
from typing import List, Dict, Iterator, Optional
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Provider 基类与实现
# ============================================================

class LLMProvider:
    """LLM 提供商抽象接口。"""

    name: str = "base"

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1,
             timeout: float = 60.0) -> str:
        """同步调用，返回完整文本。"""
        raise NotImplementedError

    def chat_stream(self, messages: List[Dict[str, str]], temperature: float = 0.1,
                    timeout: float = 60.0) -> Iterator[str]:
        """流式调用，逐块 yield 文本片段。"""
        raise NotImplementedError


class DeepSeekProvider(LLMProvider):
    """DeepSeek API（OpenAI 兼容接口）。"""

    name = "deepseek"

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat"):
        from openai import OpenAI
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages, temperature=0.1, timeout=60.0):
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(self, messages, temperature=0.1, timeout=60.0):
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


class GenericOpenAIProvider(LLMProvider):
    """通用 OpenAI 兼容 API（适用于各种中转站 / 自建部署）。"""

    name = "openai"

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages, temperature=0.1, timeout=60.0):
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(self, messages, temperature=0.1, timeout=60.0):
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


class SiliconFlowProvider(LLMProvider):
    """SiliconFlow API（国内多模型聚合平台）。"""

    name = "siliconflow"

    def __init__(self, api_key: str, model: str = "deepseek-ai/DeepSeek-V3"):
        from openai import OpenAI
        self._model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1",
        )

    def chat(self, messages, temperature=0.1, timeout=60.0):
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(self, messages, temperature=0.1, timeout=60.0):
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


# ============================================================
# 工厂函数：根据环境变量或显式参数创建 Provider
# ============================================================
# 使用方式：
#   1. 不传参数：按环境变量自动选择（向后兼容）
#   2. 传 provider_name/model：由调用方（如 Web UI）指定模型
#
# 优先级：
#   1. 函数参数 provider_name / model
#   2. LLM_PROVIDER 环境变量
#   3. SILICONFLOW_API_KEY → SiliconFlow
#   4. OPENAI_API_KEY + OPENAI_BASE_URL → 通用 OpenAI
#   5. DEEPSEEK_API_KEY → DeepSeek（默认）
# ============================================================

# 供 UI 展示的 provider 选项
AVAILABLE_PROVIDERS = ["deepseek", "openai", "siliconflow"]

# 各 provider 的默认模型（用户未指定 model 时使用）
DEFAULT_MODELS = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o-mini",
    "siliconflow": "deepseek-ai/DeepSeek-V3",
}


def create_provider(provider_name: str = None, model: str = None) -> LLMProvider:
    """
    创建 LLM Provider。

    参数：
      provider_name: 可选，指定 provider（deepseek/openai/siliconflow）
      model:         可选，指定模型名；不传则使用对应 provider 的默认模型
    """
    name = (provider_name or os.getenv("LLM_PROVIDER", "")).lower()

    if name == "siliconflow" or (not name and os.getenv("SILICONFLOW_API_KEY")):
        key = os.getenv("SILICONFLOW_API_KEY", "")
        if not key:
            raise ValueError("使用 SiliconFlow 需要设置 SILICONFLOW_API_KEY")
        selected_model = model or os.getenv("SILICONFLOW_MODEL", DEFAULT_MODELS["siliconflow"])
        return SiliconFlowProvider(key, model=selected_model)

    if name == "openai" or (not name and os.getenv("OPENAI_API_KEY")):
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("使用 OpenAI 需要设置 OPENAI_API_KEY")
        url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        selected_model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODELS["openai"])
        return GenericOpenAIProvider(key, base_url=url, model=selected_model)

    if name == "deepseek" or (not name and os.getenv("DEEPSEEK_API_KEY")):
        key = os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError("使用 DeepSeek 需要设置 DEEPSEEK_API_KEY")
        selected_model = model or os.getenv("DEEPSEEK_MODEL", DEFAULT_MODELS["deepseek"])
        return DeepSeekProvider(key, model=selected_model)

    raise ValueError(
        "未找到任何 LLM API Key。请在 .env 中设置以下任一：\n"
        "  DEEPSEEK_API_KEY（默认推荐）\n"
        "  OPENAI_API_KEY + OPENAI_BASE_URL\n"
        "  SILICONFLOW_API_KEY"
    )


# ============================================================
# 带重试的调用封装
# ============================================================

def call_with_retry(provider: LLMProvider, messages: List[Dict[str, str]],
                    temperature: float = 0.1, timeout: float = 60.0,
                    max_retries: int = 2) -> str:
    """同步调用，失败自动重试。"""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return provider.chat(messages, temperature=temperature, timeout=timeout)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
    raise last_error


def stream_with_retry(provider: LLMProvider, messages: List[Dict[str, str]],
                      temperature: float = 0.1, timeout: float = 60.0,
                      max_retries: int = 2) -> Iterator[str]:
    """流式调用，失败自动重试（仅对首次连接失败重试）。"""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            for chunk in provider.chat_stream(messages, temperature=temperature, timeout=timeout):
                yield chunk
            return  # 成功完成
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
    raise last_error

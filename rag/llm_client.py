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
# 工厂函数：根据环境变量自动选择 Provider
# ============================================================
# 优先级：
#   1. LLM_PROVIDER 显式指定
#   2. SILICONFLOW_API_KEY → SiliconFlow
#   3. OPENAI_API_KEY + OPENAI_BASE_URL → 通用 OpenAI
#   4. DEEPSEEK_API_KEY → DeepSeek（默认）
# ============================================================

def create_provider() -> LLMProvider:
    """根据环境变量创建 LLM Provider。"""
    provider_name = os.getenv("LLM_PROVIDER", "").lower()

    if provider_name == "siliconflow":
        key = os.getenv("SILICONFLOW_API_KEY", "")
        model = os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3")
        if key:
            return SiliconFlowProvider(key, model=model)
        raise ValueError("LLM_PROVIDER=siliconflow 但 SILICONFLOW_API_KEY 未设置")

    if provider_name == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if key:
            return GenericOpenAIProvider(key, base_url=url, model=model)
        raise ValueError("LLM_PROVIDER=openai 但 OPENAI_API_KEY 未设置")

    if provider_name == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY", "")
        if key:
            return DeepSeekProvider(key)
        raise ValueError("LLM_PROVIDER=deepseek 但 DEEPSEEK_API_KEY 未设置")

    # 自动探测：按优先级依次尝试
    if os.getenv("SILICONFLOW_API_KEY"):
        model = os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3")
        return SiliconFlowProvider(os.getenv("SILICONFLOW_API_KEY"), model=model)

    if os.getenv("OPENAI_API_KEY"):
        url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return GenericOpenAIProvider(os.getenv("OPENAI_API_KEY"), base_url=url, model=model)

    if os.getenv("DEEPSEEK_API_KEY"):
        return DeepSeekProvider(os.getenv("DEEPSEEK_API_KEY"))

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

"""LLM Factory - OpenAI와 Upstage 모델 선택 및 생성"""
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage

from src.config.settings import settings

# Retry configuration for API calls
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def create_llm(
    provider: Literal["openai", "upstage"] = None,
    temperature: float = 0.7,
    model: str = None
):
    """
    LLM 인스턴스를 생성합니다.

    Args:
        provider: "openai" 또는 "upstage". None이면 settings.default_llm_provider 사용
        temperature: LLM temperature (0.0 ~ 1.0)
        model: 사용할 모델명. None이면 provider별 기본 모델 사용

    Returns:
        ChatOpenAI 또는 ChatUpstage 인스턴스

    Examples:
        # 기본 설정 사용
        llm = create_llm()

        # OpenAI 명시적 선택
        llm = create_llm(provider="openai", temperature=0.5)

        # Upstage 선택
        llm = create_llm(provider="upstage", temperature=0.3)

        # 커스텀 모델
        llm = create_llm(provider="openai", model="gpt-4")
    """
    # Provider 결정
    if provider is None:
        provider = settings.default_llm_provider

    # 모델명 결정
    if model is None:
        if provider == "openai":
            model = settings.openai_model
        elif provider == "upstage":
            model = settings.upstage_model
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # LLM 인스턴스 생성 with retry configuration
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=settings.openai_api_key,
            max_retries=MAX_RETRIES,
            timeout=60,
            request_timeout=60
        )
    elif provider == "upstage":
        return ChatUpstage(
            model=model,
            temperature=temperature,
            api_key=settings.upstage_api_key,
            max_retries=MAX_RETRIES,
            timeout=60
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'upstage'.")


def get_available_providers() -> list[str]:
    """
    사용 가능한 LLM provider 목록을 반환합니다.

    Returns:
        ["openai", "upstage"]
    """
    return ["openai", "upstage"]


def get_default_provider() -> str:
    """
    현재 기본 LLM provider를 반환합니다.

    Returns:
        "openai" 또는 "upstage"
    """
    return settings.default_llm_provider

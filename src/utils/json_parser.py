"""JSON parsing utilities with validation and retry logic"""
import json
import re
from typing import Dict, Any, Optional, Callable
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_json_from_text(text: str) -> Optional[str]:
    """
    텍스트에서 JSON 블록 추출

    LLM이 추가 설명과 함께 JSON을 반환하는 경우 처리:
    예: "여기 결과입니다: ```json\n{...}\n```"

    Args:
        text: 원시 LLM 출력

    Returns:
        추출된 JSON 문자열, 없으면 None
    """
    # 1. JSON 코드 블록 패턴 (```json ... ``` 또는 ``` ... ```)
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. 중괄호로 시작하는 JSON 객체 찾기
    brace_pattern = r'\{.*\}'
    match = re.search(brace_pattern, text, re.DOTALL)
    if match:
        return match.group(0).strip()

    # 3. 추출 실패 - 원본 반환
    return text.strip()


def validate_agent_output(data: Dict[str, Any], required_fields: list[str]) -> bool:
    """
    에이전트 출력 JSON 검증

    Args:
        data: 파싱된 JSON 딕셔너리
        required_fields: 필수 필드 리스트

    Returns:
        검증 성공 여부
    """
    if not isinstance(data, dict):
        return False

    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing required field: {field}")
            return False

        # content 필드는 비어있으면 안 됨
        if field == "content" and not data.get("content"):
            logger.warning("Content field is empty")
            return False

    return True


async def parse_agent_json_with_retry(
    llm_chain: Callable,
    chain_input: Dict[str, Any],
    required_fields: list[str],
    max_retries: int = 2,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    JSON 파싱 재시도 로직

    Args:
        llm_chain: LLM 체인 (ainvoke 가능한 객체)
        chain_input: 체인 입력 데이터
        required_fields: 필수 필드 리스트
        max_retries: 최대 재시도 횟수
        config: LangSmith config

    Returns:
        파싱된 JSON 딕셔너리

    Raises:
        ValueError: 모든 재시도 실패 시
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # LLM 호출
            response = await llm_chain.ainvoke(chain_input, config=config)
            raw_content = response.content

            # JSON 추출
            extracted_json = extract_json_from_text(raw_content)

            # JSON 파싱
            parsed_data = json.loads(extracted_json)

            # 검증
            if validate_agent_output(parsed_data, required_fields):
                if attempt > 0:
                    logger.info(f"JSON parsing succeeded on retry {attempt}")
                return parsed_data
            else:
                raise ValueError(f"Validation failed: missing required fields")

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(
                f"JSON parsing attempt {attempt + 1}/{max_retries + 1} failed",
                error=str(e),
                raw_output=raw_content[:200] if 'raw_content' in locals() else "N/A"
            )

            # 마지막 시도가 아니면 재시도
            if attempt < max_retries:
                continue

    # 모든 재시도 실패
    raise ValueError(f"JSON parsing failed after {max_retries + 1} attempts: {last_error}")


def create_fallback_agent_output(
    role: str,
    error_message: str = "죄송합니다. 응답 생성 중 오류가 발생했습니다."
) -> Dict[str, Any]:
    """
    폴백 에이전트 출력 생성

    Args:
        role: 에이전트 역할
        error_message: 사용자에게 표시할 에러 메시지

    Returns:
        기본 AgentOutput 딕셔너리
    """
    return {
        "role": role,
        "content": error_message,
        "emotion": "neutral",
        "references": [],
        "_quality_degraded": True
    }

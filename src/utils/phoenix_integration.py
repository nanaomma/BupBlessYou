"""
Arize Phoenix integration for LLM Observability
Phoenix 통합을 위한 준비 모듈 (선택적 활성화)
"""
from typing import Optional, Dict, Any
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Phoenix 설치 여부 확인용 플래그
PHOENIX_AVAILABLE = False
tracer_provider = None

try:
    from openinference.instrumentation.langchain import LangChainInstrumentor
    # Phoenix Cloud (Arize) 공식 패키지
    from arize.otel import register
    PHOENIX_AVAILABLE = True
    PHOENIX_MODE = "arize"  # Arize Phoenix Cloud
except ImportError:
    # Local Phoenix fallback
    try:
        from phoenix.otel import register
        PHOENIX_AVAILABLE = True
        PHOENIX_MODE = "local"  # Local Phoenix server
    except ImportError:
        PHOENIX_AVAILABLE = False
        PHOENIX_MODE = None
        logger.debug("Phoenix packages not installed, tracing will be disabled")


def setup_phoenix() -> bool:
    """
    Phoenix 트레이싱 설정 (Arize Phoenix Cloud 또는 Local Phoenix)

    Returns:
        bool: Phoenix가 성공적으로 활성화되었는지 여부

    Configuration (in .env):
        Local Phoenix:
            PHOENIX_ENABLED=true
            PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006

        Phoenix Cloud (Arize 공식):
            PHOENIX_ENABLED=true
            PHOENIX_SPACE_ID=U3BhY2U6MzQ1OTI6S2NIbg==  # From Arize Phoenix dashboard
            PHOENIX_API_KEY=your_phoenix_cloud_api_key
            # PHOENIX_COLLECTOR_ENDPOINT는 자동 설정됨

    Note:
        Arize Phoenix Cloud 사용 시:
        - arize.otel.register 사용 (space_id + api_key)

        Local Phoenix 사용 시:
        - phoenix.otel.register 사용 (endpoint만)
    """
    global tracer_provider

    if not PHOENIX_AVAILABLE:
        logger.info("Phoenix not available (packages not installed)")
        return False

    if not settings.phoenix_enabled:
        logger.info("Phoenix tracing disabled (set PHOENIX_ENABLED=true in .env to enable)")
        return False

    try:
        # Phoenix Cloud (Arize) 사용 시
        if settings.phoenix_space_id and settings.phoenix_api_key:
            logger.info("Phoenix Cloud (Arize) mode detected")

            # Arize Phoenix Cloud 등록
            tracer_provider = register(
                space_id=settings.phoenix_space_id,
                api_key=settings.phoenix_api_key,
                project_name=settings.langsmith_project
            )

            phoenix_type = "Cloud (Arize)"
            endpoint = "https://app.phoenix.arize.com"

        # Local Phoenix 사용 시
        else:
            logger.info("Local Phoenix mode detected")

            base_endpoint = settings.phoenix_collector_endpoint

            # OTLP 엔드포인트 구성 (자동으로 /v1/traces 추가)
            if not base_endpoint.endswith("/v1/traces"):
                endpoint = f"{base_endpoint}/v1/traces"
            else:
                endpoint = base_endpoint

            # Local Phoenix 등록
            tracer_provider = register(
                project_name=settings.langsmith_project,
                endpoint=endpoint
            )

            phoenix_type = "Local"

        # LangChain 자동 계측 활성화
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        logger.info(
            f"Phoenix tracing enabled ({phoenix_type})",
            endpoint=endpoint,
            project=settings.langsmith_project
        )

        return True

    except Exception as e:
        logger.error(f"Failed to setup Phoenix: {e}", exc_info=True)
        return False


def log_ragas_scores(
    session_id: str,
    agent_name: str,
    scores: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    RAGAS 평가 점수를 Arize AX에 로깅

    Note:
        이 함수는 하위 호환성을 위해 유지되지만,
        실제로는 ArizeEvaluationClient가 자동으로 처리합니다.

        Arize AX에서는 ragas_integration.py의 evaluate_rag_quality()가
        자동으로 Span Attributes에 RAGAS 점수를 추가합니다.

    Args:
        session_id: 세션 ID
        agent_name: 에이전트 이름 (legal_advisor 등)
        scores: RAGAS 점수 딕셔너리 (faithfulness, relevancy 등)
        metadata: 추가 메타데이터

    Deprecated:
        Arize AX 통합 후 이 함수는 더 이상 직접 호출할 필요가 없습니다.
        ArizeEvaluationClient.evaluate_rag()가 자동으로 처리합니다.
    """
    if not PHOENIX_AVAILABLE:
        return

    try:
        # OpenTelemetry를 통해 Arize AX에 RAGAS 점수 전송
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)

        # 현재 활성 Span에 RAGAS 점수를 Attribute로 추가
        current_span = trace.get_current_span()

        if current_span and current_span.is_recording():
            # RAGAS 점수를 Span Attribute로 추가
            for metric_name, metric_value in scores.items():
                current_span.set_attribute(f"ragas.{metric_name}", float(metric_value))

            # 메타데이터 추가
            current_span.set_attribute("ragas.agent_name", agent_name)
            current_span.set_attribute("ragas.session_id", session_id)

            if metadata:
                for key, value in metadata.items():
                    current_span.set_attribute(f"ragas.metadata.{key}", str(value))

            logger.info(
                f"[Arize AX] {agent_name} RAGAS scores sent to span attributes",
                session_id=session_id,
                **scores
            )
        else:
            # Span이 없으면 새로 생성하여 RAGAS 점수 기록
            with tracer.start_as_current_span(
                f"ragas_evaluation_{agent_name}",
                attributes={
                    **{f"ragas.{k}": float(v) for k, v in scores.items()},
                    "ragas.agent_name": agent_name,
                    "ragas.session_id": session_id,
                    **({"ragas.metadata." + k: str(v) for k, v in metadata.items()} if metadata else {})
                }
            ):
                logger.info(
                    f"[Arize AX] {agent_name} RAGAS scores logged (new span)",
                    session_id=session_id,
                    **scores
                )

    except Exception as e:
        logger.warning(f"Failed to log RAGAS scores to Arize AX: {e}")


def create_phoenix_span(
    name: str,
    span_type: str = "chain",
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Phoenix 커스텀 span 생성 (향후 확장용)

    Args:
        name: Span 이름
        span_type: Span 타입 (chain, llm, retriever 등)
        attributes: Span 속성

    Returns:
        Span context manager (Phoenix 설치 시) 또는 None
    """
    if not PHOENIX_AVAILABLE:
        return None

    try:
        # 향후 Phoenix span API 사용
        # from opentelemetry import trace
        # tracer = trace.get_tracer(__name__)
        # return tracer.start_as_current_span(name, attributes=attributes)

        # 현재는 로깅만
        logger.debug(f"[Phoenix Span] {name} ({span_type})", **(attributes or {}))
        return None

    except Exception as e:
        logger.warning(f"Failed to create Phoenix span: {e}")
        return None


# Phoenix 상태 확인 함수
def is_phoenix_enabled() -> bool:
    """Phoenix가 활성화되어 있는지 확인"""
    return PHOENIX_AVAILABLE and settings.phoenix_enabled


def get_phoenix_dashboard_url() -> Optional[str]:
    """Phoenix 대시보드 URL 반환"""
    if not is_phoenix_enabled():
        return None

    endpoint = settings.phoenix_collector_endpoint
    # Phoenix UI는 기본적으로 같은 포트
    return endpoint.replace("/v1/traces", "")  # Collector 엔드포인트 정리

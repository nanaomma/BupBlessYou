"""
RAGAS (RAG Assessment) integration for evaluating Legal Advisor RAG quality
법률 자문 RAG 시스템의 품질 평가를 위한 RAGAS 통합

Arize AX 공식 통합:
- Arize AX Experiments API 사용
- UI에서 Traces → Span Attributes로 RAGAS 점수 확인
"""
import asyncio
from typing import Dict, List, Optional, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Arize AX 클라이언트 (전역 변수)
arize_client = None

# RAGAS 설치 여부 확인
RAGAS_AVAILABLE = False

try:
    from ragas import evaluate
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextRecall
    )
    # Compatibility aliases for older code if any
    faithfulness = Faithfulness
    answer_relevancy = AnswerRelevancy
    context_recall = ContextRecall
    
    RAGAS_AVAILABLE = True
except ImportError:
    logger.debug("RAGAS not installed, RAG evaluation will be disabled")


def setup_arize_evaluation():
    """
    Arize AX 평가 클라이언트 초기화

    앱 시작 시 한 번만 호출 (main.py의 startup 이벤트)
    """
    global arize_client

    from src.config.settings import settings
    from src.utils.arize_client import ArizeEvaluationClient, is_arize_available

    if not is_arize_available():
        logger.warning("Arize AX not available, RAGAS evaluations will be disabled")
        return

    if settings.phoenix_space_id and settings.phoenix_api_key:
        arize_client = ArizeEvaluationClient(
            space_id=settings.phoenix_space_id,
            api_key=settings.phoenix_api_key,
            project_name=settings.langsmith_project
        )
        logger.info("Arize AX evaluation client initialized successfully")
    else:
        logger.warning(
            "Arize AX credentials not found, RAGAS evaluations will be disabled. "
            "Set PHOENIX_SPACE_ID and PHOENIX_API_KEY in .env"
        )


async def evaluate_rag_quality(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, float]:
    """
    RAG 출력 품질을 RAGAS로 평가 (Arize AX 통합)

    Args:
        question: 사용자 질문
        answer: RAG 시스템의 답변
        contexts: 검색된 컨텍스트 리스트
        ground_truth: 정답 (있으면, context_recall 계산용)
        session_id: 세션 ID (로깅용)

    Returns:
        Dict[str, float]: RAGAS 평가 점수
            - faithfulness: 답변이 컨텍스트에 충실한가? (0-1)
            - answer_relevancy: 답변이 질문과 관련있는가? (0-1)
            - context_recall: 필요한 컨텍스트를 모두 검색했는가? (0-1, ground_truth 필요)

    Note:
        Arize AX UI에서 확인:
        1. https://app.arize.com/ 접속
        2. LLM → Traces 메뉴
        3. 프로젝트: bupblessyou 선택
        4. Trace 클릭 → Span Details
        5. Attributes 섹션에서 ragas.* 항목 확인

    Example:
        scores = await evaluate_rag_quality(
            question="형법 347조의 내용은?",
            answer="형법 347조는 사기죄에 대한 규정입니다...",
            contexts=["형법 제347조(사기) ...", "양형기준 ..."],
            session_id="session_123"
        )
    """
    global arize_client

    if not RAGAS_AVAILABLE:
        logger.warning("RAGAS not available, skipping evaluation")
        return {}

    # Arize AX 클라이언트 사용
    if arize_client:
        try:
            scores = await arize_client.evaluate_rag(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
                session_id=session_id,
                metadata={"question_preview": question[:50]}
            )
            return scores
        except Exception as e:
            logger.error(f"Arize AX evaluation failed: {e}", exc_info=True)
            return {}
    else:
        logger.warning(
            "Arize AX client not initialized. "
            "Call setup_arize_evaluation() on app startup."
        )
        return {}


def is_ragas_available() -> bool:
    """RAGAS가 설치되어 있는지 확인"""
    return RAGAS_AVAILABLE


async def evaluate_legal_advisor_output(
    state: Dict[str, Any],
    event_data: Dict[str, Any]
) -> Optional[Dict[str, float]]:
    """
    Legal Advisor 출력을 자동으로 평가하는 헬퍼 함수

    Args:
        state: LangGraph state
        event_data: astream_events의 legal_advisor 완료 이벤트 데이터

    Returns:
        RAGAS 평가 점수 또는 None

    Usage (in main.py):
        if event_name == "legal_advisor" and event_type == "on_chain_end":
            scores = await evaluate_legal_advisor_output(state, event["data"])
    """
    if not RAGAS_AVAILABLE:
        return None

    try:
        # Legal Advisor 출력에서 필요한 정보 추출
        output = event_data.get("output", {})

        # Legal Context에서 정보 추출
        legal_context = output.get("legal_context", {})
        relevant_laws = legal_context.get("relevant_laws", [])
        precedents = legal_context.get("similar_precedents_summary", "")

        # 질문 추출 (새 질문 또는 사건 개요)
        question = state.get("new_question") or state.get("case_summary", "")

        # 컨텍스트 구성
        contexts = relevant_laws + ([precedents] if precedents else [])

        if not contexts or not question:
            logger.debug("Insufficient data for RAGAS evaluation")
            return None

        # 답변 구성 (법률 정보 요약)
        answer = "\n".join(relevant_laws[:3])  # 상위 3개 법령

        # RAGAS 평가 실행
        scores = await evaluate_rag_quality(
            question=question,
            answer=answer,
            contexts=contexts,
            session_id=state.get("session_id")
        )

        return scores

    except Exception as e:
        logger.error(f"Failed to evaluate legal advisor output: {e}")
        return None


# 향후 확장: 배치 평가
async def batch_evaluate_rag(
    evaluations: List[Dict[str, Any]]
) -> List[Dict[str, float]]:
    """
    여러 RAG 출력을 배치로 평가

    Args:
        evaluations: 평가할 데이터 리스트
            [{"question": ..., "answer": ..., "contexts": [...]}, ...]

    Returns:
        각 평가의 RAGAS 점수 리스트
    """
    if not RAGAS_AVAILABLE:
        return []

    results = []
    for eval_data in evaluations:
        scores = await evaluate_rag_quality(**eval_data)
        results.append(scores)

    return results


def evaluate_rag_quality_background(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
    session_id: str | None = None
) -> None:
    """
    RAGAS 평가를 백그라운드 태스크로 실행 (Trace Context 유지)
    """
    import asyncio
    try:
        from opentelemetry import context as otel_context
    except ImportError:
        otel_context = None
    
    if not RAGAS_AVAILABLE:
        return
    
    # 1. 현재 실행 컨텍스트(Trace ID 등) 캡처
    current_ctx = otel_context.get_current() if otel_context else None

    async def _run_evaluation():
        """내부 비동기 평가 실행 함수"""
        token = None
        # 2. 백그라운드 태스크 내에서 컨텍스트 복원 (Trace 연결)
        if current_ctx and otel_context:
            token = otel_context.attach(current_ctx)
        
        try:
            await evaluate_rag_quality(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
                session_id=session_id
            )
        except Exception as e:
            logger.error(
                "Background RAGAS evaluation failed",
                exc_info=True,
                session_id=session_id,
                error_type=type(e).__name__
            )
        finally:
            # 3. 컨텍스트 정리
            if token and otel_context:
                otel_context.detach(token)
    
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run_evaluation())
    except RuntimeError:
        logger.debug("No running event loop, creating new one for RAGAS evaluation")
        asyncio.run(_run_evaluation())

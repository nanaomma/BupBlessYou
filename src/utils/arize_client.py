"""
Arize AX Evaluation Client for RAGAS Integration
Arize AX의 Experiments API를 사용한 RAGAS 평가 클라이언트
"""
from typing import Dict, List, Optional, Any
import asyncio
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

try:
    from opentelemetry import trace
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.debug("OpenTelemetry not installed, Arize AX evaluations will be disabled")


class ArizeEvaluationClient:
    """
    Arize AX Experiments API를 사용한 RAGAS 평가 클라이언트

    Arize AX의 공식 평가 워크플로우를 사용하여 RAGAS 점수를
    Experiments 탭에서 확인할 수 있도록 합니다.

    Note:
        현재는 OpenTelemetry Span Attributes로 RAGAS 점수를 전송합니다.
        Arize AX는 이러한 attributes를 자동으로 수집하여 UI에 표시합니다.
    """

    def __init__(self, space_id: str, api_key: str, project_name: str):
        """
        Args:
            space_id: Arize AX Space ID
            api_key: Arize AX API Key
            project_name: 프로젝트 이름 (예: "bupblessyou")
        """
        self.space_id = space_id
        self.api_key = api_key
        self.project_name = project_name
        self.tracer = None

        if OTEL_AVAILABLE:
            try:
                # Tracer 가져오기 (이미 phoenix_integration.py에서 Provider가 설정됨)
                self.tracer = trace.get_tracer(__name__)
                logger.info(
                    "ArizeEvaluationClient initialized",
                    project_name=project_name
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Arize tracer: {e}")

    async def evaluate_rag(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        RAG 출력을 RAGAS로 평가하고 Arize AX에 전송 (Manual ascore 사용)
        """
        if not OTEL_AVAILABLE or not self.tracer:
            logger.warning("OpenTelemetry tracer not available, skipping evaluation")
            return {}

        try:
            # Ragas v0.4 imports
            from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
            from ragas.llms import llm_factory
            from ragas.embeddings import embedding_factory
            from openai import AsyncOpenAI
            import os

            # 환경변수 설정
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key

            # OpenAI 클라이언트 생성
            openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            
            # Factory로 LLM/Embeddings 생성
            ragas_llm = llm_factory(model="gpt-4o-mini", client=openai_client)
            ragas_embeddings = embedding_factory(model="text-embedding-3-small", client=openai_client)

            scores = {}

            # 1. Faithfulness
            try:
                f_metric = Faithfulness(llm=ragas_llm)
                result = await f_metric.ascore(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts
                )
                scores["faithfulness"] = result.value
            except Exception as e:
                logger.warning(f"Faithfulness evaluation failed: {e}")

            # 2. Answer Relevancy
            try:
                ar_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
                result = await ar_metric.ascore(
                    user_input=question,
                    response=answer
                )
                scores["answer_relevancy"] = result.value
            except Exception as e:
                logger.warning(f"Answer Relevancy evaluation failed: {e}")

            # 3. Context Precision & Recall (if ground_truth exists)
            if ground_truth:
                try:
                    cp_metric = ContextPrecision(llm=ragas_llm)
                    result = await cp_metric.ascore(
                        user_input=question,
                        retrieved_contexts=contexts,
                        reference=ground_truth
                    )
                    scores["context_precision"] = result.value
                except Exception as e:
                    logger.warning(f"Context Precision evaluation failed: {e}")

                try:
                    cr_metric = ContextRecall(llm=ragas_llm)
                    result = await cr_metric.ascore(
                        user_input=question,
                        retrieved_contexts=contexts,
                        reference=ground_truth
                    )
                    scores["context_recall"] = result.value
                except Exception as e:
                    logger.warning(f"Context Recall evaluation failed: {e}")

            # Arize AX로 전송 (OpenTelemetry Span Attributes 사용)
            self._send_to_arize(
                scores=scores,
                question=question,
                answer=answer,
                session_id=session_id,
                metadata=metadata or {}
            )

            logger.info(
                "[Arize AX] RAGAS evaluation completed",
                session_id=session_id,
                **scores
            )

            return scores

        except Exception as e:
            logger.error(f"Arize RAGAS evaluation failed: {e}", exc_info=True)
            return {}

    def _send_to_arize(
        self,
        scores: Dict[str, float],
        question: str,
        answer: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        RAGAS 점수와 입출력을 Arize AX로 전송 (Span Attributes 사용)
        """
        try:
            from opentelemetry.trace import Status, StatusCode, SpanKind
            
            # 현재 활성 Span 가져오기 (ragas_integration에서 context.attach()로 전달됨)
            current_span = trace.get_current_span()

            # 속성 준비
            attributes = {
                "input.value": question,
                "output.value": answer,
                "ragas.agent_name": "legal_advisor",
                "ragas.session_id": session_id or "unknown",
            }
            
            # 점수 추가
            for k, v in scores.items():
                attributes[f"ragas.{k}"] = float(v)
                
            # 메타데이터 추가
            if metadata:
                for k, v in metadata.items():
                    attributes[f"ragas.metadata.{k}"] = str(v)

            if current_span and current_span.is_recording():
                # 부모 Span이 있으면 속성만 추가 (이미 존재하는 Trace에 병합)
                current_span.set_attributes(attributes)
                current_span.set_status(Status(StatusCode.OK))
                
                logger.info(
                    "[Arize AX] RAGAS scores attached to existing span",
                    session_id=session_id,
                    trace_id=f"{current_span.get_span_context().trace_id:x}",
                    **scores
                )
            else:
                # 부모 Span이 없거나 Recording이 아니면 새로 생성 (Internal Span)
                # 이 경우 Root Span의 자식으로 붙으려면 Context가 살아있어야 함
                with self.tracer.start_as_current_span(
                    "ragas_evaluation",
                    kind=SpanKind.INTERNAL,
                    attributes=attributes
                ) as span:
                    span.set_status(Status(StatusCode.OK))
                    logger.info(
                        "[Arize AX] RAGAS scores logged (new span)",
                        session_id=session_id,
                        trace_id=f"{span.get_span_context().trace_id:x}",
                        **scores
                    )

        except Exception as e:
            logger.warning(f"Failed to send RAGAS scores to Arize AX: {e}")


def is_arize_available() -> bool:
    """Arize AX 클라이언트 사용 가능 여부 확인"""
    return OTEL_AVAILABLE

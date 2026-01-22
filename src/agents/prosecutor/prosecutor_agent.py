"Prosecutor Agent - 2025 LangChain Pattern"
import json
from typing import Dict, Any, Optional, Literal
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config.settings import settings
from src.rag.vector_store import VectorStore
from src.agents.common.state import CourtSimulationState, Role
from src.prompts.prosecutor_prompts import PROSECUTOR_SYSTEM_PROMPT, PROSECUTOR_HUMAN_TEMPLATE
from src.utils.llm_factory import create_llm
from src.utils.json_parser import parse_agent_json_with_retry, create_fallback_agent_output

from src.utils.logger import get_logger
from src.utils.langsmith_integration import create_agent_tracer
from src.utils.facts_guardrail import FactsGuardrail
from src.utils.history_compressor import HistoryCompressor

logger = get_logger(__name__)

class ProsecutorAgent:
    """
    검사 에이전트 - 2025 LangChain 패턴 (CourtSimulationState 기반)

    LLM 선택:
    - provider: "openai" 또는 "upstage" (기본값: settings.default_llm_provider)
    - temperature: 0.7 (창의적이고 설득력 있는 주장을 위해)
    """

    def __init__(self, provider: Optional[Literal["openai", "upstage"]] = None):
        """
        ProsecutorAgent 초기화

        Args:
            provider: LLM provider 선택 ("openai" 또는 "upstage")
                     None이면 settings.default_llm_provider 사용
        """
        self.llm = create_llm(provider=provider, temperature=0.7)
        self.provider = provider or settings.default_llm_provider
        self.vector_store = VectorStore()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PROSECUTOR_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", PROSECUTOR_HUMAN_TEMPLATE),
            ]
        )

        # LCEL 체인 구성
        # LLM이 직접 JSON 스키마를 따르도록 유도 (with_structured_output 대신 프롬프트 지시)
        self.chain = self.prompt | self.llm

        logger.info(f"ProsecutorAgent initialized with {self.provider.upper()} LLM")

    async def generate_argument(
        self,
        state: CourtSimulationState
    ) -> Dict[str, Any]:
        """
        검사 주장 생성

        Args:
            state: 현재 법정 시뮬레이션 상태

        Returns:
            업데이트된 State 딕셔너리 (messages 필드)
        """
        # 1. State에서 필요한 정보 추출
        conversation_history = state.get("messages", [])

        # 2. Facts Guardrail: 확인된 사실만 추출
        verified_facts = FactsGuardrail.extract_verified_facts(state)
        facts_guard_prompt = FactsGuardrail.create_facts_guard_prompt(verified_facts)

        # 3. History Compression: 대화 맥락 압축 및 요약
        compressed_history = HistoryCompressor.compress_history(
            messages=conversation_history,
            max_recent_messages=4,  # 최근 2 라운드 (검사 2 + 변호사 2)
            include_initial_brief=True,
            compress_middle=True
        )

        context_summary = HistoryCompressor.create_context_summary(
            messages=conversation_history,
            current_role="prosecutor"
        )

        # 4. LangSmith 트레이싱 설정
        tracer = create_agent_tracer("prosecutor", state)
        langsmith_config = {
            "metadata": tracer.get_metadata(
                case_id=state.get("case_id"),
                legal_context_available=bool(verified_facts.get("legal_basis")),
                # ✅ NEW: Input 구조 메타데이터 추가 (Phoenix 분석 용이성)
                input_structure={
                    "case_summary_length": len(verified_facts.get("case_summary", "")),
                    "case_attributes_count": len(verified_facts.get("verified_attributes", {})),
                    "history_message_count": len(conversation_history),
                    "compressed_history_count": len(compressed_history),
                    "legal_context_available": bool(verified_facts.get("legal_basis")),
                    "facts_guardrail_enabled": True,
                    "history_compression_enabled": True
                }
            ),
            "tags": tracer.get_tags(action="generate_argument")
        }

        # 5. 체인 실행 with 재시도 로직
        chain_input = {
            "facts_guard_prompt": facts_guard_prompt,
            "context_summary": context_summary,
            "history": compressed_history
        }

        required_fields = ["role", "content", "emotion"]

        try:
            agent_output_dict = await parse_agent_json_with_retry(
                llm_chain=self.chain,
                chain_input=chain_input,
                required_fields=required_fields,
                max_retries=2,
                config=langsmith_config
            )
        except ValueError as e:
            # 모든 재시도 실패 - 폴백
            logger.error(f"Prosecutor JSON parsing failed after retries: {e}")

            # Phoenix에 품질 저하 기록
            try:
                from src.utils.phoenix_integration import is_phoenix_enabled
                if is_phoenix_enabled():
                    from opentelemetry import trace
                    current_span = trace.get_current_span()
                    if current_span and current_span.is_recording():
                        current_span.set_attribute("agent.quality_degraded", True)
                        current_span.set_attribute("agent.fallback_reason", "json_parse_error_all_retries")
                        current_span.set_attribute("agent.error_type", type(e).__name__)
            except Exception:
                pass

            agent_output_dict = create_fallback_agent_output(
                role=Role.PROSECUTOR,
                error_message="검사의 주장을 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            )

        # Graph State에 업데이트할 messages 리턴
        return {
            "messages": [AIMessage(content=json.dumps(agent_output_dict, ensure_ascii=False))]
        }


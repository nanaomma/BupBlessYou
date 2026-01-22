"""Judgment Analyzer - 판결 비교 및 분석"""
from typing import Dict, Any, Optional, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
import json

from src.config.settings import settings
from src.agents.common.state import CourtSimulationState, AgentOutput, Role
from src.prompts.judgment_prompts import (
    JUDGMENT_ANALYZER_SYSTEM_PROMPT,
    JUDGMENT_ANALYZER_HUMAN_TEMPLATE
)
from src.utils.llm_factory import create_llm

from src.utils.logger import get_logger
from src.utils.langsmith_integration import create_agent_tracer
from src.utils.json_parser import parse_agent_json_with_retry, create_fallback_agent_output

logger = get_logger(__name__)


class JudgmentAnalyzer:
    """
    판결 비교 및 피드백 생성

    담당: 판사 에이전트 팀 (2인)
    역할:
    - 유저 판결과 실제 판결 비교
    - 차이점 분석
    - 피드백 생성
    - 고려하지 못한 요소 식별

    LLM 선택:
    - provider: "openai" 또는 "upstage" (기본값: settings.default_llm_provider)
    - temperature: 0.5 (판결 분석의 일관성을 위해 낮은 온도 사용)
    """

    def __init__(self, provider: Optional[Literal["openai", "upstage"]] = None):
        """
        JudgmentAnalyzer 초기화

        Args:
            provider: LLM provider 선택 ("openai" 또는 "upstage")
                     None이면 settings.default_llm_provider 사용
        """
        # LLM 생성 (OpenAI 또는 Upstage)
        self.llm = create_llm(provider=provider, temperature=0.5)
        self.provider = provider or settings.default_llm_provider

        # 프롬프트 템플릿 구성
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", JUDGMENT_ANALYZER_SYSTEM_PROMPT),
            ("human", JUDGMENT_ANALYZER_HUMAN_TEMPLATE)
        ])

        # LCEL 체인 구성
        self.chain = self.prompt | self.llm

        logger.info(f"JudgmentAnalyzer initialized with {self.provider.upper()} LLM")

    async def analyze_and_feedback(
        self, state: CourtSimulationState
    ) -> Dict[str, Any]:
        """
        유저 판결과 실제 판결을 비교 분석하고 피드백을 생성합니다.

        Args:
            state: 현재 법정 시뮬레이션 상태 (user_verdict, actual_judgment 포함)

        Returns:
            업데이트될 State의 딕셔너리 (analysis_result 필드 및 messages 필드)
        """
        # --- 1. State에서 필요한 정보 추출 ---
        user_verdict = state.get("user_verdict", "판결 없음")
        user_sentence_text = state.get("user_sentence_text", "")
        user_reasoning = state.get("user_reasoning", "이유 없음")

        # Combine user verdict and sentence for analysis
        if user_sentence_text:
            user_verdict_full = f"{user_verdict} - {user_sentence_text}"
        else:
            user_verdict_full = user_verdict

        actual_judgment = state.get("actual_judgment", {})
        
        # Extract actual verdict and rule text, handling possible object structures
        actual_verdict = actual_judgment.get("actual_label", "실제 판결 정보 없음")
        if isinstance(actual_verdict, dict) and "text" in actual_verdict:
            actual_verdict = actual_verdict["text"]
            
        actual_rule = actual_judgment.get("actual_rule", "")
        if isinstance(actual_rule, dict) and "text" in actual_rule:
            actual_rule = actual_rule["text"]
            
        actual_reasoning = actual_judgment.get("actual_reason", "실제 판결 이유 정보 없음")

        # Combine actual verdict and rule for analysis
        if actual_rule:
            actual_verdict_full = f"{actual_verdict} ({actual_rule})"
        else:
            actual_verdict_full = actual_verdict

        case_summary = state.get("case_summary", "사건 개요 없음")

        # --- 2. LangSmith 트레이싱 설정 ---
        tracer = create_agent_tracer("judgment_analyzer", state)
        langsmith_config = {
            "metadata": tracer.get_metadata(
                user_verdict=user_verdict,
                reasoning_length=len(user_reasoning)
            ),
            "tags": tracer.get_tags(action="analyze_judgment")
        }

        # --- 3. LLM을 사용하여 분석 및 피드백 생성 with 재시도 로직 ---
        chain_input = {
            "case_summary": case_summary,
            "user_verdict": user_verdict_full,
            "user_reasoning": user_reasoning,
            "actual_verdict": actual_verdict_full,
            "actual_reasoning": actual_reasoning,
        }

        required_fields = ["comparison_summary", "user_strength", "user_weakness", "overlooked_factors", "learning_points"]

        try:
            analysis_result_dict = await parse_agent_json_with_retry(
                llm_chain=self.chain,
                chain_input=chain_input,
                required_fields=required_fields,
                max_retries=2,
                config=langsmith_config
            )
        except ValueError as e:
            # 모든 재시도 실패 - 폴백
            logger.error(f"Judgment Analyzer JSON parsing failed after retries: {e}")

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

            analysis_result_dict = {
                "comparison_summary": "판결 분석을 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "user_strength": "",
                "user_weakness": "",
                "overlooked_factors": [],
                "learning_points": []
            }

        # --- 4. 분석 결과와 함께 사용자에게 보여줄 메시지 생성 (AgentOutput 형식) ---
        feedback_content = f"### 판결 분석 결과\n\n" \
                           f"**요약**: {analysis_result_dict.get('comparison_summary')}\n\n" \
                           f"**유저님의 강점**: {analysis_result_dict.get('user_strength')}\n\n" \
                           f"**아쉬운 점**: {analysis_result_dict.get('user_weakness')}\n\n" \
                           f"**간과된 요소**: {', '.join(analysis_result_dict.get('overlooked_factors', []))}\n\n" \
                           f"**학습 포인트**: {analysis_result_dict.get('learning_points')}"
        
        agent_output_message: AgentOutput = {
            "role": Role.SYSTEM, # 시스템 메시지로 피드백 제공
            "content": feedback_content,
            "emotion": "neutral", # 분석 결과는 감정 없음
            "references": []
        }

        # --- 5. State 업데이트용 딕셔너리 반환 ---
        return {
            "analysis_result": analysis_result_dict,
            "messages": [AIMessage(content=json.dumps(agent_output_message, ensure_ascii=False))]
        }

    # 이전의 compare_judgments, generate_feedback, analyze_overlooked_factors 메서드는
    # LLM 기반 analyze_and_feedback 메서드로 통합되므로 제거됩니다.
    # 각 팀에서 필요한 경우 내부적으로 세부 로직을 구현할 수 있습니다.
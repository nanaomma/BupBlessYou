"""Legal Advisor Agent - 법률자문 AI (관련 법령/판례 제공)"""
import json
from typing import Dict, Any, List, Optional, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage

from src.config.settings import settings
from src.rag.vector_store import VectorStore
from src.agents.common.state import CourtSimulationState, LegalContext, CaseAttribute, Role
from src.prompts.legal_advisor_prompts import LEGAL_ADVISOR_SYSTEM_PROMPT, LEGAL_ADVISOR_HUMAN_TEMPLATE
from src.utils.llm_factory import create_llm
from src.utils.langsmith_integration import create_agent_tracer
from src.utils.logger import get_logger
from src.agents.legal_advisor.legal_advisor_retriever import LegalAdvisorRetriever
from src.agents.legal_advisor.legal_context_formatter import LegalContextFormatter
import asyncio

logger = get_logger(__name__)


# NOTE: 이전 BaseAgent와 AgentMessage는 더 이상 사용하지 않습니다.
# AgentOutput은 Prosecutor/Defense Agent처럼 대화를 생성하는 경우에만 사용합니다.
# Legal Advisor는 messages 대신 state['legal_context']를 업데이트합니다.

class LegalAdvisorAgent:
    """
    법률자문 에이전트 - 관련 법령과 판례를 제공하여 `legal_context`를 업데이트하는 역할

    담당: 법률자문 에이전트 팀 (2인)
    역할:
    - `CourtSimulationState`에서 `case_summary`, `case_attributes`, `messages`를 참조하여 현재 논의에 필요한 법률 정보 판단
    - 법제처 API 연동 (법령 정보)
    - 양형기준표 정보 제공 (PDF 파싱 결과 활용)
    - RAG 시스템 연동 (판례 검색)
    - 수집된 정보를 바탕으로 `LegalContext`를 구성하여 State 업데이트

    LLM 선택:
    - provider: "openai" 또는 "upstage" (기본값: settings.default_llm_provider)
    - temperature: 0.3 (정확한 법률 정보 제공을 위해 낮은 온도 사용)
    """

    def __init__(self, provider: Optional[Literal["openai", "upstage"]] = None):
        """
        LegalAdvisorAgent 초기화

        Args:
            provider: LLM provider 선택 ("openai" 또는 "upstage")
                     None이면 settings.default_llm_provider 사용
        """
        self.llm = create_llm(provider=provider, temperature=0.3)
        self.provider = provider or settings.default_llm_provider
        self.vector_store = VectorStore()  # RAG 시스템 (TODO: RAG 팀 구현)
        self.retriever = LegalAdvisorRetriever()  # 법령 및 양형기준 검색용
        self.formatter = LegalContextFormatter(provider=provider)  # RAG 검색 결과 정제용

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", LEGAL_ADVISOR_SYSTEM_PROMPT),
                ("human", LEGAL_ADVISOR_HUMAN_TEMPLATE),
            ]
        )
        self.chain = self.prompt | self.llm

        logger.info(f"LegalAdvisorAgent initialized with {self.provider.upper()} LLM")

    async def update_legal_context(
        self,
        state: CourtSimulationState,
        session_id: Optional[str] = None
    ) -> Dict[str, LegalContext]:
        """
        현재 상태를 기반으로 법률 자문 컨텍스트를 업데이트합니다.
        `messages`를 생성하지 않고, `legal_context` 필드를 반환합니다.

        Args:
            state: 현재 법정 시뮬레이션 상태
            session_id: 세션 ID (RAGAS 평가용, 선택적)

        Returns:
            업데이트될 LegalContext 딕셔너리
        """
        # --- 1. State에서 필요한 정보 추출 ---
        case_summary = state.get("case_summary", "")
        case_attributes = state.get("case_attributes", [])
        case_id = state.get("case_id")
        # 최근 대화 내용을 요약하여 자문 LLM에 전달
        conversation_summary = self._summarize_conversation_history(state.get("messages", []))
        
        # --- 1-1. sentencing_factors 조회 ---
        sentencing_factors = self.get_sentencing_factors(case_id)
        
        # --- 2. RAG 및 외부 API 호출 ---
        rag_search_results = await self._search_legal_information(
            case_summary, case_attributes, conversation_summary, sentencing_factors,
            session_id=session_id
        )

        # --- 2-1. 검색 결과를 LegalContext 형식으로 변환 (raw 데이터) ---
        raw_legal_context_dict = self._convert_retrieval_to_legal_context(rag_search_results)

        # --- 2-2. ✅ NEW: Formatter를 통해 raw 데이터 정제 ---
        crime_type = rag_search_results.get("crime_type", "")
        legal_context_dict = await self.formatter.format_full_legal_context(
            raw_legal_context=raw_legal_context_dict,
            crime_type=crime_type,
            state=state
        )
        
        # --- 3. 프롬프트 주입을 위한 포맷팅 ---
        case_attributes_formatted = self._format_case_attributes(case_attributes)
        
        # RAG 검색 결과를 문자열로 변환하여 프롬프트에 포함 (LLM이 참고용으로 사용)
        relevant_info_from_rag = json.dumps(rag_search_results, ensure_ascii=False, indent=2)
        
        # --- 4. LLM 체인 실행 (with LangSmith tracing) ---
        # Create LangSmith tracer for this agent call
        tracer = create_agent_tracer("legal_advisor", state)

        # Prepare LangSmith metadata and tags
        langsmith_config = {
            "metadata": tracer.get_metadata(
                case_id=state.get("case_id"),
                has_attributes=len(case_attributes) > 0
            ),
            "tags": tracer.get_tags(action="update_legal_context")
        }

        llm_response_message = await self.chain.ainvoke(
            {
                "case_summary": case_summary,
                "case_attributes_formatted": case_attributes_formatted,
                "conversation_summary": conversation_summary,
                "relevant_info_from_rag": relevant_info_from_rag # RAG 결과를 프롬프트에 주입
            },
            config=langsmith_config  # LangSmith tracing configuration
        )
        
        # --- 5. LLM 응답 파싱 및 LegalContext 병합 (검색 결과 우선) ---
        try:
            llm_context_dict = json.loads(llm_response_message.content)
            # LLM이 생성한 similar_precedents_summary를 검색 결과에 병합
            if llm_context_dict.get("similar_precedents_summary"):
                legal_context_dict["similar_precedents_summary"] = llm_context_dict["similar_precedents_summary"]
        except json.JSONDecodeError:
            # JSON 파싱 실패 시, 검색 결과만 사용
            logger.warning("LLM 응답 JSON 파싱 실패, 검색 결과만 사용")
            if not legal_context_dict.get("similar_precedents_summary"):
                legal_context_dict["similar_precedents_summary"] = "법률 자문 정보를 가져오지 못했습니다."

        # --- 6. State 업데이트용 딕셔너리 반환 (legal_context와 sentencing_factors 필드 업데이트) ---
        result = {"legal_context": legal_context_dict}
        if sentencing_factors is not None:
            result["sentencing_factors"] = sentencing_factors
        return result

    def get_sentencing_factors(self, case_id: Optional[int]) -> Optional[Dict[str, Any]]:
        """
        DB에서 sentencing_factors를 조회하여 반환합니다.
        
        Args:
            case_id: 사건 ID (None이면 None 반환)
            
        Returns:
            sentencing_factors 딕셔너리 또는 None
        """
        if case_id is None:
            return None
        
        try:
            case_info = self.retriever.get_case_info(case_id)
            if case_info.get("error"):
                logger.warning(f"Case {case_id} not found: {case_info.get('error')}")
                return None
            return case_info.get("sentencing_factors")
        except Exception as e:
            logger.error(f"Failed to retrieve sentencing_factors for case {case_id}: {e}")
            return None

    async def _search_legal_information(
        self,
        case_summary: str,
        case_attributes: List[CaseAttribute],
        conversation_summary: str,
        sentencing_factors: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        RAG: 법률 정보 검색 (LegalAdvisorRetriever 사용)
        
        Args:
            case_summary: 사건 개요
            case_attributes: 사건 속성 리스트
            conversation_summary: 대화 요약
            sentencing_factors: 양형 고려 요소 (선택적)
            
        Returns:
            검색 결과 딕셔너리: {"laws": {...}, "guidelines": {...}}
        """
        # crime_type 추출 (case_summary와 case_attributes에서)
        crime_type = self._extract_crime_type(case_summary, case_attributes)
        
        # description 구성 (case_summary + case_attributes + sentencing_factors)
        description_parts = [case_summary] if case_summary else []
        if case_attributes:
            attr_descriptions = [attr.get("description", "") for attr in case_attributes]
            description_parts.extend(attr_descriptions)
        
        # sentencing_factors를 텍스트로 변환하여 추가
        if sentencing_factors:
            factors_text = self._format_sentencing_factors_to_text(sentencing_factors)
            if factors_text:
                description_parts.append(f"양형 요소: {factors_text}")
        
        description = " ".join(description_parts)
        
        # ✅ NEW: Phoenix 추적을 위한 OpenTelemetry Tracer 가져오기
        try:
            from src.utils.phoenix_integration import is_phoenix_enabled
            from opentelemetry import trace
            phoenix_enabled = is_phoenix_enabled()
            tracer = trace.get_tracer(__name__) if phoenix_enabled else None
        except Exception:
            phoenix_enabled = False
            tracer = None
        
        # 비동기 작업을 스레드 풀에서 실행 (blocking 호출 래핑)
        def _search():
            # ✅ NEW: 법령 검색 Span 추가
            if phoenix_enabled and tracer:
                with tracer.start_as_current_span("rag.search_legal_provisions") as span:
                    span.set_attribute("rag.crime_type", crime_type)
                    span.set_attribute("rag.description_length", len(description[:500]))
                    span.set_attribute("rag.top_k", 5)
                    
                    laws_result = self.retriever.search_legal_provisions(
                        crime_type=crime_type,
                        description=description[:500],  # 길이 제한
                        top_k=5
                    )
                    
                    # 검색 결과 메타데이터 기록
                    if laws_result.get("status") == "success":
                        span.set_attribute("rag.result_count", len(laws_result.get("laws", [])))
                        span.set_attribute("rag.citation_count", len(laws_result.get("citations", [])))
                    else:
                        span.set_attribute("rag.status", "fail")
                        span.set_attribute("rag.error", laws_result.get("error", "unknown"))
            else:
                # Phoenix 비활성화 시 기존 로직
                laws_result = self.retriever.search_legal_provisions(
                    crime_type=crime_type,
                    description=description[:500],
                    top_k=5
                )

            # ✅ NEW: 양형기준 검색 Span 추가
            if phoenix_enabled and tracer:
                with tracer.start_as_current_span("rag.search_sentencing_guidelines") as span:
                    span.set_attribute("rag.crime_type", crime_type)
                    span.set_attribute("rag.description_length", len(description[:500]))
                    span.set_attribute("rag.top_k", 3)
                    span.set_attribute("rag.sentencing_factors_provided", bool(sentencing_factors))
                    
                    guidelines_result = self.retriever.get_sentencing_guidelines(
                        crime_type=crime_type,
                        description=description[:500],
                        top_k=3,
                        sentencing_factors=sentencing_factors
                    )
                    
                    # 검색 결과 메타데이터 기록
                    if guidelines_result.get("status") == "success":
                        span.set_attribute("rag.status", "success")
                        # guidelines 리스트에서 첫 번째 항목 추출
                        guidelines = guidelines_result.get("guidelines", [])
                        if guidelines:
                            guideline = guidelines[0]
                            span.set_attribute("rag.guideline_name", guideline.get("guideline_name", ""))
                            span.set_attribute("rag.base_range_exists", bool(guideline.get("base_range")))
                    else:
                        span.set_attribute("rag.status", "fail")
                        span.set_attribute("rag.error", guidelines_result.get("error", "unknown"))
            else:
                # Phoenix 비활성화 시 기존 로직
                guidelines_result = self.retriever.get_sentencing_guidelines(
                    crime_type=crime_type,
                    description=description[:500],
                    top_k=3,
                    sentencing_factors=sentencing_factors
                )

            # ✅ RAGAS 평가 실행 (백그라운드)
            if session_id and settings.ragas_enabled:
                try:
                    from src.utils.ragas_adapters import (
                        prepare_sentencing_guideline_evaluation,
                        prepare_legal_provision_evaluation
                    )
                    from src.utils.ragas_integration import evaluate_rag_quality_background

                    # 1. 양형기준 검색 성공 시 RAGAS 평가
                    if guidelines_result.get("status") == "success":
                        eval_data = prepare_sentencing_guideline_evaluation(
                            crime_type=crime_type,
                            description=description[:500],
                            guideline_result=guidelines_result
                        )

                        if eval_data:
                            evaluate_rag_quality_background(
                                **eval_data,
                                session_id=session_id
                            )
                            logger.debug(
                                "RAGAS evaluation triggered for sentencing guidelines",
                                session_id=session_id,
                                crime_type=crime_type
                            )
                            
                    # 2. 법령 검색 성공 시 RAGAS 평가
                    if laws_result.get("status") == "success":
                        eval_data_laws = prepare_legal_provision_evaluation(
                            crime_type=crime_type,
                            description=description[:500],
                            laws_result=laws_result
                        )
                        
                        if eval_data_laws:
                            evaluate_rag_quality_background(
                                **eval_data_laws,
                                session_id=session_id
                            )
                            logger.debug(
                                "RAGAS evaluation triggered for legal provisions",
                                session_id=session_id,
                                crime_type=crime_type
                            )

                except Exception as e:
                    # RAGAS 평가 실패해도 메인 로직에 영향 없음
                    logger.warning(f"RAGAS evaluation setup failed: {e}")

            return {
                "laws": laws_result,
                "guidelines": guidelines_result,
                "crime_type": crime_type
            }
        
        try:
            result = await asyncio.to_thread(_search)
            return result
        except Exception as e:
            logger.error(f"RAG 검색 중 오류 발생: {e}")
            return {
                "laws": {"status": "fail", "laws": [], "citations": [], "error": str(e)},
                "guidelines": {"status": "fail", "guideline": None, "error": str(e)},
                "crime_type": crime_type
            }
    
    def _extract_crime_type(self, case_summary: str, case_attributes: List[CaseAttribute]) -> str:
        """
        case_summary와 case_attributes에서 범죄 유형 추출
        
        Args:
            case_summary: 사건 개요
            case_attributes: 사건 속성 리스트
            
        Returns:
            범죄 유형 문자열 (예: "사기", "폭행", "절도")
        """
        # 형법 범죄 유형 키워드 목록
        crime_keywords = [
            "사기", "폭행", "상해", "협박", "감금", "절도", "강도", "강간", "강제추행",
            "살인", "방화", "공갈", "횡령", "배임", "명예훼손", "모욕", "도박", "유기",
            "교통사고", "음주운전", "마약", "조세범"
        ]
        
        # case_summary에서 범죄 유형 찾기
        combined_text = case_summary.lower() if case_summary else ""
        for keyword in crime_keywords:
            if keyword in combined_text:
                return keyword
        
        # case_attributes에서 찾기
        for attr in case_attributes:
            attr_desc = attr.get("description", "").lower()
            for keyword in crime_keywords:
                if keyword in attr_desc:
                    return keyword
        
        # 찾지 못한 경우 "UNKNOWN" 반환
        return "UNKNOWN"
    
    def _summarize_conversation_history(self, messages: List[BaseMessage]) -> str:
        """
        대화 히스토리를 요약하여 자문 LLM에 전달합니다.
        실제로는 더 정교한 요약 로직이 필요할 수 있습니다.
        """
        summaries = []
        for msg in messages[-5:]: # 최근 5개 메시지만 요약
            if isinstance(msg.content, str):
                try:
                    agent_output = json.loads(msg.content)
                    if 'role' in agent_output and 'content' in agent_output:
                        summaries.append(f"{agent_output['role']}: {agent_output['content'][:50]}...") # 처음 50자만
                except json.JSONDecodeError:
                    summaries.append(f"{msg.type}: {msg.content[:50]}...")
            else: # content가 string이 아닌 경우 (e.g., list of dicts)
                summaries.append(f"{msg.type}: (복합 메시지)...")
        return "\n".join(summaries) if summaries else "이전 대화 없음."

    def _format_case_attributes(self, attributes: List[CaseAttribute]) -> str:
        """사건 속성 포맷팅"""
        if not attributes: return "없음"
        return "\n".join([f"- {attr['description']}" for attr in attributes])
    
    def _format_sentencing_factors_to_text(self, factors: Optional[Dict[str, Any]]) -> str:
        """
        sentencing_factors 딕셔너리를 자연어 텍스트로 변환합니다.
        
        Args:
            factors: sentencing_factors 딕셔너리 (None 가능)
            
        Returns:
            자연어로 변환된 텍스트 문자열
        """
        if not factors:
            return ""
        
        parts = []
        
        # 전력 관련
        if "전력_관련" in factors:
            전력 = factors["전력_관련"]
            if 전력.get("동종_전과_횟수") is not None:
                횟수 = 전력["동종_전과_횟수"]
                if 횟수 > 0:
                    parts.append(f"동종 전과 {횟수}회")
                else:
                    parts.append("전과 없음")
            if 전력.get("집행유예_이상_전과_존재") is True:
                parts.append("집행유예 이상 전과 존재")
            elif 전력.get("집행유예_이상_전과_존재") is False:
                parts.append("집행유예 이상 전과 없음")
        
        # 범행 위험성 관련
        if "범행_위험성_관련" in factors:
            위험성 = factors["범행_위험성_관련"]
            if 위험성.get("상해_정도"):
                parts.append(f"상해 정도: {위험성['상해_정도']}")
            if 위험성.get("전치_기간"):
                parts.append(f"전치 기간: {위험성['전치_기간']}")
            if 위험성.get("피해_규모") is not None:
                parts.append(f"피해 규모: {위험성['피해_규모']}")
            if 위험성.get("피해자_관계"):
                parts.append(f"피해자 관계: {위험성['피해자_관계']}")
        
        # 책임 인정 관련
        if "책임_인정_관련" in factors:
            책임 = factors["책임_인정_관련"]
            if 책임.get("범행_인정") is True:
                parts.append("범행 인정")
            elif 책임.get("범행_인정") is False:
                parts.append("범행 부인")
            if 책임.get("반성_여부") is True:
                parts.append("반성함")
            elif 책임.get("반성_여부") is False:
                parts.append("반성 안 함")
        
        # 기타 참작사유
        if "기타_참작사유" in factors:
            기타 = factors["기타_참작사유"]
            if 기타.get("피해_회복") is not None:
                parts.append(f"피해 회복: {기타['피해_회복']}")
            if 기타.get("합의") is True:
                parts.append("합의 완료")
            elif 기타.get("합의") is False:
                parts.append("합의 안 됨")
            if 기타.get("자수") is True:
                parts.append("자수함")
            if 기타.get("생계_목적") is True:
                parts.append("생계 목적")
            if 기타.get("조직적_범행") is True:
                parts.append("조직적 범행")
        
        return ", ".join(parts) if parts else ""
    
    def _convert_retrieval_to_legal_context(self, rag_results: Dict[str, Any]) -> LegalContext:
        """
        검색 결과를 LegalContext 형식으로 변환
        
        Args:
            rag_results: _search_legal_information()의 반환값
            
        Returns:
            LegalContext 딕셔너리 (Dict 형식의 laws와 guidelines 포함)
        """
        legal_context: LegalContext = {
            "relevant_laws": [],
            "sentencing_guidelines": [],
            "similar_precedents_summary": "검색된 법률 정보를 참고하세요."
        }
        
        # 법령 검색 결과 변환 (score >= 0.6인 결과만 포함)
        laws_result = rag_results.get("laws", {})
        if laws_result.get("status") == "success":
            laws_list = laws_result.get("laws", [])
            for law in laws_list[:5]:  # 최대 5개
                score = law.get("score") or 0.0
                # combined_score가 아니라 기본 score 사용 (나중에 strengthen_filtering에서 개선)
                #if score >= 0.6:
                legal_context["relevant_laws"].append({
                    "law_name": law.get("law_name", ""),
                    "article_no": law.get("article_no", ""),
                    "article_title": law.get("article_title", ""),
                    "text": law.get("text", "")[:200],  # 길이 제한
                    "score": score,
                    "citation_id": law.get("citation_id", "")
                })
        
        # 양형기준 검색 결과 변환 (여러 개 결과 지원)
        guidelines_result = rag_results.get("guidelines", {})
        if guidelines_result.get("status") == "success":
            guidelines_list = guidelines_result.get("guidelines", [])
            for guideline in guidelines_list[:3]:  # 최대 3개
                score = guideline.get("score") or 0.0
                #if score >= 0.6:
                legal_context["sentencing_guidelines"].append({
                    "guideline_name": guideline.get("guideline_name", ""),
                    "crime_type": guideline.get("crime_type", ""),
                    "base_range": guideline.get("base_range", {}),
                    "factors": guideline.get("factors", {}),
                    "score": score,
                    "citation_id": guideline.get("citation_id", "")
                })
        
        return legal_context
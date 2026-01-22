"""Legal Context Formatter - RAG 검색 결과를 사용자 친화적 형태로 정제"""
from typing import Dict, Any, List, Optional, Literal
from langchain_core.prompts import ChatPromptTemplate
import json

from src.config.settings import settings
from src.agents.common.state import LegalContext
from src.utils.llm_factory import create_llm
from src.utils.logger import get_logger
from src.utils.langsmith_integration import create_agent_tracer
from src.utils.json_parser import parse_agent_json_with_retry

logger = get_logger(__name__)

SENTENCING_SEX_TABLE = """강제 추행 형량기준표
1. 일반추행
유형의 정의는 형량기준표_형량범위에 구분 정보로 포함됨

형량기준표_양형인자
    [
      {
        "구분": "특별양형인자",
        "구분.1": "행위",
        "가중요소": "• 가학적·변태적 침해행위 또는 극도의 성적 불쾌감 증대 • 다수 피해자 대상 계속적·반복적 범행 • 범행에 취약한 피해자 • 친족관계인 사람의 주거침입 등 강제추행 또는 특수강제추행 범행인 경우 • 피지휘자에 대한 교사",
        "감경요소": "• 유형력의 행사가 현저히 약한 경우 • 추행의 정도가 약한 경우"
      },
      {
        "구분": "특별양형인자",
        "구분.1": "행위자/기타",
        "가중요소": "• 특정강력범죄(누범)에 해당하지 않는 동종 누범 • 신고의무자 또는 보호시설 등 종사자의 범행이거나 아동학대처벌법 제7조에 규정된 아동학대 신고의무자의 아동학대범죄에 해당하는 경우 • 상습범인 경우",
        "감경요소": "• 청각 및 언어 장애인 • 심신미약(본인 책임 없음) • 자수 • 처벌불원"
      },
      {
        "구분": "일반양형인자",
        "구분.1": "행위",
        "가중요소": "• 계획적 범행 • 비난 동기 • 심신장애 상태를 야기하여 강제추행한 경우 • 친족관계인 사람의 범행인 경우(3, 6유형) • 청소년에 대한 범행인 경우(1, 4, 5, 6유형)",
        "감경요소": "• 소극 가담 • 타인의 강압이나 위협 등에 의한 범행가담"
      },
      {
        "구분": "일반양형인자",
        "구분.1": "행위자/기타",
        "가중요소": "• 인적 신뢰관계 이용 • 특정강력범죄(누범)에 해당하지 않는 이종 누범, 누범에 해당하지 않는 동종 및 폭력 실형전과(집행종료 후 10년 미만) • 2차 피해 야기(강요죄 등 다른 범죄가 성립하는 경우는 제외)",
        "감경요소": "• 진지한 반성 • 형사처벌 전력 없음 • 상당한 피해 회복"
      }
    ]
형량기준표_형량범위
    [
      {
        "가중": "10월 ~ 2년",
        "감경": "~ 8월",
        "구분": "공중밀집장소 추행",
        "기본": "6월 ~ 1년",
        "유형": 1
      },
      {
        "가중": "1년6월 ~ 3년",
        "감경": "~ 1년",
        "구분": "일반강제추행",
        "기본": "6월 ~ 2년",
        "유형": 2
      },
      {
        "가중": "2년8월 ~ 4년8월",
        "감경": "1년 ~ 2년",
        "구분": "청소년 강제추행",
        "기본": "1년8월 ~ 3년4월",
        "유형": 3
      },
      {
        "가중": "5년 ~ 8년",
        "감경": "2년6월 ~ 4년",
        "구분": "친족관계에 의한 강제추행/특수강제추행",
        "기본": "3년 ~ 6년",
        "유형": 4
      },
      {
        "가중": "6년 ~ 9년",
        "감경": "3년6월 ~ 5년",
        "구분": "주거침입 등 강제추행",
        "기본": "4년 ~ 7년",
        "유형": 5
      },
      {
        "가중": "9년 ~ 13년",
        "감경": "5년 ~ 8년",
        "구분": "특수강도강제추행",
        "기본": "7년 ~ 11년",
        "유형": 6
      }
    ]
"""
SENTENCING_FRAUD_TABLE = """사기 형량기준표
1. 일반사기
유형의 정의
가. 제1유형
사기범죄로 인한 이득액이 1억 원 미만인 경우를 의미한다. 이득액이란 범죄행위로 인하여 취득하거나 제3자로 하여금 취득하게 한 재물 또는 재산상 이익의 가액을 의미한다(이하 같음).
나. 제2유형
사기범죄로 인한 이득액이 1억 원 이상, 5억 원 미만인 경우를 의미한다.
다. 제3유형
사기범죄로 인한 이득액이 5억 원 이상, 50억 원 미만인 경우를 의미한다.
라. 제4유형
사기범죄로 인한 이득액이 50억 원 이상, 300억 원 미만인 경우를 의미한다.
마. 제5유형
사기범죄로 인한 이득액이 300억 원 이상인 경우를 의미한다.

형량기준표_양형인자
    [
      {
        "구분": "특별양형인자",
        "구분.1": "행위",
        "가중요소": "• 불특정 또는 다수의 피해자를 대상으로 하거나 상당한 기간에 걸쳐 반복적으로  범행한 경우 • 피해자에게 심각한 피해를 야기한 경우 • 범행수법이 매우 불량하거나 재판절차에서 법원을 기망하여 소송사기 범죄를 저지른 경우 • 범죄수익을 의도적으로 은닉한 경우 • 피지휘자에 대한 교사",
        "감경요소": "• 미필적 고의로 기망행위를 저지른 경우 또는 기망행위의 정도가 약한 경우 • 손해발생의 위험이 크게 현실화되지 아니한 경우 • 사실상 압력 등에 의한 소극적 범행 가담 • 피해자에게도 범행의 발생 또는 피해의 확대에 상당한 책임이 있는 경우"
      },
      {
        "구분": "특별양형인자",
        "구분.1": "행위자/기타",
        "가중요소": "• 상습범인 경우 • 동종 누범",
        "감경요소": "• 청각 및 언어 장애인 • 심신미약(본인 책임 없음) • 자수 또는 내부 고발 • 처벌불원 또는 실질적 피해 회복"
      },
      {
        "구분": "일반양형인자",
        "구분.1": "행위",
        "가중요소": "• 비난할 만한 범행동기 • 범행에 취약한 피해자 • 인적 신뢰관계 이용",
        "감경요소": "• 기본적 생계·치료비 등의 목적이 있는 경우 • 범죄수익의 대부분을 소비하지 못하고 보유하지도 못한 경우 • 소극 가담"
      },
      {
        "구분": "일반양형인자",
        "구분.1": "행위자/기타",
        "가중요소": "• 범행 후 증거은폐 또는 은폐 시도 • 이종 누범, 누범에 해당하지 않는 동종 및 횡령배임범죄 실형전과(집행 종료 후 10년 미만) • 합의 시도 중 피해 야기(강요죄 등 다른 범죄가 성립하는 경우는 제외)",
        "감경요소": "• 심신미약(본인 책임 있음) • 진지한 반성 • 형사처벌 전력 없음 • 상당한 피해 회복"
      }
    ],
형량기준표_형량범위
    [
      {
        "가중": "1년 ~ 2년6월",
        "감경": "~ 1년",
        "구분": "1억 원 미만",
        "기본": "6월 ~ 1년6월",
        "유형": 1
      },
      {
        "가중": "2년6월 ~ 6년",
        "감경": "10월 ~ 2년6월",
        "구분": "1억 원 이상,  5억 원 미만",
        "기본": "1년 ~ 4년",
        "유형": 2
      },
      {
        "가중": "4년 ~ 8년",
        "감경": "1년6월 ~ 4년",
        "구분": "5억 원 이상,  50억 원 미만",
        "기본": "3년 ~ 6년",
        "유형": 3
      },
      {
        "가중": "6년 ~ 11년",
        "감경": "3년 ~ 6년",
        "구분": "50억 원 이상,  300억 원 미만",
        "기본": "5년 ~ 9년",
        "유형": 4
      },
      {
        "가중": "8년 ~ 17년",
        "감경": "5년 ~ 9년",
        "구분": "300억 원 이상",
        "기본": "6년 ~ 11년",
        "유형": 5
      }
    ]
"""
# 양형기준 정제용 시스템 프롬프트
SENTENCING_FORMATTER_SYSTEM_PROMPT = """당신은 법률 전문가입니다. 양형기준 검색 결과를 사용자와 AI 에이전트가 이해하기 쉽게 정제하는 역할을 합니다.

입력: 양형기준 검색 결과 (raw text, tab_title, table_classification 등)
출력: 구조화되고 읽기 쉬운 양형기준 정보

**정제 규칙:**
1. **형량범위 통합**: 같은 범죄 유형의 여러 검색 결과를 하나의 테이블로 통합
2. **가독성 개선**: 표 형식을 명확한 구조로 변환 (Markdown 테이블 사용)
3. **중복 제거**: 동일한 정보는 한 번만 표시
4. **핵심 정보 강조**: 유형별 형량 범위, 가중/감경 요소를 명확히 구분
5. **원문 보존**: 법적 정확성을 위해 원문 내용은 변경하지 않고 구조만 개선
6. **컨텍스트 활용**: 제공된 전체 양형기준표와 현재 사건의 양형 속성을 참고하여, 검색된 파편화된 데이터를 완전한 형태로 재구성

**출력 형식 (JSON):**
```json
{{
  "formatted_guidelines": [
    {{
      "crime_type": "사기",
      "guideline_name": "2. 조직적 사기",
      "summary": "피해액 규모에 따라 5개 유형으로 구분",
      "sentencing_table": "| 유형 | 피해액 | 기본형 | 가중 | 감경 |\\n|---|---|---|---|---|\\n| 제1유형 | 1억 미만 | 1년6월~3년 | 2년6월~4년 | 1년~2년6월 |\\n...",
      "aggravating_factors": ["조직적 사기", "다수 피해자"],
      "mitigating_factors": ["피해 회복", "합의"],
      "source_citations": ["C_GUIDE_1", "C_GUIDE_2"]
    }}
  ],
  "evaluation_principles": [
    {{
      "title": "형량범위의 결정방법",
      "content": "형량범위는 특별양형인자를 고려하여 결정한다."
    }},
    {{
      "title": "선고형의 결정방법",
      "content": "양형기준상 형량범위 상한이 25년을 초과하는 경우에는 무기징역을 선택할 수 있다."
    }}
  ]
}}
```

**중요:** 반드시 유효한 JSON만 출력하세요. 추가 설명이나 마크다운 코드 블록 없이 순수 JSON만 반환하세요.
"""

SENTENCING_FORMATTER_HUMAN_TEMPLATE = """다음 양형기준 검색 결과를 정제해주세요:

범죄 유형: {crime_type}

현재 사건 양형 속성:
{sentencing_attributes}

참고 자료 (전체 양형기준표):
{reference_table}

검색 결과 (파편화된 데이터):
{raw_guidelines}

위 검색 결과를 전체 양형기준표와 현재 사건의 양형 속성을 참고하여, 사용자가 이해하기 쉽게 정제하여 JSON 형식으로 반환하세요.
파편화된 검색 결과만으로는 정보가 부족할 수 있으므로, 참고 자료의 전체 양형기준표를 활용하여 완전한 형태로 재구성하세요.
"""


# 법령 정제용 시스템 프롬프트
LAW_FORMATTER_SYSTEM_PROMPT = """당신은 법률 전문가입니다. 법령 검색 결과를 사용자와 AI 에이전트가 이해하기 쉽게 정제하는 역할을 합니다.

입력: 법령 검색 결과 (law_name, article_no, text 등)
출력: 구조화되고 읽기 쉬운 법령 정보

**정제 규칙:**
1. **관련성 순 정렬**: 가장 관련성 높은 조문부터 표시
2. **조문 요약**: 긴 조문은 핵심 내용 요약 제공
3. **중복 제거**: 같은 조문의 중복 검색 결과 통합
4. **구성요건 명시**: 범죄 성립 요건을 명확히 표시
5. **처벌 규정 강조**: 형량 정보를 명확히 구분

**출력 형식 (JSON):**
```json
{{
  "formatted_laws": [
    {{
      "law_name": "형법",
      "article_no": "제347조",
      "article_title": "사기",
      "summary": "타인을 기망하여 재물 교부받거나 재산상 이익을 취득한 경우",
      "key_elements": ["기망행위", "착오 유발", "재물 교부", "인과관계"],
      "punishment": "10년 이하의 징역 또는 2천만원 이하의 벌금",
      "full_text": "사람을 기망하여 재물의 교부를 받거나 재산상의 이익을 취득한 자는...",
      "source_citation": "C_LAW_1",
      "relevance_score": 0.92
    }}
  ]
}}
```

**중요:** 반드시 유효한 JSON만 출력하세요. 추가 설명이나 마크다운 코드 블록 없이 순수 JSON만 반환하세요.
"""

LAW_FORMATTER_HUMAN_TEMPLATE = """다음 법령 검색 결과를 정제해주세요:

범죄 유형: {crime_type}
검색 결과:
{raw_laws}

위 검색 결과를 사용자가 이해하기 쉽게 정제하여 JSON 형식으로 반환하세요.
"""


class LegalContextFormatter:
    """
    Legal Context Formatter - RAG 검색 결과 정제

    역할:
    - RAG 검색으로 얻은 raw 데이터를 LLM으로 정제
    - 양형기준: 표 형식 통합, 가독성 개선, 중복 제거
    - 법령: 조문 요약, 구성요건 명시, 처벌 규정 강조
    - 사용자 및 에이전트가 state를 참조하기 쉬운 구조화된 데이터 생성

    설계 원칙:
    - 단일 책임: 정제만 담당 (검색은 Retriever, 최종 조합은 Agent)
    - 느슨한 결합: Retriever나 Agent와 독립적으로 동작
    - 확장 가능: 새로운 정제 규칙 추가 용이
    """

    def __init__(self, provider: Optional[Literal["openai", "upstage"]] = None):
        """
        LegalContextFormatter 초기화

        Args:
            provider: LLM provider 선택 ("openai" 또는 "upstage")
                     None이면 settings.default_llm_provider 사용
        """
        # 정제 작업은 정확성이 중요하므로 낮은 temperature 사용
        self.llm = create_llm(provider=provider, temperature=0.3)
        self.provider = provider or settings.default_llm_provider

        # 양형기준 정제용 체인
        self.sentencing_prompt = ChatPromptTemplate.from_messages([
            ("system", SENTENCING_FORMATTER_SYSTEM_PROMPT),
            ("human", SENTENCING_FORMATTER_HUMAN_TEMPLATE)
        ])
        self.sentencing_chain = self.sentencing_prompt | self.llm

        # 법령 정제용 체인
        self.law_prompt = ChatPromptTemplate.from_messages([
            ("system", LAW_FORMATTER_SYSTEM_PROMPT),
            ("human", LAW_FORMATTER_HUMAN_TEMPLATE)
        ])
        self.law_chain = self.law_prompt | self.llm

        logger.info(f"LegalContextFormatter initialized with {self.provider.upper()} LLM")

    async def format_sentencing_guidelines(
        self,
        raw_guidelines: List[Dict[str, Any]],
        crime_type: str,
        state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        양형기준 검색 결과를 정제합니다.

        Args:
            raw_guidelines: Retriever에서 받은 raw 양형기준 리스트
            crime_type: 범죄 유형
            state: LangSmith 트레이싱용 state (선택적)

        Returns:
            정제된 양형기준 정보
        """
        if not raw_guidelines:
            return {
                "formatted_guidelines": [],
                "evaluation_principles": [],
                "formatting_status": "no_data"
            }

        # LangSmith 트레이싱 설정
        langsmith_config = {}
        if state:
            tracer = create_agent_tracer("legal_context_formatter", state)
            langsmith_config = {
                "metadata": tracer.get_metadata(
                    formatter_type="sentencing_guidelines",
                    raw_count=len(raw_guidelines),
                    crime_type=crime_type
                ),
                "tags": tracer.get_tags(action="format_sentencing_guidelines")
            }

        # 범죄 유형에 따른 참고 테이블 선택
        crime_type_lower = crime_type.lower()
        if "사기" in crime_type_lower:
            reference_table = SENTENCING_FRAUD_TABLE
        elif "강제추행" in crime_type_lower or "성범죄" in crime_type_lower:
            reference_table = SENTENCING_SEX_TABLE
        else:
            reference_table = "해당 범죄 유형의 전체 양형기준표가 제공되지 않았습니다."

        # 현재 사건의 양형 속성 추출
        sentencing_attributes = {}
        if state:
            sentencing_factors = state.get("sentencing_factors", {})
            sentencing_attributes = {
                "피해액": sentencing_factors.get("amount_gained", "정보 없음"),
                "동종전과": sentencing_factors.get("criminal_history", "정보 없음"),
                "피해 회복": sentencing_factors.get("restitution", "정보 없음"),
                "합의 여부": sentencing_factors.get("settlement", "정보 없음"),
                "사건 유형": crime_type
            }

        sentencing_attributes_str = json.dumps(sentencing_attributes, ensure_ascii=False, indent=2)

        # LLM 입력 준비
        chain_input = {
            "crime_type": crime_type,
            "sentencing_attributes": sentencing_attributes_str,
            "reference_table": reference_table,
            "raw_guidelines": json.dumps(raw_guidelines, ensure_ascii=False, indent=2)
        }

        required_fields = ["formatted_guidelines", "evaluation_principles"]

        try:
            formatted_result = await parse_agent_json_with_retry(
                llm_chain=self.sentencing_chain,
                chain_input=chain_input,
                required_fields=required_fields,
                max_retries=2,
                config=langsmith_config
            )
            formatted_result["formatting_status"] = "success"
            return formatted_result

        except ValueError as e:
            logger.error(f"Sentencing guidelines formatting failed after retries: {e}")

            # Phoenix에 품질 저하 기록
            try:
                from src.utils.phoenix_integration import is_phoenix_enabled
                if is_phoenix_enabled():
                    from opentelemetry import trace
                    current_span = trace.get_current_span()
                    if current_span and current_span.is_recording():
                        current_span.set_attribute("formatter.quality_degraded", True)
                        current_span.set_attribute("formatter.fallback_reason", "json_parse_error_all_retries")
                        current_span.set_attribute("formatter.error_type", type(e).__name__)
            except Exception:
                pass

            # 폴백: raw 데이터를 최소한의 구조로 반환
            return {
                "formatted_guidelines": raw_guidelines,  # raw 데이터 그대로 반환
                "evaluation_principles": [],
                "formatting_status": "failed_fallback_to_raw",
                "formatting_error": str(e)
            }

    async def format_legal_provisions(
        self,
        raw_laws: List[Dict[str, Any]],
        crime_type: str,
        state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        법령 검색 결과를 정제합니다.

        Args:
            raw_laws: Retriever에서 받은 raw 법령 리스트
            crime_type: 범죄 유형
            state: LangSmith 트레이싱용 state (선택적)

        Returns:
            정제된 법령 정보
        """
        if not raw_laws:
            return {
                "formatted_laws": [],
                "formatting_status": "no_data"
            }

        # LangSmith 트레이싱 설정
        langsmith_config = {}
        if state:
            tracer = create_agent_tracer("legal_context_formatter", state)
            langsmith_config = {
                "metadata": tracer.get_metadata(
                    formatter_type="legal_provisions",
                    raw_count=len(raw_laws),
                    crime_type=crime_type
                ),
                "tags": tracer.get_tags(action="format_legal_provisions")
            }

        # LLM 입력 준비
        chain_input = {
            "crime_type": crime_type,
            "raw_laws": json.dumps(raw_laws, ensure_ascii=False, indent=2)
        }

        required_fields = ["formatted_laws"]

        try:
            formatted_result = await parse_agent_json_with_retry(
                llm_chain=self.law_chain,
                chain_input=chain_input,
                required_fields=required_fields,
                max_retries=2,
                config=langsmith_config
            )
            formatted_result["formatting_status"] = "success"
            return formatted_result

        except ValueError as e:
            logger.error(f"Legal provisions formatting failed after retries: {e}")

            # Phoenix에 품질 저하 기록
            try:
                from src.utils.phoenix_integration import is_phoenix_enabled
                if is_phoenix_enabled():
                    from opentelemetry import trace
                    current_span = trace.get_current_span()
                    if current_span and current_span.is_recording():
                        current_span.set_attribute("formatter.quality_degraded", True)
                        current_span.set_attribute("formatter.fallback_reason", "json_parse_error_all_retries")
                        current_span.set_attribute("formatter.error_type", type(e).__name__)
            except Exception:
                pass

            # 폴백: raw 데이터를 최소한의 구조로 반환
            return {
                "formatted_laws": raw_laws,  # raw 데이터 그대로 반환
                "formatting_status": "failed_fallback_to_raw",
                "formatting_error": str(e)
            }

    async def format_full_legal_context(
        self,
        raw_legal_context: LegalContext,
        crime_type: str,
        state: Optional[Dict[str, Any]] = None
    ) -> LegalContext:
        """
        전체 LegalContext를 정제합니다 (법령 + 양형기준).

        Args:
            raw_legal_context: Retriever에서 받은 raw LegalContext
            crime_type: 범죄 유형
            state: LangSmith 트레이싱용 state (선택적)

        Returns:
            정제된 LegalContext
        """
        formatted_context = raw_legal_context.copy()

        # 양형기준 정제
        if raw_legal_context.get("sentencing_guidelines"):
            sentencing_result = await self.format_sentencing_guidelines(
                raw_guidelines=raw_legal_context["sentencing_guidelines"],
                crime_type=crime_type,
                state=state
            )
            formatted_context["sentencing_guidelines"] = sentencing_result.get("formatted_guidelines", [])
            formatted_context["evaluation_principles"] = sentencing_result.get("evaluation_principles", [])
            formatted_context["sentencing_formatting_status"] = sentencing_result.get("formatting_status", "unknown")

        # 법령 정제 (relevant_laws)
        if raw_legal_context.get("relevant_laws"):
            laws_result = await self.format_legal_provisions(
                raw_laws=raw_legal_context["relevant_laws"],
                crime_type=crime_type,
                state=state
            )
            formatted_context["relevant_laws"] = laws_result.get("formatted_laws", [])
            formatted_context["laws_formatting_status"] = laws_result.get("formatting_status", "unknown")

        return formatted_context

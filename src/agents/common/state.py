"""
LangGraph State Definition - 2025 BupBlessYou Standard
모든 에이전트와 프론트엔드가 공유하는 데이터 구조 정의 파일입니다.
"""

from enum import Enum
from typing import Annotated, Sequence, TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# --- 1. 공통 상수 정의 (Constants) ---
# Enum으로 정의하여 속성 접근 가능하도록 함
class Role(str, Enum):
    PROSECUTOR = "prosecutor"
    DEFENSE = "defense"
    LEGAL_ADVISOR = "legal_advisor"
    JUDGE = "judge"
    USER = "user"
    SYSTEM = "system"

class Phase(str, Enum):
    BRIEFING = "briefing"
    DEBATE = "debate"
    JUDGMENT = "judgment"
    RESULT = "result"

# --- 2. 하위 데이터 구조체 (Sub-structures) ---

class CaseAttribute(TypedDict):
    """
    피고인 및 사건의 핵심 속성 (자문 AI 추출 or DB 로드)
    검사/변호사가 논리 구성에 사용하는 '팩트 조각'들입니다.
    """
    key: str         # 예: "criminal_history", "damage_recovery"
    value: str       # 예: "first_offense", "none"
    description: str # 예: "초범임", "피해 회복되지 않음"

class LegalContext(TypedDict, total=False):
    """
    모든 에이전트가 공유하는 법률 배경 지식 (자문 AI가 제공)
    
    relevant_laws와 sentencing_guidelines는 Dict 또는 str 모두 지원 (하위 호환성)
    - Dict 형식: {"law_name": "...", "article_no": "...", "score": 0.85, "citation_id": "...", ...}
    - str 형식: "형법 제347조" (기존 형식, 하위 호환성 유지)
    """
    relevant_laws: List[Any]        # 관련 법령 (Dict 또는 str)
    sentencing_guidelines: List[Any] # 양형 기준 (Dict 또는 str)
    similar_precedents_summary: str  # 유사 판례 경향 요약 (스포일러 방지)

class AgentOutput(TypedDict):
    """
    [중요] 에이전트가 생성하는 메시지의 페이로드 규격
    LangChain Message의 content나 additional_kwargs에 JSON 형태로 담깁니다.
    """
    role: Role
    content: str            # 실제 대사
    emotion: Optional[str]   # "neutral", "angry", "confident", "shocked", "thinking" (선택 사항)
    references: List[str]   # 발언의 근거 (법령, 판례 등)

# --- 3. 메인 상태 정의 (Main State) ---

class CourtSimulationState(TypedDict):
    """
    법정 시뮬레이션 전체 상태 관리 (Shared Memory)
    """
    
    # [1] 커뮤니케이션 (Communication)
    # add_messages 리듀서를 통해 대화 내역이 자동으로 누적됩니다.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # [2] 흐름 제어 (Flow Control)
    current_phase: Phase    # 현재 진행 단계
    next_speaker: Role      # 다음 발언 권한을 가진 에이전트
    turn_count: int         # 전체 게임 턴 수
    phase_turn: int         # 현재 debate 페이즈 내 턴 수 (검사-변호사 대화만, user_judge 후 리셋)
    
    # [3] 공개 컨텍스트 (Public Context) - 모든 AI/유저 공유
    case_id: Optional[int]          # DB PK (int4)
    case_number: str                # 사건 번호 (예: "2024고합123")
    case_summary: str               # 사건 개요 텍스트
    case_attributes: List[CaseAttribute] # 사건 속성 리스트
    sentencing_factors: Optional[Dict[str, Any]]  # 양형 고려 요소 (DB에서 추출된 구조화된 데이터)
    legal_context: LegalContext     # 법률 자문 정보

    # [4] 공방 평가 로그 저장
    evaluations: List[Dict[str, Any]]      # 예: 현재 턴의 공방 평가
    evaluations_log: List[Dict[str, Any]]  # 전체 평가 누적 로그
    round_summary : Optional[Dict[str, Any]] = None
    judge_round: int                       # 판사 개입 기준 라운드 (기본값: 1)

    # [4] 유저 판결 (User Input)
    user_verdict: Optional[str]     # 유저가 내린 주문 (예: "징역 1년")
    user_sentence: Optional[Dict[str, Any]] # 유저가 입력한 형량 구조체
    user_sentence_text: Optional[str] # 유저가 입력한 형량 텍스트 (예: "징역 1년")
    user_reasoning: Optional[str]   # 유저의 양형 이유
    choices: Optional[List[Dict]]   # 유저에게 제시할 선택지 리스트 [{"id":..., "label":..., "value":...}]
    
    # [5] 정답 및 분석 (Hidden / System Only)
    # 주의: 이 정보는 Debate 단계의 검사/변호사 에이전트에게 노출되면 안 됩니다.
    actual_judgment: Optional[Dict] # 실제 판결 결과 { "verdict": ..., "reason": ... }
    analysis_result: Optional[Dict] # 유저 판결 vs 실제 판결 분석 리포트

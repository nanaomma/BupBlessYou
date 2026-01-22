"""FastAPI application entry point"""
import uuid
import json
import time
import asyncio
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config.settings import settings
from src.agents.common.graph import CourtSimulationGraph
from src.agents.common.state import CourtSimulationState
from src.services.simulation_service import simulation_service
from src.services.case_service import get_random_case_by_scenario, get_available_scenarios, get_case_by_id
from src.utils.logger import setup_logging, get_logger
from src.utils.langsmith_integration import setup_langsmith
from src.utils.phoenix_integration import setup_phoenix, is_phoenix_enabled, get_phoenix_dashboard_url
from src.utils.ragas_integration import setup_arize_evaluation
from src.middleware.debug_middleware import DebugLoggingMiddleware
# NOTE: 충돌나서 phoenix_integration.py 사용
# from src.utils.phoenix_monitoring import setup_phoenix

# Setup centralized logging
setup_logging()
logger = get_logger(__name__)

# Setup LangSmith tracing (if enabled)
setup_langsmith()

# Setup Phoenix tracing (optional, if enabled)
phoenix_enabled = setup_phoenix()
if phoenix_enabled:
    logger.info(f"🔥 Phoenix Dashboard: {get_phoenix_dashboard_url()}")

# Setup Arize AX Evaluation (RAGAS)
setup_arize_evaluation()
logger.info("✅ Arize AX RAGAS evaluation initialized")

# Background task for session cleanup
async def cleanup_expired_sessions():
    """Background task to cleanup expired sessions"""
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            current_time = time.time()
            expired_sessions = [
                session_id for session_id, session_data in sessions.items()
                if current_time - session_data.get("last_activity", 0) > SESSION_TIMEOUT
            ]

            for session_id in expired_sessions:
                try:
                    # Close graph resources
                    manager = sessions[session_id].get("manager")
                    if manager:
                        await manager.close()
                    del sessions[session_id]
                    logger.info(f"Session cleaned up: {session_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up session {session_id}: {e}")

            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting background session cleanup task")
    cleanup_task = asyncio.create_task(cleanup_expired_sessions())

    yield

    # Shutdown
    logger.info("Shutting down: cleaning up all sessions")
    cleanup_task.cancel()
    for session_id, session_data in list(sessions.items()):
        try:
            manager = session_data.get("manager")
            if manager:
                await manager.close()
        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
    sessions.clear()
    logger.info("All sessions cleaned up")

app = FastAPI(
    title="BupBlessYou API",
    description="AI Mock Court Simulator Backend",
    version="0.2.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug logging middleware
if settings.debug:
    app.add_middleware(DebugLoggingMiddleware)

# Static & Templates
app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")

# Session Storage (In-Memory) with Metadata
sessions: Dict[str, Dict[str, Any]] = {}
SESSION_TIMEOUT = 3600  # 1시간 (초 단위)

# --- Models ---
class InitRequest(BaseModel):
    case_summary: Optional[str] = None # Optional custom case
    case_number: Optional[str] = None
    case_id: Optional[int] = None

class ActionRequest(BaseModel):
    session_id: str
    action_type: str  # 'next', 'judgment', 'choice'
    payload: Dict[str, Any] = {}

class GameResponse(BaseModel):
    session_id: str
    current_phase: str
    speaker: str
    content: str
    emotion: str
    references: List[str] = []
    case_info: str = ""
    legal_context: str = ""
    sentencing_info: str = ""  # 양형 기준 정보 (HTML 포맷)
    choices: List[Dict[str, Any]] = []
    history: List[Dict[str, str]] = []
    evaluations: List[Dict[str, Any]] = []
    evaluations_log: List[Dict[str, Any]] = []
    round_summary: Optional[Dict[str, Any]] = None
    # Judgment analysis fields
    analysis_result: Optional[Dict[str, Any]] = None
    user_verdict: Optional[str] = None
    user_sentence_text: Optional[str] = None
    user_reasoning: Optional[str] = None
    actual_judgment: Optional[Dict[str, Any]] = None
# --- Routes ---

@app.get("/")
async def root(request: Request):
    """Serve the scenario selection page"""
    return templates.TemplateResponse("scenario_selection.html", {"request": request})

@app.get("/game")
async def game(request: Request):
    """Serve the game interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/scenarios")
async def get_scenarios():
    """Get available scenarios"""
    scenarios = await get_available_scenarios()
    return {"scenarios": scenarios}

@app.post("/api/select-scenario")
async def select_scenario(request: Request):
    """Select a scenario and get a random case"""
    # Accept scenario_type from query parameter (for compatibility with current frontend)
    scenario_type = request.query_params.get("scenario_type")

    if not scenario_type:
        raise HTTPException(status_code=400, detail="scenario_type parameter is required")

    case = await get_random_case_by_scenario(scenario_type)
    if not case:
        raise HTTPException(status_code=404, detail=f"No cases found for scenario: {scenario_type}")
    return {"case": case}

@app.get("/health")
async def health_check():
    """Health check endpoint with observability status"""
    return {
        "status": "healthy",
        "observability": {
            "langsmith": settings.langsmith_tracing,
            "phoenix": is_phoenix_enabled(),
            "phoenix_dashboard": get_phoenix_dashboard_url() if is_phoenix_enabled() else None
        }
    }

@app.post("/api/cleanup-session")
async def cleanup_session(request: Request):
    """Cleanup session resources when user explicitly ends the game"""
    data = await request.json()
    session_id = data.get("session_id")

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    if session_id not in sessions:
        # Session already cleaned up or doesn't exist
        return {"status": "ok", "message": "Session already cleaned up"}

    try:
        # Close graph resources
        manager = sessions[session_id].get("manager")
        if manager:
            await manager.close()

        del sessions[session_id]
        logger.info(f"Session manually cleaned up: {session_id}")

        return {"status": "ok", "message": "Session cleaned up successfully"}
    except Exception as e:
        logger.error(f"Error cleaning up session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.post("/api/init", response_model=GameResponse)
async def init_game(request: InitRequest):
    """Initialize a new game session"""
    session_id = str(uuid.uuid4())
    
    # Initialize Graph
    court_graph = CourtSimulationGraph()
    await court_graph.setup()
    
    # Compile
    graph = court_graph.compile(enable_interrupts=True)
    
    sessions[session_id] = {
        "graph": graph,
        "thread_id": session_id, # Use session_id as unique thread_id
        "manager": court_graph, # Keep ref to close later if needed
        "last_activity": time.time(), # Track last activity for timeout
        "created_at": time.time()
    }
    
    # 1. Determine Initial Case Data
    case_summary = request.case_summary
    case_number = request.case_number
    actual_judgment = None
    sentencing_factors = None
    case_id = request.case_id

    # If case_id is provided, try to fetch from DB
    if case_id:
        case_data = await get_case_by_id(case_id)
        if case_data:
            # Override with DB data
            case_summary = case_data.get("description") or case_summary
            case_number = case_data.get("case_number") or case_number
            sentencing_factors = case_data.get("sentencing_factors")
            
            # Construct actual_judgment dict
            if case_data.get("actual_label"):
                actual_judgment = {
                    "verdict": case_data.get("actual_label"),
                    "reason": case_data.get("actual_reason", "")
                }
            logger.info(f"Initialized game with Case ID {case_id}: {case_number}")
        else:
            logger.warning(f"Case ID {case_id} provided but not found in DB.")

    # Fallback default case if summary is still empty
    if not case_summary:
        case_summary = (
            "피고인은 2024년 12월 25일 서울 명동 거리에서 산타 복장을 하고 "
            "지나가는 행인들에게 '착한 일 안 하면 선물 없다'며 얼음이 섞인 눈뭉치를 강하게 던져 "
            "피해자 김철수(30세)의 안면부에 전치 2주의 타박상을 입히고 안경을 파손시켰다."
        )

    initial_input = {
        "case_summary": case_summary,
        "case_number": case_number,
        "case_id": case_id,
        "actual_judgment": actual_judgment,
        "sentencing_factors": sentencing_factors,
        "turn_count": 0,
        "phase_turn": 0,  # 페이즈별 턴 초기화
        "messages": []
    }
    
    # Run until first interruption or end
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # First run (Briefing -> ... -> Router -> First Speaker)
        # Using ainvoke to start
        final_state = await graph.ainvoke(initial_input, config)
        return _format_response(session_id, final_state)
    except Exception as e:
        logger.error(f"Init failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/action", response_model=GameResponse)
async def game_action(request: ActionRequest):
    """Process user action and advance game"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[request.session_id]

    # Update last activity timestamp
    session["last_activity"] = time.time()

    graph = session["graph"]
    config = {"configurable": {"thread_id": session["thread_id"]}}

    input_data = None
    
    # Handle Action Types
    if request.action_type == "next":
        user_input = request.payload.get("user_input")
        if user_input:
            # ===== Pure Debate Architecture: 유저 질문 시 Legal Context 갱신 =====
            # 유저 질문은 항상 legal context 갱신 트리거
            current_state = await graph.aget_state(config)
            # court_graph = session["manager"] # No longer needed for advisor access

            # Legal Advisor 직접 호출하여 컨텍스트 갱신 (Service Layer 사용)
            updated_legal_context = await simulation_service.advisor.update_legal_context(
                {**current_state.values, "new_question": user_input}
            )

            input_data = {
                "messages": [
                    {"role": "user", "content": user_input}
                ],
                "legal_context": updated_legal_context.get("legal_context", {}),  # 갱신된 컨텍스트
                "choices": []  # Clear choices when user takes action
            }
        else:
            input_data = None # Resume normally
            
    elif request.action_type == "choice":
        # User selected a suggested question
        choice_id = request.payload.get("choice_id")
        # We need to find the text. But for simplicity, let's assume payload has text or we just send ID.
        # Ideally frontend sends value. Let's assume frontend sends choice_id which maps to value?
        # Let's check game.js... it sends { choice_id: choice.id }.
        # State choices are [ {id, label, value} ].
        # We need to retrieve the text from the graph state, but we don't have easy access here without reading state first.
        # Strategy: Pass the full text from frontend or read state.
        # Reading state:
        current_state = await graph.aget_state(config)
        choices = current_state.values.get("choices", [])
        selected_text = next((c["value"] for c in choices if c["id"] == choice_id), None)
        
        if selected_text:
             input_data = {
                "messages": [
                    {"role": "user", "content": selected_text}
                ],
                "choices": [] # Clear choices
            }
        else:
             input_data = None

    elif request.action_type == "enter_judgment":
        # Force transition to Judgment Phase
        # State를 실제로 업데이트하고 그래프를 실행하여 judgment UI 표시
        input_data = {
            "current_phase": "judgment",
            "choices": []
        }

    elif request.action_type == "judgment":
        # ===== CRITICAL: This is the final, irreversible action =====
        # Once judgment is submitted, debate phase is PERMANENTLY ended
        payload = request.payload

        # New structured format
        if "sentence" in payload:
            verdict = payload.get("verdict", "guilty")
            sentence = payload.get("sentence", {})
            sentence_text = payload.get("sentence_text", "")
            reasoning = payload.get("reasoning", "")

            # Format full judgment message
            judgment_message = f"판결 선고: {sentence_text}"
            if reasoning:
                judgment_message += f"\n\n양형 이유: {reasoning}"

            input_data = {
                "messages": [
                    {"role": "user", "content": judgment_message}
                ],
                "user_verdict": verdict,
                "user_sentence": sentence,  # Structured sentence data
                "user_sentence_text": sentence_text,  # Human-readable sentence
                "user_reasoning": reasoning,
                "current_phase": "result",  # LOCKED: Cannot return to debate
                "debate_ended": True,       # Flag to prevent phase rollback
                "choices": []               # Clear all choices - no more questions
            }
        # Fallback: Old text-based format (for backward compatibility)
        else:
            user_text = payload.get("user_text", "")
            input_data = {
                "messages": [
                    {"role": "user", "content": f"판결 선고: {user_text}"}
                ],
                "user_verdict": "guilty",  # Default to guilty
                "user_reasoning": user_text,
                "current_phase": "result",  # LOCKED: Cannot return to debate
                "debate_ended": True,       # Flag to prevent phase rollback
                "choices": []               # Clear all choices
            }
        
    else:
        raise HTTPException(status_code=400, detail="Invalid action type")

    try:
        # Resume with astream_events pattern
        # ===== Phase 1: Explicit State Update & Resume (Fix for Reset Issue) =====

        # 1. Update State explicitly if there is input
        if input_data:
            logger.debug(f"Updating state with: {list(input_data.keys())}")
            await graph.aupdate_state(config, input_data)

        # 2. Resume graph execution (input=None tells LangGraph to continue from checkpoint)
        stream_input = None

        # 이벤트 스트리밍으로 그래프 재개
        final_state = None
        event_count = 0

        async for event in graph.astream_events(stream_input, config, version="v2"):
            event_type = event.get("event")
            event_name = event.get("name", "")
            event_count += 1

            # 디버그 로깅 필터링 (스트리밍 이벤트 제외)
            if settings.debug:

                # 필터링할 이벤트 타입
                filtered_events = [
                    "on_chat_model_stream",
                    "on_prompt_start",
                    "on_prompt_end",
                    "on_parser_start",
                    "on_parser_end",
                    "on_chain_stream",
                    "openinference",
                ]

                if event_type not in filtered_events:
                    logger.debug(f"[Event {event_count}] {event_type} | {event_name}")

            # 노드 완료 이벤트 감지 (향후 Phoenix/RAGAS 통합 지점)
            if event_type == "on_chain_end":
                # 특정 노드 완료 시 로깅
                if event_name in ["prosecutor", "defense", "judge", "legal_advisor"]:
                    logger.info(f"Node '{event_name}' completed")

                # 최종 상태 추출 (__end__ = 그래프 완료)
                if event_name == "__end__":
                    final_state = event["data"]["output"]
                    logger.debug("Graph execution completed, final state extracted")
                    break

            # LangGraph 노드 완료 이벤트에서도 상태 추출 (fallback)
            if event_type == "on_chain_end" and event_name == "LangGraph":
                if final_state is None:  # __end__를 못 찾은 경우에만
                    final_state = event["data"].get("output")
                    if final_state:
                        logger.debug("Final state extracted from LangGraph end event")

        # 3. Fallback: 이벤트로 상태를 못 얻으면 직접 조회
        if final_state is None:
            logger.debug("Final state not found in events, using aget_state")
            state_snapshot = await graph.aget_state(config)
            final_state = state_snapshot.values

        logger.info(f"Action processed: {event_count} events, phase: {final_state.get('current_phase')}")
        return _format_response(request.session_id, final_state)

    except Exception as e:
        logger.error(f"Action failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _format_response(session_id: str, state: Dict[str, Any]) -> GameResponse:
    """Map LangGraph state to GameResponse"""
    
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    
    # 1. Determine Speaker & Content (Current Turn)
    speaker = "system"
    content = ""
    emotion = "neutral"
    references = []
    
    if last_msg:
        if hasattr(last_msg, "content"):
            raw_content = last_msg.content
            # Try parsing JSON (Agent Output)
            if isinstance(raw_content, str):
                try:
                    data = json.loads(raw_content, strict=False)
                    if isinstance(data, dict):
                        speaker = data.get("role", "system")
                        content = data.get("content", "")
                        emotion = data.get("emotion", "neutral")
                        references = data.get("references", [])
                    else:
                        content = raw_content
                except Exception as e:
                    logger.warning(f"Failed to parse JSON content: {e}. Raw: {raw_content[:100]}...")
                    content = raw_content
                    # Heuristic: Briefing message?
                    if state.get("current_phase") == "briefing":
                        speaker = "judge" # Fallback for briefing if parsing fails
            else:
                content = str(raw_content)

    # 2. Case nfo & Context
    case_info = state.get("case_summary", "")
    
    # Format Legal Context (supports both formatted and raw data)
    legal_ctx = state.get("legal_context", {})
    legal_text = ""
    if isinstance(legal_ctx, dict) and legal_ctx:
        # ✅ Format relevant_laws (contains formatted data from LegalContextFormatter)
        laws = legal_ctx.get("relevant_laws", [])
        if laws:
            legal_text += "<b>[관련 법령]</b><br>"
            for law in laws:
                if isinstance(law, dict):
                    law_name = law.get("law_name", "")
                    article_no = law.get("article_no", "")
                    summary = law.get("summary", "")  # Formatted summary from LegalContextFormatter
                    if summary:
                        legal_text += f"• {law_name} {article_no}: {summary}<br>"
                    else:
                        legal_text += f"• {law_name} {article_no}<br>"
            legal_text += "<br>"

        # TODO: 유사 판례 실제 RAG 구현 후 주석 해제
        # 현재 Prosecutor/Defense 에이전트에서 더미 데이터만 반환하므로 표시 비활성화
        # precedents = legal_ctx.get("similar_precedents_summary", "")
        # if precedents:
        #      legal_text += "<b>[유사 판례]</b><br>" + precedents

    logger.debug(f"Legal context formatting: has_data={bool(legal_ctx)}, relevant_laws={bool(legal_ctx.get('relevant_laws'))}, formatted_length={len(legal_text)}")

    # Format Sentencing Info (양형 정보 섹션)
    sentencing_text = ""

    # 1. Format sentencing_guidelines (RAG 검색 결과 - LegalContextFormatter에서 정제됨)
    sentencing_guidelines = legal_ctx.get("sentencing_guidelines", []) if legal_ctx else []
    if sentencing_guidelines:
        sentencing_text += "<b>[양형 기준표]</b><br>"
        for guideline in sentencing_guidelines:
            if isinstance(guideline, dict):
                guideline_name = guideline.get("guideline_name", "")
                summary = guideline.get("summary", "")
                sentencing_table = guideline.get("sentencing_table", "")

                if guideline_name:
                    sentencing_text += f"<b>{guideline_name}</b><br>"
                if summary:
                    sentencing_text += f"{summary}<br>"
                if sentencing_table:
                    # Markdown 테이블을 HTML로 변환 (간단한 방식)
                    sentencing_text += f"<pre>{sentencing_table}</pre><br>"
        sentencing_text += "<br>"

    # 2. Format sentencing_factors (DB에서 가져온 데이터)
    sentencing_factors = state.get("sentencing_factors", {})
    # 만약 simulation_service에서 이미 포맷팅된 정보가 있다면 그것을 우선 사용
    preformatted_info = state.get("sentencing_info", "")
    
    if preformatted_info and isinstance(preformatted_info, str) and "<ul>" in preformatted_info:
        if sentencing_text: # 이미 양형기준표 내용이 있다면 구분선 추가
            sentencing_text += "<hr>"
        sentencing_text += preformatted_info
    elif isinstance(sentencing_factors, dict) and sentencing_factors:
        if sentencing_text:
            sentencing_text += "<hr>"
            
        sentencing_text += "<h4>⚖️ 양형 참고 자료</h4><ul>"
        
        valid_factors_html = []
        for category, details in sentencing_factors.items():
            if isinstance(details, dict):
                for factor_name, value in details.items():
                    if value is not None:
                        valid_factors_html.append(f"<li><b>{factor_name}</b>: {value}</li>")
            elif details is not None:
                valid_factors_html.append(f"<li><b>{category}</b>: {details}</li>")
        
        if valid_factors_html:
            sentencing_text += "".join(valid_factors_html)
        else:
             sentencing_text += "<li>정보 없음</li>"
             
        sentencing_text += "</ul>"

    logger.debug(f"Sentencing info formatting: has_guidelines={bool(sentencing_guidelines)}, has_factors={bool(sentencing_factors)}, formatted_length={len(sentencing_text)}")

    # 3. Format History
    history_list = []
    for msg in messages:
        msg_content = msg.content
        msg_role = "system"
        msg_text = msg_content
        
        try:
            data = json.loads(msg_content)
            if isinstance(data, dict):
                msg_role = data.get("role", "system")
                msg_text = data.get("content", "")
        except:
            pass
            
        # Skip system logic messages if empty or internal
        if msg_text:
            history_list.append({
                "role": msg_role,
                "content": msg_text
            })

    return GameResponse(
        session_id=session_id,
        current_phase=state.get("current_phase", "unknown"),
        speaker=speaker,
        content=content,
        emotion=emotion,
        references=references,
        case_info=case_info,
        legal_context=legal_text,
        sentencing_info=sentencing_text,
        history=history_list,
        choices=state.get("choices", []), # Return choices
        evaluations=state.get("evaluations", []),
        round_summary=state.get("round_summary"),
        evaluations_log=state.get("evaluations_log", []),
        # Judgment analysis data
        analysis_result=state.get("analysis_result"),
        user_verdict=state.get("user_verdict"),
        user_sentence_text=state.get("user_sentence_text"),
        user_reasoning=state.get("user_reasoning"),
        actual_judgment=state.get("actual_judgment")
    )
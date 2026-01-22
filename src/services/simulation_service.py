import asyncio
import json
from typing import Dict, Any, List, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from sqlalchemy import select

from src.agents.common.state import CourtSimulationState, Role
from src.database.connection import SessionLocal
from src.database.models.case import Case
from src.utils.history_manager import HistoryManager
from src.utils.logger import get_logger
from src.config.settings import settings

# Agents
from src.agents.prosecutor.prosecutor_agent import ProsecutorAgent
from src.agents.defense.defense_agent import DefenseAgent
from src.agents.legal_advisor.legal_advisor_agent import LegalAdvisorAgent
from src.agents.judge.judge_agent import JudgeAgent
from src.agents.judge.judgment_analyzer import JudgmentAnalyzer

logger = get_logger(__name__)

class SimulationService:
    """
    Simulation Service Layer
    
    Responsibility:
    - Orchestrate calls between Agents and Infrastructure (DB, History).
    - Handle Concurrency: Wrap blocking synchronous calls (DB, RAG) in threads.
    - Decouple Workflow (Graph) from Implementation Details.
    """
    
    def __init__(self):
        # Initialize Agents
        # NOTE: If agent initialization performs blocking IO (e.g. Pinecone connection),
        # it might slow down startup. Ideally, this should also be deferred or async.
        self.prosecutor = ProsecutorAgent()
        self.defense = DefenseAgent()
        self.advisor = LegalAdvisorAgent()
        self.judge = JudgeAgent()
        self.analyzer = JudgmentAnalyzer()

    # --- Helper: Async Wrappers for Blocking Calls ---
    
    async def _fetch_case_from_db(self, case_id: int) -> Optional[Dict[str, Any]]:
        """Fetch case data from DB in a separate thread to avoid blocking event loop."""
        def _fetch():
            session = SessionLocal()
            try:
                query = select(Case).where(Case.id == case_id)
                result = session.execute(query)
                case_record = result.scalar_one_or_none()
                if case_record:
                    return {
                        "id": case_record.id,
                        "case_number": case_record.case_number,
                        "actual_label": case_record.actual_label,
                        "actual_rule": case_record.actual_rule,
                        "actual_reason": case_record.actual_reason,
                        "facts": case_record.facts,
                        "description": case_record.description,
                        "sentencing_factors": case_record.sentencing_factors
                    }
                return None
            except Exception as e:
                logger.error(f"DB Fetch Error: {e}")
                return None
            finally:
                session.close()
        
        return await asyncio.to_thread(_fetch)

    async def _save_history(self, session_id: str, role: str, content: str, turn_count: int, phase: str, metadata: dict, case_id: Optional[int]):
        """Save conversation history in a separate thread."""
        def _save():
            try:
                with HistoryManager(session_id) as history:
                    history.save_turn(
                        role=role, content=content, turn_count=turn_count, 
                        phase=phase, metadata=metadata, case_id=case_id
                    )
            except Exception as e:
                logger.error(f"Failed to save history: {e}")
        
        await asyncio.to_thread(_save)

    # --- Core Logic for Workflow Nodes ---

    async def run_briefing(self, state: CourtSimulationState, config: RunnableConfig) -> dict:
        """Logic for 'briefing' node."""
        session_id = config["configurable"].get("thread_id", "default_session")

        # 1. Legal Advisor Context (Async)
        # Pass session_id for RAGAS evaluation
        legal_context_result = await self.advisor.update_legal_context(state, session_id=session_id)
        legal_context = legal_context_result.get("legal_context", {})

        # 2. DB Fetch (Blocking -> Wrapped)
        case_id = state.get("case_id")
        case_data = await self._fetch_case_from_db(case_id) if case_id else None

        case_number = case_data.get("case_number", state.get("case_number", "Unknown")) if case_data else state.get("case_number", "Unknown")
        facts = case_data.get("facts", "") if case_data else ""
        description = case_data.get("description", "") if case_data else ""
        sentencing_factors = case_data.get("sentencing_factors", {}) if case_data else {}

        actual_judgment = {}
        if case_data:
            actual_judgment = {
                "actual_label": case_data["actual_label"],
                "actual_rule": case_data["actual_rule"],
                "actual_reason": case_data["actual_reason"]
            }

        # 3. Initial Message with Sentencing Factors
        sentencing_info_html = "정보 없음"
        
        if case_data and facts:
            message_content = f"재판을 개정합니다.\n사건 개요: {description}\n사건 설명: {facts}\n"

            # 양형 인자 추가
            if sentencing_factors:
                valid_factors_msg = [] # For Dialogue (Text)
                valid_factors_html = [] # For Popup (HTML)
                
                for category, details in sentencing_factors.items():
                    if isinstance(details, dict):
                        for factor_name, value in details.items():
                            if value is not None:
                                valid_factors_msg.append(f"{factor_name}: {value}")
                                valid_factors_html.append(f"<li><b>{factor_name}</b>: {value}</li>")
                    elif details is not None:
                        valid_factors_msg.append(f"{category}: {details}")
                        valid_factors_html.append(f"<li><b>{category}</b>: {details}</li>")

                if valid_factors_msg:
                    # Add to Dialogue
                    message_content += "\n⚖️ 양형 참고 자료:\n"
                    for factor in valid_factors_msg:
                        message_content += f"  - {factor}\n"
                    
                    # Create HTML for Popup
                    sentencing_info_html = "<h4>⚖️ 양형 참고 자료</h4><ul>" + "".join(valid_factors_html) + "</ul>"
        else:
            message_content = f"재판을 개정합니다.\n\n사건 개요: {state.get('case_summary')}"
        
        agent_output = {
            "role": Role.JUDGE.value,
            "content": message_content,
            "emotion": "neutral",
            "references": []
        }

        # 4. Save History (Blocking -> Wrapped)
        await self._save_history(
            session_id, Role.JUDGE.value, message_content, 0, "briefing", {}, case_id
        )

        return {
            "current_phase": "briefing",
            "turn_count": 0,
            "phase_turn": 0,
            "judge_round": 1,
            "case_id": case_id,
            "case_number": case_number,
            "legal_context": legal_context,
            "sentencing_factors": sentencing_factors,
            "sentencing_info": sentencing_info_html, # Added for Frontend Popup
            "actual_judgment": actual_judgment,
            "evaluations": [],
            "messages": [AIMessage(content=json.dumps(agent_output, ensure_ascii=False))]
        }

    async def run_prosecutor(self, state: CourtSimulationState, config: RunnableConfig) -> dict:
        """Logic for 'prosecutor' node."""
        session_id = config["configurable"].get("thread_id", "default_session")
        
        if settings.debug_graph_execution:
            logger.graph_node("prosecutor", "executing", turn_count=state.get("turn_count"))

        # Agent Generation (Async)
        result = await self.prosecutor.generate_argument(state)
        
        # History Saving (Blocking -> Wrapped)
        try:
            last_msg = result.get("messages", [])[-1]
            content_json = json.loads(last_msg.content)
            
            await self._save_history(
                session_id, 
                Role.PROSECUTOR.value, 
                content_json.get("content", ""),
                state.get("turn_count", 0) + 1, 
                "debate",
                {"emotion": content_json.get("emotion"), "references": content_json.get("references")},
                state.get("case_id")
            )
        except Exception as e:
            logger.error(f"Error processing prosecutor output for history: {e}")

        return result

    async def run_defense(self, state: CourtSimulationState, config: RunnableConfig) -> dict:
        """Logic for 'defense' node."""
        session_id = config["configurable"].get("thread_id", "default_session")
        
        if settings.debug_graph_execution:
            logger.graph_node("defense", "executing", turn_count=state.get("turn_count"))

        # Agent Generation (Async)
        result = await self.defense.generate_argument(state)
        
        # History Saving (Blocking -> Wrapped)
        try:
            last_msg = result.get("messages", [])[-1]
            content_json = json.loads(last_msg.content)
            
            await self._save_history(
                session_id, 
                Role.DEFENSE.value, 
                content_json.get("content", ""),
                state.get("turn_count", 0) + 1, 
                "debate",
                {"emotion": content_json.get("emotion"), "references": content_json.get("references")},
                state.get("case_id")
            )
        except Exception as e:
            logger.error(f"Error processing defense output for history: {e}")

        return result

    async def run_user_judge(self, state: CourtSimulationState) -> dict:
        """Logic for 'user_judge' node."""
        try:
            result = await self.judge.handle_user_judge(state)
            
            # Merge result with state updates that were originally in graph.py
            updated_state_updates = {
                "current_phase": result["current_phase"],
                "phase_turn": result["phase_turn"],
                "choices": result["choices"],
                "evaluations": result["evaluations"],
                "evaluations_log": result["evaluations_log"],
                "round_summary": result["round_summary"],
                "messages": result["messages"],
                "judge_round": state.get("judge_round", 1) + 1,
                "last_user_judge_turn": state.get("turn_count", 0)
            }
            # LangGraph nodes usually return a dict of updates, which are merged into state.
            # However, handle_user_judge might be returning a partial dict or full dict.
            # In graph.py it was returning {**state, ...updates}.
            # LangGraph merges return value into state. We can just return the updates.
            # But let's return the full merged state to be safe and consistent with previous code.
            return {**state, **updated_state_updates}
            
        except Exception as e:
            logger.error(f"[user_judge error] {e}")
            return {
                **state,
                "messages": [AIMessage(content="⚠️ 유저 개입 처리 중 오류가 발생했습니다.")],
                "current_phase": "user_judge",
                "choices": [],
                "evaluations": [],
            }

    async def run_analysis(self, state: CourtSimulationState) -> dict:
        """Logic for 'analysis' node."""
        if settings.debug_graph_execution:
            logger.graph_node("analysis", "analyzing")
        return await self.analyzer.analyze_and_feedback(state)

# Singleton Instance
simulation_service = SimulationService()

"""LangGraph Workflow Definition - Dynamic Router 패턴 멀티에이전트 워크플로우"""
import json
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from src.agents.common.state import CourtSimulationState
from src.config.settings import settings
from src.utils.llm_factory import create_llm
from src.prompts.router_prompts import ROUTER_SYSTEM_PROMPT, ROUTER_HUMAN_TEMPLATE
from src.database.checkpoint_manager import CheckpointManager
from src.utils.logger import get_logger
from src.utils.langsmith_integration import create_agent_tracer

# Service Layer Import
from src.services.simulation_service import simulation_service

logger = get_logger(__name__)


class RouterDecision(BaseModel):
    """
    Router 결정 구조화
    """
    next_speaker: Literal["prosecutor", "defense", "user_judge", "end"] = Field(
        description="다음 발화자 선택"
    )
    reasoning: str = Field(description="라우팅 결정 이유")


class CourtSimulationGraph:
    """
    법정 시뮬레이션 LangGraph 워크플로우 (2025 Standard)

    Refactored to use SimulationService for infrastructure and agent interactions.
    """

    def __init__(self):
        self.router_llm = self._setup_router()
        self.checkpoint_manager = CheckpointManager()
        self.checkpointer = self.checkpoint_manager.setup_checkpointer()
        
        # Agents are now managed by simulation_service
        
        self.graph = self._build_graph()
        self.compiled_graph = None

    def _setup_router(self):
        """Router LLM 설정"""
        base_llm = create_llm(temperature=0.1)
        return base_llm.with_structured_output(schema=RouterDecision)

    def _build_graph(self) -> StateGraph:
        """
        LangGraph 워크플로우 구성
        """
        workflow = StateGraph(CourtSimulationState)

        # 노드 정의
        workflow.add_node("briefing", self.node_briefing)
        workflow.add_node("router", self.node_router)
        workflow.add_node("prosecutor", self.node_prosecutor)
        workflow.add_node("defense", self.node_defense)
        workflow.add_node("user_judge", self.node_user_judge)
        workflow.add_node("analysis", self.node_analysis)

        # 엣지 연결
        workflow.set_entry_point("briefing")

        # Judge 브리핑 표시 후 router로 (interrupt_after로 유저가 확인 가능)
        workflow.add_edge("briefing", "router")

        # 메인 debate loop: prosecutor ↔ defense
        workflow.add_conditional_edges(
            "router",
            self.edge_router_decision,
            {
                "prosecutor": "prosecutor",
                "defense": "defense",
                "user_judge": "user_judge",
                "analysis": "analysis",  # Direct route to analysis when judgment is made
                "end": END
            }
        )

        # 각 에이전트 발언 후 다시 라우터로
        workflow.add_edge("prosecutor", "router")
        workflow.add_edge("defense", "router")

        workflow.add_conditional_edges(
            "user_judge",
            self.edge_after_user_judge,
            {
                "router": "router",     # 질문 입력 → 대화 계속
                "analysis": "analysis"  # 판결 선고 → 분석
            }
        )

        # 판결 분석 후 종료
        workflow.add_edge("analysis", END)

        return workflow

    # ========== Node Implementations ========== 

    async def node_briefing(self, state: CourtSimulationState, config: RunnableConfig) -> dict:
        """[Briefing] 사건 초기화"""
        return await simulation_service.run_briefing(state, config)

    async def node_router(self, state: CourtSimulationState) -> dict:
        """[Router] 다음 발화자 결정"""
        # Router logic kept here for workflow visibility
        turn_count = state.get("turn_count", 0)
        phase_turn = state.get("phase_turn", 0)
        current_phase = state.get("current_phase", "debate")

        # SAFETY CHECK: 판결 단계면 바로 분석으로 이동 (user_judge 스킵)
        if current_phase in ["judgment", "result"]:
            if settings.debug_graph_execution:
                logger.graph_edge("router", "analysis", decision=f"terminal_phase_{current_phase}_direct_to_analysis")
            return {
                "next_speaker": "analysis",  # Skip user_judge, go directly to analysis
                "turn_count": turn_count + 1,
                "phase_turn": phase_turn
                # current_phase를 덮어쓰지 않음!
            }

        # 1. 초기 2턴 강제
        if phase_turn == 0:
            if settings.debug_graph_execution:
                logger.graph_edge("router", "prosecutor", decision="forced_initial_turn_0")
            return {
                "next_speaker": "prosecutor",
                "turn_count": turn_count + 1,
                "phase_turn": 1,
                "current_phase": "debate"
            }
        elif phase_turn == 1:
            if settings.debug_graph_execution:
                logger.graph_edge("router", "defense", decision="forced_initial_turn_1")
            return {
                "next_speaker": "defense",
                "turn_count": turn_count + 1,
                "phase_turn": 2,
                "current_phase": "debate"
            }

        # 2. 무한 반복 방지
        if phase_turn > 5:
            if settings.debug_graph_execution:
                logger.graph_edge("router", "user_judge", decision=f"phase_turn_{phase_turn}_exceeded_max")
            return {
                "next_speaker": "user_judge",
                "turn_count": turn_count + 1,
                "phase_turn": phase_turn,
                "current_phase": "debate"
            }

        # 3. LLM 자유 판단
        recent_messages = []
        for msg in state["messages"][-5:]:
            content = msg.content
            if msg.type == 'human':
                recent_messages.append(f"Judge(User): {content}")
                continue
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "role" in data:
                    recent_messages.append(f"{data['role']}: {data['content'][:50]}...")
            except:
                recent_messages.append(f"System: {content[:50]}...")

        chain = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", ROUTER_HUMAN_TEMPLATE)
        ]) | self.router_llm

        tracer = create_agent_tracer("router", state)
        langsmith_config = {
            "metadata": tracer.get_metadata(),
            "tags": tracer.get_tags(action="route_decision")
        }

        # Router LLM 호출 with Exception Handling
        try:
            decision = await chain.ainvoke({
                "current_phase": state['current_phase'],
                "turn_count": turn_count,
                "history": recent_messages
            },
            config=langsmith_config
            )
        except Exception as e:
            logger.error(f"Router decision failed: {e}", exc_info=True)
            # Fallback: 안전한 기본값 - 유저에게 제어권 넘김
            decision = RouterDecision(
                next_speaker="user_judge",
                reasoning=f"Router error, yielding control to user: {str(e)[:100]}"
            )
            if settings.debug_graph_execution:
                logger.graph_edge("router", "user_judge", decision="router_error_fallback")

        if settings.debug_graph_execution:
            logger.graph_edge("router", decision.next_speaker,
                            decision=f"llm_decided_{decision.reasoning[:30]}")

        return {
            "next_speaker": decision.next_speaker,
            "turn_count": turn_count + 1,
            "phase_turn": phase_turn + 1,
            "current_phase": "debate"
        }

    async def node_prosecutor(self, state: CourtSimulationState, config: RunnableConfig) -> dict:
        """[Prosecutor] 검사 발언"""
        return await simulation_service.run_prosecutor(state, config)

    async def node_defense(self, state: CourtSimulationState, config: RunnableConfig) -> dict:
        """[Defense] 변호사 발언"""
        return await simulation_service.run_defense(state, config)

    async def node_user_judge(self, state: CourtSimulationState, config: RunnableConfig) -> dict:
        """[User Judge] 유저 개입 및 평가"""
        return await simulation_service.run_user_judge(state)

    async def node_analysis(self, state: CourtSimulationState) -> dict:
        """[Analysis] 판결 분석"""
        return await simulation_service.run_analysis(state)

    # ========== Edge Functions ==========

    def edge_router_decision(self, state: CourtSimulationState) -> str:
        return state["next_speaker"]

    def edge_after_user_judge(self, state: CourtSimulationState) -> str:
        current_phase = state.get("current_phase", "debate")
        if current_phase in ["judgment", "result"]:
            if settings.debug_graph_execution:
                logger.graph_edge("user_judge", "analysis", decision=f"phase={current_phase}")
            return "analysis"
        else:
            if settings.debug_graph_execution:
                logger.graph_edge("user_judge", "router", decision=f"phase={current_phase}_continue_debate")
            return "router"

    # ========== Compilation ========== 

    def compile(self, enable_interrupts: bool = True):
        interrupt_after_nodes = ["briefing", "prosecutor", "defense", "user_judge"] if enable_interrupts else []
        interrupt_before_nodes = [] if enable_interrupts else []

        self.compiled_graph = self.graph.compile(
            checkpointer=self.checkpointer,
            interrupt_after=interrupt_after_nodes,
            interrupt_before=interrupt_before_nodes
        )
        return self.compiled_graph

    async def setup(self):
        """비동기 리소스 초기화"""
        await self.checkpoint_manager.setup()

    async def close(self):
        """리소스 정리"""
        await self.checkpoint_manager.close()
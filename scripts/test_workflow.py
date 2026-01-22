"""
LangGraph Workflow Test Script
법정 시뮬레이션 전체 워크플로우를 테스트합니다.
"""
import phoenix as px
import asyncio
import sys
import os
import json
import platform

# Windows에서 psycopg 비동기 작업을 위한 이벤트 루프 정책 설정
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.phoenix_monitoring import setup_phoenix
from src.agents.common.graph import CourtSimulationGraph
from src.agents.common.state import CourtSimulationState, Role, AgentOutput
from langchain_core.messages import HumanMessage

async def run_test():
    print("=== [Test] Court Simulation Workflow Start ===")

    # 1. Phoenix 설정 (Arize Key가 있으면 클라우드로, 없으면 로컬 UI 뜸)
    setup_phoenix(project_name="bupblessyou")
    
    # 2. 그래프 초기화
    print("1. Initializing Graph...")
    court_graph = CourtSimulationGraph()

    # Force MemorySaver for Test/Demo stability (avoids DB state mismatch errors like 'pending_sends')
    from langgraph.checkpoint.memory import MemorySaver
    print("[Info] Forcing In-Memory Checkpointer for Test Script stability.")
    court_graph.checkpointer = MemorySaver()
    
    # 비동기 리소스 셋업 (DB 테이블 생성 등) - MemorySaver 사용시 pass됨
    await court_graph.setup()
    
    graph = court_graph.compile(enable_interrupts=True)
    
    # 3. 초기 상태 설정 (가짜 사건 데이터)
    initial_state = {
        "case_id": 20250101,
        "case_summary": "피고인은 2024년 12월경 피해자에게 '고수익 코인 투자'를 미끼로 접근하여 1억 원을 편취하였다. 피고인은 받은 돈을 도박 자금으로 탕진하였다.",
        "case_attributes": [
            {"key": "crime_type", "value": "fraud", "description": "사기죄"},
            {"key": "damage_amount", "value": "100,000,000", "description": "피해액 1억원"},
            {"key": "recovery", "value": "none", "description": "피해 회복되지 않음"},
            {"key": "criminal_history", "value": "first_offense", "description": "초범"}
        ],
        "messages": [],
        "turn_count": 0,
        "current_phase": "briefing",
        "legal_context": {},
        "actual_judgment": {
            "verdict": "징역 1년 6월",
            "reason": "피해액이 크고 회복되지 않았으나 초범인 점 참작"
        }
    }
    
    config = {"configurable": {"thread_id": "test_session_async_1"}}

    # 4. 그래프 실행 Loop
    print("\n2. Starting Execution Loop...")
    
    # 무한 루프 방지용 최대 스텝
    MAX_STEPS = 15
    step = 0
    
    current_state = initial_state
    
    try:
        while step < MAX_STEPS:
            step += 1
            print(f"\n--- Step {step} ---")
            
            # 그래프 실행 (중단점까지)
            # stream()은 제너레이터이므로 list로 변환하여 실행 완료 대기
            events = []
            async for event in graph.astream(current_state, config):
                events.append(event)
                for key, value in event.items():
                    print(f"Node Finished: {key}")
                    if "messages" in value:
                        last_msg = value["messages"][-1]
                        try:
                            content = json.loads(last_msg.content)
                            print(f"   -> [{content.get('role', 'System')}] {content.get('content')}")
                        except:
                            print(f"   -> [System] {last_msg.content}")

            # 현재 상태 조회 (Snapshot)
            snapshot = await graph.aget_state(config)
            next_step = snapshot.next
            
            print(f"Status: {next_step}")

            if not next_step:
                print("=== Workflow Finished ===")
                break
                
            # 5. Human-in-the-Loop 처리
            # Check current_phase to detect if we are pausing AFTER user_judge (interrupt_after)
            current_phase = snapshot.values.get("current_phase")
            
            if "router" in next_step and current_phase != "user_judge":
                # Router 단계에서 멈춘 경우 -> 그냥 재개 (자동 진행)
                print(">> Resuming from Router...")
                current_state = None # None을 주면 snapshot 상태에서 계속 진행
                
            elif current_phase == "user_judge":
                # 판결 단계에서 멈춘 경우 -> 유저 입력 시뮬레이션
                print("\n!!! [User Input Required] Judge Phase !!!")
                
                # Test Scenario 1: Ask a question (to break the loop)
                # Test Scenario 2: Final Verdict (to end the case)
                
                # Let's alternate or just pick one. For this test, let's ask a question first, then verdict later?
                # Simple logic: if turn < 10, ask question. Else verdict.
                
                current_turn = snapshot.values.get("turn_count", 0)
                
                if current_turn < 8:
                    print(">> User asks a question (continuing debate)")
                    user_question = "피해자와의 합의 진행 상황은 정확히 어떻습니까?"
                    
                    # Inject User Message
                    # Note: We need to ensure the Router sees this.
                    # The graph expects messages.
                    from langchain_core.messages import HumanMessage
                    
                    msg = HumanMessage(content=json.dumps({
                        "role": "judge",
                        "content": user_question,
                        "emotion": "serious"
                    }, ensure_ascii=False))
                    
                    await graph.aupdate_state(config, {
                        "messages": [msg],
                        "current_phase": "debate" # Explicitly set to debate
                    })
                    print(">> Resuming with Question...")
                    
                else:
                    print(">> User enters verdict (ending case)")
                    user_verdict = "징역 2년"
                    user_reasoning = "죄질이 나쁘고 피해 회복이 안됨."
                    
                    await graph.aupdate_state(config, {
                        "user_verdict": user_verdict,
                        "user_reasoning": user_reasoning,
                        "current_phase": "judgment" # Move to judgment phase to trigger analysis
                    })
                    print(">> Resuming for Analysis...")
                
                current_state = None
                
            else:
                # 그 외의 경우 (보통 발생 안함)
                print(f">> Continuing execution... {next_step}")
                current_state = None
                
    finally:
        # 리소스 정리
        await court_graph.close()

if __name__ == "__main__":
    
    asyncio.run(run_test())
import streamlit as st
import asyncio
import json
import os
import sys

# Add src to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.common.graph import CourtSimulationGraph
from src.agents.common.state import Role

# Page Config
st.set_page_config(page_title="BupBlessYou Orchestration", page_icon="⚖️", layout="wide")

st.title("⚖️ BupBlessYou Orchestration Demo")

# --- Helper Functions ---
def get_thread_id():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "demo_session_v1"
    return st.session_state.thread_id

@st.cache_resource
def get_court_graph():
    """Initialize the graph once."""
    return CourtSimulationGraph()

# --- Async Utils ---
# Streamlit runs in a loop, handling async needs care.
# We will use a simple asyncio.run for each action for this demo.

async def setup_graph_async(court_graph):
    await court_graph.setup()

async def get_state_async(graph, config):
    return await graph.aget_state(config)

async def run_graph_async(graph, input_state, config):
    events = []
    async for event in graph.astream(input_state, config):
        events.append(event)
    return events

async def update_state_async(graph, config, update_dict):
    await graph.aupdate_state(config, update_dict)

# --- Initialization ---
court_graph = get_court_graph()

# Ensure setup is called (lazy init via session state check)
if "graph_setup_done" not in st.session_state:
    with st.spinner("Initializing System..."):
        asyncio.run(setup_graph_async(court_graph))
    st.session_state.graph_setup_done = True

# Compile graph
graph = court_graph.compile(enable_interrupts=True)
config = {"configurable": {"thread_id": get_thread_id()}}

# --- Sidebar ---
with st.sidebar:
    st.header("Simulation Controls")
    if st.button("Reset Simulation", type="primary"):
        st.session_state.thread_id = f"demo_session_{st.session_state.get('reset_count', 0) + 1}"
        st.session_state.reset_count = st.session_state.get('reset_count', 0) + 1
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Current Session:**")
    st.code(st.session_state.thread_id)

# --- Main Logic ---

# 1. Fetch Current State
try:
    snapshot = asyncio.run(get_state_async(graph, config))
except Exception as e:
    st.error(f"Error fetching state: {e}")
    st.stop()

# 2. Display History
# The graph state stores messages. We can retrieve them from snapshot.
if snapshot.values and "messages" in snapshot.values:
    messages = snapshot.values["messages"]
    for msg in messages:
        # Try to parse JSON content if possible
        try:
            content_json = json.loads(msg.content)
            role = content_json.get("role", "system")
            text = content_json.get("content", "")
            emotion = content_json.get("emotion", "neutral")
        except:
            # Fallback for system messages or unformatted text
            role = "assistant" if msg.type == "ai" else "user"
            text = msg.content
            emotion = "neutral"

        # Map roles to avatars or names
        role_map = {
            "judge": "⚖️",
            "prosecutor": "🦅",
            "defense": "🛡️",
            "user_judge": "🧑‍⚖️"
        }
        avatar = role_map.get(role, "💬")

        with st.chat_message(role, avatar=avatar):
            st.markdown(f"**{role.upper()}**: {text}")
            if emotion != "neutral":
                st.caption(f"*Emotion: {emotion}*")

else:
    st.info("No history yet. Start the simulation.")

# 3. Action Area
st.markdown("---")

# Determine current status
current_values = snapshot.values if snapshot.values else {}
current_phase = current_values.get("current_phase", "start")
next_step = snapshot.next

# Debug info
with st.expander("Debug Info"):
    st.write("Next Step:", next_step)
    st.write("Current Phase:", current_phase)
    # st.json(current_values)

# --- Interaction Logic ---

if not next_step:
    # Start of simulation
    if st.button("Start Court Session", type="primary"):
        initial_state = {
            "case_id": 20250101,
            "case_summary": "피고인은 2024년 12월경 피해자에게 '고수익 코인 투자'를 미끼로 접근하여 1억 원을 편취하였다. 피고인은 받은 돈을 도박 자금으로 탕진하였다.",
            "messages": [],
            "turn_count": 0,
            "current_phase": "briefing",
            "phase_turn": 0,
            "legal_context": {},
            "actual_judgment": {
                "verdict": "징역 1년 6월",
                "reason": "피해액이 크고 회복되지 않았으나 초범인 점 참작"
            }
        }
        with st.spinner("Starting..."):
            asyncio.run(run_graph_async(graph, initial_state, config))
        st.rerun()

elif current_phase == "user_judge":
    # Special Handling for Judge Input
    st.subheader("🧑‍⚖️ Judge's Decision")
    
    # Get choices from state if available
    choices = current_values.get("choices", [])
    
    tab1, tab2 = st.tabs(["Ask Question", "Final Verdict"])
    
    with tab1:
        st.write("Select a question to ask the counsel:")
        # Display buttons for generated questions
        for choice in choices:
            if st.button(f"❓ {choice['label']}", key=choice['id']):
                # Update state with selected question
                update_dict = {
                    "current_phase": "debate", # Return to debate
                    # We might need to add this as a HumanMessage so it appears in history
                    "messages": [json.dumps({
                        "role": "judge",
                        "content": choice['value'],
                        "emotion": "serious"
                    })]
                }
                # But wait, we should probably use update_state to inject the human message
                # The graph expects 'messages' in state update to append.
                
                # We need to construct a HumanMessage that looks like what router expects?
                # The router looks at 'messages'.
                
                # Let's inject it as a HumanMessage (User input)
                # But the graph uses JSON format for consistency usually?
                # Let's look at router: "if msg.type == 'human': Judge(User): {content}"
                
                human_msg = json.dumps({
                        "role": "judge",
                        "content": choice['value'],
                        "emotion": "serious"
                }, ensure_ascii=False)
                
                from langchain_core.messages import HumanMessage
                
                # We update state with the user's message
                asyncio.run(update_state_async(graph, config, {
                    "messages": [HumanMessage(content=human_msg)],
                    "current_phase": "debate" # Explicitly set to continue debate
                }))
                
                # Then resume
                with st.spinner("Processing..."):
                    asyncio.run(run_graph_async(graph, None, config))
                st.rerun()
                
    with tab2:
        st.write("Enter the final verdict:")
        verdict = st.text_input("Verdict (e.g., 징역 1년)")
        reasoning = st.text_area("Reasoning")
        
        if st.button("Pronounce Judgment"):
            if verdict and reasoning:
                asyncio.run(update_state_async(graph, config, {
                    "user_verdict": verdict,
                    "user_reasoning": reasoning,
                    "current_phase": "judgment" # Move to judgment phase
                }))
                
                with st.spinner("Analyzing Judgment..."):
                    asyncio.run(run_graph_async(graph, None, config))
                st.rerun()
            else:
                st.warning("Please fill in both fields.")

else:
    # Standard Step Forward
    # Show who is next or what happened
    
    if st.button("Next Step ➡️", type="primary"):
        with st.spinner("Processing..."):
            asyncio.run(run_graph_async(graph, None, config))
        st.rerun()


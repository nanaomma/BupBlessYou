"""Agent-specific logging utilities for tracking agent state and behavior"""
from typing import Dict, Any, Optional
from functools import wraps
import time
import json

from src.config.settings import settings
from src.utils.logger import get_logger


def log_agent_execution(agent_type: str):
    """Decorator for logging agent execution with timing and state tracking

    Args:
        agent_type: Type of agent (prosecutor, defense, legal_advisor, etc.)

    Usage:
        @log_agent_execution("prosecutor")
        async def generate_argument(self, state):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, state: Dict[str, Any], *args, **kwargs):
            logger = get_logger(func.__module__)

            # Pre-execution logging
            if settings.debug_agent_state:
                logger.agent_state(
                    agent_type=agent_type,
                    state_name="execution_start",
                    state_data={
                        "function": func.__name__,
                        "turn_count": state.get("turn_count"),
                        "phase": state.get("current_phase"),
                        "message_count": len(state.get("messages", []))
                    }
                )

            start_time = time.time()

            try:
                # Execute agent function
                result = await func(self, state, *args, **kwargs)
                duration = time.time() - start_time

                # Post-execution logging
                if settings.debug_agent_state:
                    result_summary = {}
                    if isinstance(result, dict):
                        # Extract key info from result
                        if "messages" in result:
                            new_messages = result["messages"]
                            if new_messages:
                                last_msg = new_messages[-1]
                                try:
                                    content = json.loads(last_msg.content) if hasattr(last_msg, 'content') else {}
                                    result_summary = {
                                        "role": content.get("role"),
                                        "content_length": len(content.get("content", "")),
                                        "emotion": content.get("emotion")
                                    }
                                except:
                                    pass

                    logger.agent_state(
                        agent_type=agent_type,
                        state_name="execution_complete",
                        state_data={
                            "function": func.__name__,
                            "duration_ms": round(duration * 1000, 2),
                            "result_summary": result_summary
                        }
                    )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Agent execution failed: {agent_type}.{func.__name__}",
                    agent_type=agent_type,
                    duration_ms=round(duration * 1000, 2),
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


def log_llm_call(provider: str, model: str):
    """Decorator for logging LLM API calls

    Args:
        provider: LLM provider (openai, upstage)
        model: Model name

    Usage:
        @log_llm_call("openai", "gpt-4")
        async def call_llm(self, prompt):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not settings.debug_llm_calls:
                return await func(*args, **kwargs)

            logger = get_logger(func.__module__)
            start_time = time.time()

            try:
                # Log call start
                logger.debug(
                    f"LLM call: {provider}/{model}",
                    provider=provider,
                    model=model,
                    function=func.__name__
                )

                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Log call completion
                logger.debug(
                    f"LLM call complete",
                    provider=provider,
                    model=model,
                    duration_ms=round(duration * 1000, 2)
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"LLM call failed: {provider}/{model}",
                    provider=provider,
                    model=model,
                    duration_ms=round(duration * 1000, 2),
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


class AgentStateTracker:
    """Context manager for tracking agent state transitions"""

    def __init__(self, agent_type: str, state_name: str, logger_name: str = None):
        self.agent_type = agent_type
        self.state_name = state_name
        self.logger = get_logger(logger_name or __name__)
        self.start_time = None
        self.state_data = {}

    def __enter__(self):
        """Enter state tracking context"""
        if settings.debug_agent_state:
            self.start_time = time.time()
            self.logger.agent_state(
                agent_type=self.agent_type,
                state_name=f"{self.state_name}_start",
                state_data=self.state_data
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit state tracking context"""
        if settings.debug_agent_state:
            duration = time.time() - self.start_time if self.start_time else 0
            status = "failed" if exc_type else "complete"

            self.logger.agent_state(
                agent_type=self.agent_type,
                state_name=f"{self.state_name}_{status}",
                state_data={
                    **self.state_data,
                    "duration_ms": round(duration * 1000, 2),
                    "error": str(exc_val) if exc_val else None
                }
            )

        return False  # Don't suppress exceptions

    def update_state(self, **kwargs):
        """Update state data during tracking"""
        self.state_data.update(kwargs)


# Usage example for context manager:
# with AgentStateTracker("prosecutor", "argument_generation", __name__) as tracker:
#     tracker.update_state(turn=5, phase="debate")
#     result = await generate_argument()

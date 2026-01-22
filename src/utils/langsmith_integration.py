"""LangSmith integration for LLM tracing and monitoring"""
import os
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_langsmith():
    """
    Configure LangSmith tracing environment variables

    This should be called early in application startup (before LLM initialization)
    to enable automatic tracing of all LangChain LLM calls.
    """
    if not settings.langsmith_tracing:
        logger.info("LangSmith tracing disabled")
        return

    if not settings.langsmith_api_key:
        logger.warning("LangSmith tracing enabled but LANGSMITH_API_KEY not set")
        return

    # Set environment variables for LangSmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    logger.info(
        f"LangSmith tracing enabled",
        project=settings.langsmith_project,
        endpoint=settings.langsmith_endpoint
    )


def disable_langsmith():
    """Disable LangSmith tracing temporarily"""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    logger.debug("LangSmith tracing disabled")


def enable_langsmith():
    """Re-enable LangSmith tracing"""
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        logger.debug("LangSmith tracing enabled")


@contextmanager
def langsmith_trace(
    name: str,
    run_type: str = "chain",
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
):
    """
    Context manager for custom LangSmith trace runs

    Args:
        name: Name of the trace run
        run_type: Type of run (chain, llm, tool, retriever, etc.)
        metadata: Additional metadata to attach to the trace
        tags: Tags for filtering and organization

    Usage:
        with langsmith_trace("prosecutor_argument", run_type="agent",
                            metadata={"turn": 3}, tags=["prosecutor"]):
            result = await generate_argument(state)
    """
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        # If tracing is disabled, just execute the code block
        yield None
        return

    try:
        from langsmith import trace as langsmith_trace_decorator
        from langsmith.run_helpers import traceable

        # Create a traceable context
        @traceable(name=name, run_type=run_type, metadata=metadata or {}, tags=tags or [])
        def traced_block():
            return None

        # Execute to create the trace context
        traced_block()

        yield traced_block

    except ImportError:
        logger.warning("langsmith package not installed, tracing skipped")
        yield None
    except Exception as e:
        logger.error(f"LangSmith tracing error: {e}", exc_info=True)
        yield None


def add_langsmith_metadata(
    agent_type: str,
    turn_count: int,
    phase: str,
    session_id: Optional[str] = None,
    **extra_metadata
) -> Dict[str, Any]:
    """
    Create standardized metadata for LangSmith traces

    Args:
        agent_type: Type of agent (prosecutor, defense, etc.)
        turn_count: Current turn number
        phase: Current phase (briefing, debate, judgment, etc.)
        session_id: Session identifier
        **extra_metadata: Additional custom metadata

    Returns:
        Dictionary of metadata for LangSmith
    """
    metadata = {
        "agent_type": agent_type,
        "turn_count": turn_count,
        "phase": phase,
        "environment": settings.environment,
        "llm_provider": settings.default_llm_provider
    }

    if session_id:
        metadata["session_id"] = session_id

    metadata.update(extra_metadata)

    return metadata


def get_langsmith_tags(
    agent_type: str,
    phase: str,
    **custom_tags
) -> List[str]:
    """
    Create standardized tags for LangSmith traces

    Args:
        agent_type: Type of agent
        phase: Current phase
        **custom_tags: Additional custom tags

    Returns:
        List of tags for filtering in LangSmith
    """
    tags = [
        f"agent:{agent_type}",
        f"phase:{phase}",
        f"env:{settings.environment}",
        f"provider:{settings.default_llm_provider}"
    ]

    # Add custom tags
    for key, value in custom_tags.items():
        tags.append(f"{key}:{value}")

    return tags


class LangSmithCallbackHandler:
    """
    Custom callback handler for LangSmith integration with additional context

    This can be used as a callback in LangChain calls to provide additional
    context and metadata to LangSmith traces.
    """

    def __init__(
        self,
        agent_type: str,
        turn_count: int,
        phase: str,
        session_id: Optional[str] = None
    ):
        self.agent_type = agent_type
        self.turn_count = turn_count
        self.phase = phase
        self.session_id = session_id

    def get_metadata(self, **extra) -> Dict[str, Any]:
        """Get metadata for the current context"""
        return add_langsmith_metadata(
            agent_type=self.agent_type,
            turn_count=self.turn_count,
            phase=self.phase,
            session_id=self.session_id,
            **extra
        )

    def get_tags(self, **custom) -> List[str]:
        """Get tags for the current context"""
        return get_langsmith_tags(
            agent_type=self.agent_type,
            phase=self.phase,
            **custom
        )


def log_langsmith_url(run_id: Optional[str] = None):
    """
    Log the LangSmith trace URL for easy access

    Args:
        run_id: Optional run ID to construct direct URL
    """
    if not settings.langsmith_tracing:
        return

    base_url = f"https://smith.langchain.com/o/-/projects/p/{settings.langsmith_project}"

    if run_id:
        url = f"{base_url}/r/{run_id}"
        logger.info(f"LangSmith trace: {url}")
    else:
        logger.info(f"LangSmith project: {base_url}")


# Convenience function for agent developers
def create_agent_tracer(agent_type: str, state: Dict[str, Any]):
    """
    Create a LangSmith callback handler for an agent

    Args:
        agent_type: Type of agent (prosecutor, defense, etc.)
        state: Current agent state

    Returns:
        LangSmithCallbackHandler instance

    Usage:
        tracer = create_agent_tracer("prosecutor", state)
        result = await llm.ainvoke(prompt, metadata=tracer.get_metadata())
    """
    return LangSmithCallbackHandler(
        agent_type=agent_type,
        turn_count=state.get("turn_count", 0),
        phase=state.get("current_phase", "unknown"),
        session_id=state.get("session_id")
    )

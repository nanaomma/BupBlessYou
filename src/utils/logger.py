"""Centralized logging configuration with debug support"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json

from src.config.settings import settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON support"""

    def __init__(self, include_json: bool = False):
        super().__init__()
        self.include_json = include_json

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structure"""
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Base log format
        if self.include_json:
            log_data = {
                "timestamp": timestamp,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }

            # Add extra context if available
            if hasattr(record, 'session_id'):
                log_data['session_id'] = record.session_id
            if hasattr(record, 'agent_type'):
                log_data['agent_type'] = record.agent_type
            if hasattr(record, 'state_data'):
                log_data['state_data'] = record.state_data
            if hasattr(record, 'exc_info') and record.exc_info:
                log_data['exception'] = self.formatException(record.exc_info)

            return json.dumps(log_data, ensure_ascii=False)
        else:
            # Human-readable format
            prefix = f"[{timestamp}] [{record.levelname}] [{record.name}]"
            message = record.getMessage()

            # Add context markers
            context = []
            if hasattr(record, 'session_id'):
                context.append(f"session={record.session_id[:8]}")
            if hasattr(record, 'agent_type'):
                context.append(f"agent={record.agent_type}")

            context_str = f" ({', '.join(context)})" if context else ""

            result = f"{prefix}{context_str} {message}"

            if record.exc_info:
                result += "\n" + self.formatException(record.exc_info)

            return result


class DebugLogger:
    """Debug-aware logger wrapper with context support"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context = {}

    def set_context(self, **kwargs):
        """Set persistent context for this logger instance"""
        self._context.update(kwargs)

    def clear_context(self):
        """Clear all context"""
        self._context.clear()

    def _log(self, level: int, msg: str, **kwargs):
        """Internal log method with context injection"""
        # exc_info는 예약된 파라미터이므로 extra에서 분리
        exc_info = kwargs.pop('exc_info', False)
        extra = {**self._context, **kwargs}
        self.logger.log(level, msg, extra=extra, exc_info=exc_info)

    def debug(self, msg: str, **kwargs):
        """Debug level log (only in debug mode)"""
        if settings.debug:
            self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        """Info level log"""
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        """Warning level log"""
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        """Error level log"""
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        """Critical level log"""
        self._log(logging.CRITICAL, msg, **kwargs)

    # Agent-specific helpers
    def agent_state(self, agent_type: str, state_name: str, state_data: dict):
        """Log agent state changes"""
        self.debug(
            f"Agent state: {state_name}",
            agent_type=agent_type,
            state_data=state_data
        )

    def graph_node(self, node_name: str, action: str, **kwargs):
        """Log graph node execution"""
        self.debug(
            f"Graph node [{node_name}]: {action}",
            node_name=node_name,
            **kwargs
        )

    def graph_edge(self, from_node: str, to_node: str, decision: Optional[str] = None):
        """Log graph edge traversal"""
        msg = f"Graph edge: {from_node} -> {to_node}"
        if decision:
            msg += f" (reason: {decision})"
        self.debug(msg, from_node=from_node, to_node=to_node, decision=decision)


def setup_logging():
    """Configure logging based on settings"""

    # Determine log level
    log_level_str = settings.log_level.upper() if hasattr(settings, 'log_level') else 'INFO'
    if settings.debug:
        log_level_str = 'DEBUG'

    log_level = getattr(logging, log_level_str, logging.INFO)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Formatter based on environment
    if settings.environment == "production":
        # JSON format for production (easier for log aggregation)
        formatter = StructuredFormatter(include_json=True)
    else:
        # Human-readable format for development
        formatter = StructuredFormatter(include_json=False)

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler for debug mode
    if settings.debug:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(
            log_dir / f"debug_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter(include_json=False))
        root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langsmith").setLevel(logging.WARNING)
    logging.getLogger("langsmith.client").setLevel(logging.WARNING)

    # Additional noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Keep our application loggers at configured level
    logging.getLogger("src").setLevel(log_level)

    # Log initialization
    init_logger = DebugLogger(__name__)
    init_logger.info(f"Logging configured: level={log_level_str}, debug={settings.debug}, env={settings.environment}")


def get_logger(name: str) -> DebugLogger:
    """Get a debug-aware logger instance

    Args:
        name: Logger name (typically __name__)

    Returns:
        DebugLogger instance with context support
    """
    return DebugLogger(name)

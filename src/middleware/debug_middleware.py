"""Debug middleware for FastAPI request/response logging"""
import time
import json
from typing import Callable
from fastapi import Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DebugLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses in debug mode"""

    async def dispatch(self, request: Request, call_next: Callable):
        """Log request and response details"""

        # Skip logging for static files
        if request.url.path.startswith("/static"):
            return await call_next(request)

        # Generate request ID
        request_id = id(request)

        # Request logging
        if settings.debug:
            logger.debug(
                f"Incoming request: {request.method} {request.url.path}",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                query_params=dict(request.query_params)
            )

            # Log request body for POST/PUT (if JSON)
            if request.method in ["POST", "PUT"]:
                try:
                    body = await request.body()
                    if body:
                        body_json = json.loads(body)
                        logger.debug(
                            "Request body",
                            request_id=request_id,
                            body=body_json
                        )
                        # Restore body for downstream handlers
                        async def receive():
                            return {"type": "http.request", "body": body}
                        request._receive = receive
                except Exception as e:
                    logger.warning(f"Failed to log request body: {e}")

        # Execute request
        start_time = time.time()
        try:
            response = await call_next(request)
            duration = time.time() - start_time

            # Response logging
            if settings.debug:
                logger.debug(
                    f"Response: {response.status_code}",
                    request_id=request_id,
                    status_code=response.status_code,
                    duration_ms=round(duration * 1000, 2)
                )

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {str(e)}",
                request_id=request_id,
                duration_ms=round(duration * 1000, 2),
                exc_info=True
            )
            raise


class SessionDebugRoute(APIRoute):
    """Custom route handler with session state logging"""

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            """Add session state logging"""

            # Execute original handler
            response = await original_route_handler(request)

            # Log session state changes if debug enabled
            if settings.debug_agent_state and request.url.path.startswith("/api/"):
                try:
                    # Extract session info from request if available
                    if hasattr(request.state, 'session_id'):
                        logger.agent_state(
                            agent_type="session",
                            state_name="session_updated",
                            state_data={
                                "session_id": request.state.session_id,
                                "path": request.url.path
                            }
                        )
                except Exception:
                    pass

            return response

        return custom_route_handler

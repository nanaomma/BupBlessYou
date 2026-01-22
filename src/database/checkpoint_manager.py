"""
Checkpoint Manager - DB 연결 및 LangGraph Checkpointer 관리 전담 모듈

관심사의 분리:
- Graph 클래스는 워크플로우 로직에만 집중
- DB 연결/해제/폴백 로직은 이 모듈에서 전담

재사용성:
- API 서버, 데이터 적재 스크립트 등 다른 모듈에서도 재사용 가능

테스트 용이성:
- DB 없이 그래프 테스트 시 MemorySaver로 쉽게 모킹 가능

Windows 호환성:
- psycopg 비동기 작업을 위한 이벤트 루프 정책 설정 필요
- asyncio.WindowsSelectorEventLoopPolicy() 사용
"""

import asyncio
import platform
from typing import Optional
from langgraph.checkpoint.memory import MemorySaver

# Windows 이벤트 루프 정책 설정 (psycopg 비동기 작업 호환)
if platform.system() == 'Windows':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass  # Python 3.7 이하에서는 WindowsSelectorEventLoopPolicy가 없음

# 비동기 Connection Pool 및 PostgresSaver import
try:
    from psycopg_pool import AsyncConnectionPool
except ImportError:
    AsyncConnectionPool = None

try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
except ImportError:
    AsyncPostgresSaver = None

from src.config.settings import settings


class CheckpointManager:
    """
    LangGraph Checkpointer 및 DB 연결 관리자

    역할:
    1. PostgreSQL 연결 설정 및 Connection Pool 생성
    2. AsyncPostgresSaver 초기화
    3. 연결 실패 시 MemorySaver로 자동 폴백
    4. 리소스 정리 (close)
    """

    def __init__(self):
        self.connection_pool: Optional[AsyncConnectionPool] = None
        self.checkpointer = None

    def setup_checkpointer(self):
        """
        Checkpointer 설정 (동기 메서드, 초기화 시 호출)

        Returns:
            MemorySaver 또는 AsyncPostgresSaver 인스턴스
        """
        # 1. 필수 라이브러리 확인
        if not AsyncConnectionPool:
            print("[Info] AsyncConnectionPool not available. Using In-Memory Checkpointer.")
            print("      Install: pip install 'psycopg[binary,pool]'")
            self.checkpointer = MemorySaver()
            return self.checkpointer

        if not AsyncPostgresSaver:
            print("[Info] AsyncPostgresSaver not available. Using In-Memory Checkpointer.")
            print("      Install: pip install --upgrade langgraph")
            self.checkpointer = MemorySaver()
            return self.checkpointer

        # 2. DB 자격증명 확인
        if not all([settings.db_host, settings.db_user, settings.db_password, settings.db_name]):
            print("[Warning] DB credentials incomplete. Using In-Memory Checkpointer.")
            print(f"         Missing: {[k for k, v in [('DB_HOST', settings.db_host), ('DB_USER', settings.db_user), ('DB_PASSWORD', settings.db_password), ('DB_NAME', settings.db_name)] if not v]}")
            self.checkpointer = MemorySaver()
            return self.checkpointer

        try:
            # 3. Connection string 생성 (개별 파라미터 방식)
            conninfo = (
                f"host={settings.db_host} "
                f"port={settings.db_port} "
                f"dbname={settings.db_name} "
                f"user={settings.db_user} "
                f"password={settings.db_password}"
            )

            # 4. Async Connection Pool 생성 (open=False로 지연 초기화)
            self.connection_pool = AsyncConnectionPool(
                conninfo=conninfo,
                min_size=1,
                max_size=5,
                open=False,  # 비동기 컨텍스트에서 open() 호출 예정
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,  # Prepared statement 비활성화 (호환성)
                }
            )

            # 5. AsyncPostgresSaver 초기화
            self.checkpointer = AsyncPostgresSaver(self.connection_pool)

            print(f"[Info] PostgreSQL Checkpointer configured: {settings.db_host}:{settings.db_port}/{settings.db_name}")
            print("[Info] Connection Pool will open during setup()")
            return self.checkpointer

        except Exception as e:
            print(f"[Error] Failed to configure PostgreSQL Checkpointer: {e}")
            print("[Info] Falling back to In-Memory Checkpointer")
            self.connection_pool = None
            self.checkpointer = MemorySaver()
            return self.checkpointer

    async def setup(self):
        """
        비동기 리소스 초기화 (Connection Pool 열기 + 테이블 생성)

        사용:
            manager = CheckpointManager()
            checkpointer = manager.setup_checkpointer()
            await manager.setup()  # Connection Pool 열고 DB 테이블 생성
        """
        # MemorySaver 사용 중이면 setup 불필요
        if self.is_using_memory():
            print("[Info] Using In-Memory Checkpointer (no DB setup needed)")
            return

        # PostgreSQL Checkpointer 사용 중
        try:
            # 1. Connection Pool 열기
            if self.connection_pool:
                # Pool 상태 확인 (closed 속성 체크)
                if hasattr(self.connection_pool, 'closed') and not self.connection_pool.closed:
                    print("[Info] Connection Pool already open, skipping open()")
                else:
                    print("[Info] Opening Connection Pool...")
                    await self.connection_pool.open()
                    print("[Info] ✓ Connection Pool opened successfully")

                # 2. Checkpointer 테이블 생성
                if self.checkpointer and hasattr(self.checkpointer, "setup"):
                    print("[Info] Creating Checkpointer tables...")
                    await self.checkpointer.setup()
                    print("[Info] ✓ Checkpointer tables ready")

        except Exception as e:
            print(f"[Error] PostgreSQL setup failed: {e}")
            print("[Warning] Workflow will continue with In-Memory Checkpointer")
            print("[Warning] State will NOT persist between runs")

            # Pool 정리 및 MemorySaver로 폴백
            if self.connection_pool:
                try:
                    await self.connection_pool.close()
                except Exception:
                    pass  # 이미 닫혔거나 닫는 중 발생한 에러 무시
            self.connection_pool = None
            self.checkpointer = MemorySaver()

    async def close(self):
        """
        리소스 정리 (Connection Pool 닫기)

        사용:
            await manager.close()
        """
        if self.is_using_memory():
            print("[Info] In-Memory Checkpointer (no cleanup needed)")
            return

        if self.connection_pool:
            try:
                # Pool이 이미 닫혔는지 확인
                if hasattr(self.connection_pool, 'closed') and self.connection_pool.closed:
                    print("[Info] Connection Pool already closed")
                else:
                    print("[Info] Closing Connection Pool...")
                    await self.connection_pool.close()
                    print("[Info] ✓ Connection Pool closed successfully")
            except Exception as e:
                print(f"[Warning] Error closing Connection Pool: {e}")

    def is_using_memory(self) -> bool:
        """
        현재 MemorySaver를 사용하는지 확인

        Returns:
            bool: MemorySaver 사용 여부
        """
        return isinstance(self.checkpointer, MemorySaver)

    def is_using_postgres(self) -> bool:
        """
        현재 PostgreSQL Checkpointer를 사용하는지 확인

        Returns:
            bool: PostgreSQL Checkpointer 사용 여부
        """
        return self.checkpointer is not None and not self.is_using_memory()

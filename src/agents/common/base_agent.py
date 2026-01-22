"""Base Agent Class for Legal AI System"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class AgentMessage:
    """에이전트 메시지 표준 포맷"""
    role: str  # 'judge', 'prosecutor', 'lawyer', 'legal_advisor'
    content: str  # JSON 문자열 또는 일반 텍스트
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata or {}
        }


class BaseAgent(ABC):
    """모든 에이전트의 기본 클래스"""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    @abstractmethod
    async def generate_response(
        self,
        case_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> AgentMessage:
        """
        에이전트 응답 생성

        Args:
            case_info: 사건 정보
                {
                    "id" or "case_id": "2024고합1234",
                    "crime_type": "살인",
                    "charges": ["형법 제250조"],
                    "facts_summary": "피고인이..."
                }
            context: 추가 컨텍스트

        Returns:
            AgentMessage: 표준화된 응답 메시지
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, role={self.role})"

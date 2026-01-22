"""
History Compressor - 대화 이력 압축 및 정제

변호사/검사 에이전트의 대화 History를 압축하여:
1. 중요한 정보만 유지
2. 반복적인 정보 제거
3. 최근 논점에 집중
"""
import json
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HistoryCompressor:
    """
    대화 History 압축 및 핵심 추출 클래스

    Purpose:
        - LLM이 최근 논점에 집중하도록 History 압축
        - 대화 반복 방지 및 맥락 유지
    """

    @staticmethod
    def compress_history(
        messages: List[BaseMessage],
        max_recent_messages: int = 4,  # 최근 2 라운드 (각 에이전트 1개씩)
        include_initial_brief: bool = True,
        compress_middle: bool = True
    ) -> List[BaseMessage]:
        """
        History 압축 전략

        Strategy:
            1. 초기 브리핑 (첫 메시지) 유지 (선택적)
            2. 중간 대화 요약 (선택적)
            3. 최근 N개 메시지만 상세 유지

        Args:
            messages: 전체 대화 History
            max_recent_messages: 최근 메시지 유지 개수 (기본 4개 = 2 라운드)
            include_initial_brief: 초기 브리핑 포함 여부
            compress_middle: 중간 대화 요약 여부

        Returns:
            압축된 History

        Example:
            Before: [초기] + [20개 중간 메시지] + [최근 4개]
            After:  [초기] + [요약] + [최근 4개]
        """
        if not messages:
            return []

        # 메시지가 적으면 압축 불필요
        if len(messages) <= max_recent_messages + (1 if include_initial_brief else 0):
            return messages

        compressed = []

        # ==========================================
        # 1. 초기 브리핑 유지 (선택적)
        # ==========================================
        start_idx = 0
        if include_initial_brief and messages:
            compressed.append(messages[0])
            start_idx = 1
            logger.debug("History compression: Included initial brief")

        # ==========================================
        # 2. 중간 대화 처리
        # ==========================================
        if len(messages) > max_recent_messages + start_idx:
            middle_messages = messages[start_idx:-max_recent_messages]

            if compress_middle and middle_messages:
                # 중간 대화 요약
                summary = HistoryCompressor._summarize_middle_history(middle_messages)
                summary_message = AIMessage(content=json.dumps({
                    "role": "system",
                    "content": f"[이전 대화 요약]\n{summary}",
                    "emotion": "neutral"
                }, ensure_ascii=False))
                compressed.append(summary_message)
                logger.debug(
                    f"History compression: Summarized {len(middle_messages)} middle messages"
                )
            else:
                # 요약 없이 중간 메시지 포함 (압축 없음)
                compressed.extend(middle_messages)

        # ==========================================
        # 3. 최근 대화 유지
        # ==========================================
        recent_messages = messages[-max_recent_messages:]
        compressed.extend(recent_messages)
        logger.debug(f"History compression: Kept {len(recent_messages)} recent messages")

        logger.info(
            f"History compressed: {len(messages)} → {len(compressed)} messages"
        )

        return compressed

    @staticmethod
    def _summarize_middle_history(messages: List[BaseMessage]) -> str:
        """
        중간 대화 요약 (핵심 논점만 추출)

        Args:
            messages: 중간 대화 메시지들

        Returns:
            요약된 논점 텍스트

        Strategy:
            - 각 메시지의 첫 문장만 추출 (핵심 논점)
            - 최대 4개 논점만 유지
            - Role별로 정리
        """
        summaries = []

        for msg in messages:
            try:
                content = json.loads(msg.content)
                role = content.get("role", "unknown")
                text = content.get("content", "")

                # 첫 문장만 추출 (핵심 논점)
                sentences = text.split('.')
                first_sentence = sentences[0].strip() if sentences else text[:50]

                # Role 한글 변환
                role_kr = {
                    "prosecutor": "검사",
                    "defense": "변호사",
                    "judge": "판사"
                }.get(role, role)

                summaries.append(f"- {role_kr}: {first_sentence}")

            except (json.JSONDecodeError, KeyError, AttributeError):
                # JSON 파싱 실패 시 무시
                continue

        # 최대 4개 논점만 유지 (최근 순)
        recent_summaries = summaries[-4:] if len(summaries) > 4 else summaries

        if not recent_summaries:
            return "(이전 대화 내용 없음)"

        return "\n".join(recent_summaries)

    @staticmethod
    def extract_opponent_last_argument(
        messages: List[BaseMessage],
        current_role: str
    ) -> Optional[str]:
        """
        상대방의 마지막 주장만 추출

        Args:
            messages: 전체 대화 History
            current_role: 현재 에이전트 Role ("prosecutor" 또는 "defense")

        Returns:
            상대방의 마지막 주장 텍스트 또는 None

        Purpose:
            프롬프트에 명시적으로 "이것을 반박하세요"라고 지시하기 위함
        """
        # 상대방 Role 결정
        opponent_role = "defense" if current_role == "prosecutor" else "prosecutor"

        # 역순으로 탐색 (최근 메시지부터)
        for msg in reversed(messages):
            try:
                content = json.loads(msg.content)
                role = content.get("role", "")

                if role == opponent_role:
                    argument = content.get("content", "")
                    logger.debug(
                        f"Extracted opponent ({opponent_role}) last argument: "
                        f"{argument[:50]}..."
                    )
                    return argument

            except (json.JSONDecodeError, KeyError, AttributeError):
                continue

        logger.debug(f"No opponent ({opponent_role}) argument found in history")
        return None

    @staticmethod
    def extract_key_points_from_history(
        messages: List[BaseMessage],
        max_points: int = 3
    ) -> Dict[str, List[str]]:
        """
        History에서 각 Role별 핵심 논점 추출

        Args:
            messages: 전체 대화 History
            max_points: 각 Role별 최대 논점 개수

        Returns:
            {
                "prosecutor": ["논점1", "논점2"],
                "defense": ["논점1", "논점2"]
            }

        Purpose:
            - 대화 흐름 파악
            - 반복 방지 (이미 주장한 논점 체크)
        """
        key_points = {
            "prosecutor": [],
            "defense": []
        }

        for msg in messages:
            try:
                content = json.loads(msg.content)
                role = content.get("role", "")
                text = content.get("content", "")

                if role in ["prosecutor", "defense"]:
                    # 첫 문장을 핵심 논점으로 간주
                    sentences = text.split('.')
                    first_sentence = sentences[0].strip() if sentences else text[:100]

                    if first_sentence and first_sentence not in key_points[role]:
                        key_points[role].append(first_sentence)

            except (json.JSONDecodeError, KeyError, AttributeError):
                continue

        # 각 Role별 최대 개수만 유지 (최근 순)
        for role in key_points:
            if len(key_points[role]) > max_points:
                key_points[role] = key_points[role][-max_points:]

        return key_points

    @staticmethod
    def create_context_summary(
        messages: List[BaseMessage],
        current_role: str
    ) -> str:
        """
        대화 맥락 요약 생성

        Args:
            messages: 전체 대화 History
            current_role: 현재 에이전트 Role

        Returns:
            구조화된 맥락 요약 텍스트

        Purpose:
            LLM에게 대화 흐름과 현재 상황을 명확히 전달
        """
        # 1. 상대방 마지막 주장
        opponent_last = HistoryCompressor.extract_opponent_last_argument(
            messages, current_role
        )

        # 2. 각 Role별 핵심 논점
        key_points = HistoryCompressor.extract_key_points_from_history(messages)

        # 3. 요약 구성
        summary_parts = []

        summary_parts.append("## 📊 대화 맥락 요약")
        summary_parts.append("")

        # 상대방 마지막 주장 (가장 중요)
        if opponent_last:
            opponent_name = "변호사" if current_role == "prosecutor" else "검사"
            summary_parts.append(f"### 🎯 {opponent_name}의 마지막 주장:")
            summary_parts.append(f"> {opponent_last}")
            summary_parts.append("")
            summary_parts.append("**→ 이 주장을 반박하세요!**")
            summary_parts.append("")

        # 각 Role별 기존 논점
        prosecutor_points = key_points.get("prosecutor", [])
        defense_points = key_points.get("defense", [])

        if prosecutor_points:
            summary_parts.append("### 검사의 기존 논점:")
            for i, point in enumerate(prosecutor_points, 1):
                summary_parts.append(f"{i}. {point}")
            summary_parts.append("")

        if defense_points:
            summary_parts.append("### 변호사의 기존 논점:")
            for i, point in enumerate(defense_points, 1):
                summary_parts.append(f"{i}. {point}")
            summary_parts.append("")

        # 반복 방지 경고
        if current_role == "prosecutor" and prosecutor_points:
            summary_parts.append("⚠️ **주의**: 위 검사 논점은 이미 주장했으므로 반복하지 마세요.")
        elif current_role == "defense" and defense_points:
            summary_parts.append("⚠️ **주의**: 위 변호사 논점은 이미 주장했으므로 반복하지 마세요.")

        return "\n".join(summary_parts)

"""
History Manager - 대화 로그 관리 유틸리티

역할:
1. LangGraph의 실행 결과(Message)를 RDBMS(Conversations 테이블)에 구조화하여 저장
2. 에이전트가 특정 Phase의 Context를 요청할 때, 읽기 좋은 텍스트 형식으로 변환하여 제공
"""
import json
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from src.database.models.conversation import Conversation
from src.database.connection import SessionLocal

class HistoryManager:
    def __init__(self, session_id: str, db: Optional[Session] = None):
        self.session_id = session_id
        self.db = db if db else SessionLocal()
        self.own_db_session = db is None  # 나중에 close 할지 여부

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """명시적으로 DB 세션을 닫음"""
        if self.own_db_session and self.db:
            self.db.close()
            self.db = None

    def save_turn(self, 
                  role: str, 
                  content: str, 
                  turn_count: int, 
                  phase: str, 
                  metadata: Dict[str, Any] = None,
                  case_id: Optional[int] = None) -> Conversation:
        """
        대화 한 턴을 DB에 저장
        
        Args:
            role: 발화자 (prosecutor, defense, etc.)
            content: 발화 내용 (순수 텍스트)
            turn_count: 현재 턴 번호
            phase: 현재 진행 단계
            metadata: 기타 속성 (emotion, references 등)
        """
        # 혹시 content가 JSON 문자열로 들어오는 레거시 케이스 처리
        clean_content = content
        final_metadata = metadata or {}

        try:
            # 만약 content가 JSON 형태라면 파싱 시도 (Refactoring 과도기 대응)
            if content.strip().startswith('{') and content.strip().endswith('}'):
                data = json.loads(content)
                if "content" in data:
                    clean_content = data["content"]
                # 메타데이터 병합
                for key in ["role", "emotion", "references"]:
                    if key in data and key not in final_metadata:
                        final_metadata[key] = data[key]
        except (json.JSONDecodeError, TypeError):
            pass # 일반 텍스트로 간주

        conversation = Conversation(
            session_id=self.session_id,
            turn_count=turn_count,
            phase=phase,
            speaker=role,
            content=clean_content,
            metadata_json=final_metadata,
            case_id=case_id
        )
        
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def get_phase_history(self, phase: str, limit: int = 20) -> str:
        """
        특정 페이즈(Phase)의 대화 내용을 텍스트로 복원
        
        Args:
            phase: 조회할 단계 (debate, briefing 등)
            limit: 가져올 최대 턴 수
            
        Returns:
            "Prosecutor: ... \n Defense: ..." 형태의 문자열
        """
        conversations = self.db.query(Conversation).filter(
            Conversation.session_id == self.session_id,
            Conversation.phase == phase
        ).order_by(Conversation.turn_count.asc()).limit(limit).all()

        formatted_lines = []
        for conv in conversations:
            speaker_display = conv.speaker.upper()
            formatted_lines.append(f"[{speaker_display}]: {conv.content}")
            
        if not formatted_lines:
            return "(No history for this phase yet.)"
            
        return "\n\n".join(formatted_lines)

    def get_full_history_for_ui(self) -> List[Dict]:
        """
        프론트엔드 표시에 적합한 전체 히스토리 리스트 반환
        """
        conversations = self.db.query(Conversation).filter(
            Conversation.session_id == self.session_id
        ).order_by(Conversation.turn_count.asc()).all()
        
        result = []
        for conv in conversations:
            meta = conv.metadata_json or {}
            result.append({
                "role": conv.speaker,
                "content": conv.content,
                "emotion": meta.get("emotion", "neutral"),
                "references": meta.get("references", []),
                "turn": conv.turn_count,
                "phase": conv.phase
            })
        return result

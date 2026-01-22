"""
Conversation Model - 대화 히스토리 저장 (Hybrid Architecture)
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from src.database.connection import Base

class Conversation(Base):
    """
    대화 히스토리 모델
    
    목적:
    1. LangGraph Checkpoint와 별도로, 구조화된 대화 로그를 장기 보관
    2. 에이전트가 특정 Phase(단계)의 대화 내용만 쉽고 빠르게 조회 (Context Retrieval)
    3. UI 렌더링을 위한 메타데이터(감정, 참고자료 등) 저장
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True, comment="PK")
    
    # 세션 및 순서 정보
    session_id = Column(String(100), index=True, nullable=False, comment="세션 ID")
    turn_count = Column(Integer, nullable=False, index=True, comment="전체 턴 수 (정렬용)")
    
    # 컨텍스트 정보
    case_id = Column(Integer, nullable=True, comment="관련 사건 ID (Optional, No FK constraint)")
    phase = Column(String(50), index=True, nullable=False, comment="진행 단계 (briefing, debate, judgment, etc.)")
    speaker = Column(String(50), index=True, nullable=False, comment="발화자 (prosecutor, defense, judge, etc.)")
    
    # 내용
    content = Column(Text, nullable=False, comment="순수 발화 내용 (JSON 아님)")
    metadata_json = Column(JSON, default={}, comment="메타데이터 (emotion, references, role_detail 등)")
    
    created_at = Column(DateTime, default=datetime.utcnow, comment="생성 일시")

    # Relationships
    # Case 모델이 정의되어 있다면 아래 주석 해제하여 사용
    # case = relationship("Case", backref="conversations")
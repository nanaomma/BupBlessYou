"""Judgment model - 판결 결과"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from src.database.connection import Base


class Judgment(Base):
    """판결 결과 모델"""
    __tablename__ = "judgments"

    id = Column(Integer, primary_key=True, index=True, comment="판결 ID")
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False, comment="관련 사건 ID")
    session_id = Column(String(100), index=True, comment="세션 식별자")

    # 유저 판결
    user_sentence = Column(String(200), comment="유저가 선고한 형량")
    user_reasoning = Column(JSON, comment="유저의 양형 사유")

    # 비교 결과
    similarity_score = Column(Float, comment="실제 판결과의 유사도 점수")
    difference_analysis = Column(Text, comment="실제 판결과의 차이점 분석")
    feedback = Column(Text, comment="AI 판사의 피드백")

    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow, comment="판결 생성 일시")

    # Relationships
    case = relationship("Case", backref="judgments")

"""Case model - 사건 데이터"""
import json
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.types import TypeDecorator
from datetime import datetime
from src.database.connection import Base


class UnicodeJSON(TypeDecorator):
    """
    PostgreSQL JSON 타입에 한글(유니코드) 문자를 올바르게 저장하기 위한 커스텀 타입
    ensure_ascii=False로 설정하여 한글을 유니코드 이스케이프 없이 저장
    
    SQLAlchemy의 기본 JSON 타입은 ensure_ascii=True를 사용하므로,
    TEXT 타입을 사용하고 process_bind_param에서 JSON 문자열로 변환
    """
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            # 이미 문자열인 경우 (JSON 문자열) - 유효성 검증만 수행
            if isinstance(value, str):
                try:
                    # JSON 형식 검증
                    json.loads(value)
                    return value
                except (json.JSONDecodeError, TypeError):
                    # 유효하지 않은 JSON 문자열인 경우 그대로 반환 (에러 발생 가능)
                    return value
            
            # Python 딕셔너리/리스트를 JSON 문자열로 변환 (ensure_ascii=False)
            return json.dumps(value, ensure_ascii=False)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            # TEXT 타입이므로 문자열로 반환됨 - JSON으로 파싱
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            # 이미 딕셔너리인 경우 (드물지만)
            return value
        return value


class Case(Base):
    """사건 정보 모델"""
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True, comment="사건 ID")
    case_number = Column(String(50), unique=True, index=True, comment="사건 번호")
    title = Column(String(200), comment="사건 제목")
    casetype = Column(String(50), comment="민사/형사")
    casename = Column(String(50), comment="범죄 유형")
    description = Column(Text, comment="사건 개요")
    facts = Column(Text, comment="사실관계")

    # 실제 판결 정보
    actual_label = Column(JSON, comment="실제 형량")
    actual_rule = Column(JSON, comment="형량 적용 규칙")
    actual_reason = Column(Text, comment="실제 양형 사유")

    # 양형 요소 (actual_reason에서 추출, facts/description에 있는 정보만)
    sentencing_factors = Column(UnicodeJSON, comment="양형 고려 요소 (카테고리화된 구조)")

    # 사용 여부
    use_yn = Column(String(1), default='N', nullable=False, comment="사용 여부 (Y/N)")

    # 메타데이터
    source = Column(String(100), comment="데이터 출처")
    created_at = Column(DateTime, default=datetime.utcnow, comment="생성 일시")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="수정 일시")
    case_metadata = Column(JSON, comment="메타데이터")
    
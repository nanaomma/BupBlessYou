"""add_sentencing_factors

Revision ID: add_sentencing_factors
Revises: 7c3a71824fa2
Create Date: 2025-01-XX

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_sentencing_factors'
down_revision = '7c3a71824fa2'  # 최신 마이그레이션 head
branch_labels = None
depends_on = None


def upgrade() -> None:
    # cases 테이블에 sentencing_factors JSON 컬럼 추가
    # TEXT 타입 사용 (UnicodeJSON TypeDecorator가 JSON 문자열로 변환)
    # 한글 유니코드 이스케이프 방지를 위해 TEXT + 커스텀 TypeDecorator 사용
    op.add_column('cases', sa.Column('sentencing_factors', sa.Text(), nullable=True, comment='양형 고려 요소 (카테고리화된 구조)'))


def downgrade() -> None:
    op.drop_column('cases', 'sentencing_factors')


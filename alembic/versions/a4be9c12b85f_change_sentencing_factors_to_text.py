"""change_sentencing_factors_to_text

Revision ID: a4be9c12b85f
Revises: add_sentencing_factors
Create Date: 2025-12-29 12:18:30.536840

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a4be9c12b85f'
down_revision: Union[str, Sequence[str], None] = 'add_sentencing_factors'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # sentencing_factors 컬럼을 JSON에서 TEXT로 변경
    # UnicodeJSON TypeDecorator가 TEXT 타입을 사용하므로 컬럼 타입도 TEXT로 변경 필요
    # JSON -> TEXT로 변경 (한글 유니코드 이스케이프 방지)
    op.alter_column('cases', 'sentencing_factors',
                    type_=sa.Text(),
                    existing_type=sa.JSON(),
                    existing_nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    # TEXT -> JSON으로 되돌림
    op.alter_column('cases', 'sentencing_factors',
                    type_=sa.JSON(),
                    existing_type=sa.Text(),
                    existing_nullable=True)

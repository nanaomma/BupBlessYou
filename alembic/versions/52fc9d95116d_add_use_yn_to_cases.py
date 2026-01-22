"""add_use_yn_to_cases

Revision ID: 52fc9d95116d
Revises: a4be9c12b85f
Create Date: 2026-01-02 17:28:48.720666

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '52fc9d95116d'
down_revision: Union[str, Sequence[str], None] = 'a4be9c12b85f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add USE_YN column to cases table with default value 'N'
    op.add_column('cases', sa.Column('use_yn', sa.String(1), nullable=False, server_default='N', comment='사용 여부 (Y/N)'))


def downgrade() -> None:
    """Downgrade schema."""
    # Remove USE_YN column from cases table
    op.drop_column('cases', 'use_yn')

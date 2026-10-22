"""create job_preferences table for mysql

Revision ID: 004_create_job_preferences_table
Revises: 003_create_user_profiles_table
Create Date: 2026-08-13 15:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '004_create_job_preferences_table'
down_revision: Union[str, None] = '003_create_user_profiles_table'
branch_labels: Union[Sequence[str], str, None] = None
depends_on: Union[Sequence[str], str, None] = None

def upgrade() -> None:
    op.create_table(
        'job_preferences',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('auto_apply_enabled', sa.Boolean(), nullable=False, server_default=sa.text('0')),
        sa.Column('daily_apply_limit', sa.Integer(), nullable=False, server_default='10'),
        sa.Column('desired_job_titles', sa.Text(), nullable=True),
        sa.Column('preferred_industries', sa.Text(), nullable=True),
        sa.Column('min_salary', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_job_preferences_id'), 'job_preferences', ['id'], unique=False)
    op.create_index(op.f('ix_job_preferences_user_id'), 'job_preferences', ['user_id'], unique=True)

def downgrade() -> None:
    op.drop_index(op.f('ix_job_preferences_user_id'), table_name='job_preferences')
    op.drop_index(op.f('ix_job_preferences_id'), table_name='job_preferences')
    op.drop_table('job_preferences')

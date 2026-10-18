"""v3.2: jobs.hits, the result-cache hits already handed back to a job's quota reservation.

Persisted so the refund of a cancelled / failed job is exact after a restart (it used to live in the runner's
memory only).

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("hits", sa.BigInteger, nullable=False, server_default=sa.text("0")))


def downgrade() -> None:
    raise NotImplementedError("clef migrations are forward-only")

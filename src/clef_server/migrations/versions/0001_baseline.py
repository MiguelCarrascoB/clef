"""Baseline: the v3.1 jobs schema (jobs, items, jobs_status, jobs_idem).

A legacy jobs.db (a ``jobs`` table without ``alembic_version``) is not migrated through this revision:
``Database.upgrade`` brings it to exactly this shape and stamps it ``0001``.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

IdentityInt = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("seq", IdentityInt, sa.Identity(always=False), primary_key=True),
        sa.Column("id", sa.Text, nullable=False, unique=True),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("owner", sa.Text),
        sa.Column("payload", sa.Text, nullable=False),
        sa.Column("metadata", sa.Text),
        sa.Column("webhook", sa.Text),
        sa.Column("deliveries", sa.Text, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("total", sa.Integer),
        sa.Column("done", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("failed", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.Double, nullable=False),
        sa.Column("started_at", sa.Double),
        sa.Column("finished_at", sa.Double),
        sa.Column("run_started_at", sa.Double),
        sa.Column("run_base_done", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("resumes", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("cancel_requested", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("error", sa.Text),
        sa.Column("result", sa.Text),
        sa.Column("idem_key", sa.Text),
        sqlite_autoincrement=True,
    )
    op.create_index("jobs_status", "jobs", ["status", "seq"])
    op.create_index(
        "jobs_idem",
        "jobs",
        [sa.text("COALESCE(owner, '')"), "idem_key"],
        unique=True,
        sqlite_where=sa.text("idem_key IS NOT NULL"),
        postgresql_where=sa.text("idem_key IS NOT NULL"),
    )
    op.create_table(
        "items",
        sa.Column("job_id", sa.Text, sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("idx", sa.Integer, nullable=False),
        sa.Column("data", sa.Text, nullable=False),
        sa.PrimaryKeyConstraint("job_id", "idx"),
        sqlite_with_rowid=False,
    )


def downgrade() -> None:
    raise NotImplementedError("clef migrations are forward-only")

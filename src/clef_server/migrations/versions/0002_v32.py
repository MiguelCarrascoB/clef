"""v3.2: every table the v3.2 waves need (storage, keys, router / review, files / batches) + new jobs columns.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

IdentityInt = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
T, D, I, B = sa.Text, sa.Double, sa.Integer, sa.BigInteger  # noqa: E741


def _identity(name: str) -> sa.Column:
    return sa.Column(name, IdentityInt, sa.Identity(always=False), primary_key=True)


def upgrade() -> None:
    # ---- jobs: C1 (model) and E1 (scheduler) columns
    op.add_column("jobs", sa.Column("model", T))
    op.add_column("jobs", sa.Column("priority", I, nullable=False, server_default=sa.text("1")))
    op.add_column("jobs", sa.Column("schedule", T))
    op.add_column("jobs", sa.Column("expire_after_s", D))
    op.add_column("jobs", sa.Column("eligible_at", D))
    op.add_column("jobs", sa.Column("quota_day", T))
    op.add_column("jobs", sa.Column("quota_reserved", B, nullable=False, server_default=sa.text("0")))
    op.add_column("jobs", sa.Column("tokens", B, nullable=False, server_default=sa.text("0")))

    # ---- storage (C1)
    op.create_table("clef_meta", sa.Column("k", T, primary_key=True), sa.Column("v", T, nullable=False))
    op.create_table(
        "classifiers",
        sa.Column("name", T, primary_key=True),
        sa.Column("owner", T),
        sa.Column("latest_version", I, nullable=False),
        sa.Column("created_at", D, nullable=False),
        sa.Column("updated_at", D, nullable=False),
    )
    op.create_table(
        "classifier_versions",
        sa.Column("name", T, sa.ForeignKey("classifiers.name", ondelete="CASCADE"), nullable=False),
        sa.Column("version", I, nullable=False),
        sa.Column("definition", T, nullable=False),
        sa.Column("model", T),
        sa.Column("created_at", D, nullable=False),
        sa.Column("created_by", T),
        sa.PrimaryKeyConstraint("name", "version"),
    )
    op.create_table(
        "audit_log",
        _identity("id"),
        sa.Column("ts", D, nullable=False),
        sa.Column("actor", T, nullable=False),
        sa.Column("action", T, nullable=False),
        sa.Column("target", T, nullable=False),
        sa.Column("detail", T),
        sa.Column("ip", T),
        sa.Column("request_id", T),
        sqlite_autoincrement=True,
    )
    op.create_index("audit_ts", "audit_log", ["ts"])

    # ---- keys and principals (D1)
    op.create_table(
        "api_principals",
        sa.Column("name", T, primary_key=True),
        sa.Column("scopes", T, nullable=False),
        sa.Column("daily_quota", B),
        sa.Column("rate_limit", I),
        sa.Column("note", T),
        sa.Column("created_at", D, nullable=False),
        sa.Column("created_by", T),
        sa.Column("disabled_at", D),
    )
    op.create_table(
        "api_keys",
        sa.Column("id", T, primary_key=True),
        sa.Column("principal", T, sa.ForeignKey("api_principals.name"), nullable=False),
        sa.Column("secret_sha256", T, nullable=False),
        sa.Column("created_at", D, nullable=False),
        sa.Column("created_by", T),
        sa.Column("expires_at", D),
        sa.Column("revoked_at", D),
        sa.Column("last_used_at", D),
    )
    op.create_index("api_keys_principal", "api_keys", ["principal"])
    op.create_table(
        "api_usage",
        sa.Column("principal", T, nullable=False),
        sa.Column("day", T, nullable=False),
        sa.Column("records", B, nullable=False, server_default=sa.text("0")),
        sa.Column("requests", B, nullable=False, server_default=sa.text("0")),
        sa.Column("rejected", B, nullable=False, server_default=sa.text("0")),
        sa.PrimaryKeyConstraint("principal", "day"),
    )

    # ---- review queue and router (E2)
    op.create_table(
        "routers",
        sa.Column("name", T, primary_key=True),
        sa.Column("owner", T),
        sa.Column("classifier", T, nullable=False, unique=True),
        sa.Column("config", T, nullable=False),
        sa.Column("mode", T, nullable=False),
        sa.Column("active_version", I),
        sa.Column("created_at", D, nullable=False),
        sa.Column("updated_at", D, nullable=False),
    )
    op.create_table(
        "router_label_sets",
        sa.Column("router", T, sa.ForeignKey("routers.name", ondelete="CASCADE"), nullable=False),
        sa.Column("version", I, nullable=False),
        sa.Column("labels", T, nullable=False),
        sa.Column("source", T, nullable=False),
        sa.Column("classifier_version", I),
        sa.Column("created_at", D, nullable=False),
        sa.Column("created_by", T),
        sa.PrimaryKeyConstraint("router", "version"),
    )
    op.create_table(
        "review",
        sa.Column("id", T, primary_key=True),
        sa.Column("kind", T, nullable=False),
        sa.Column("state", T, nullable=False),
        sa.Column("owner", T),
        sa.Column("router", T),
        sa.Column("classifier", T),
        sa.Column("classifier_version", I),
        sa.Column("reason", T, nullable=False),
        sa.Column("input", T),
        sa.Column("input_sha256", T),
        sa.Column("prediction", T),
        sa.Column("proposal", T),
        sa.Column("label", T),
        sa.Column("reroutes", I, nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", D, nullable=False),
        sa.Column("resolved_at", D),
        sa.Column("resolved_by", T),
    )
    op.create_index("review_state", "review", ["state", "created_at"])
    op.create_index("review_router", "review", ["router", "state"])
    op.create_table(
        "llm_calls",
        _identity("id"),
        sa.Column("ts", D, nullable=False),
        sa.Column("router", T),
        sa.Column("purpose", T, nullable=False),
        sa.Column("provider_url", T, nullable=False),
        sa.Column("model", T, nullable=False),
        sa.Column("status", T, nullable=False),
        sa.Column("latency_ms", D),
        sa.Column("input_tokens", B),
        sa.Column("output_tokens", B),
        sa.Column("items", I),
        sa.Column("error", T),
        sqlite_autoincrement=True,
    )

    # ---- OpenAI files and batches (F1)
    op.create_table(
        "files",
        sa.Column("id", T, primary_key=True),
        sa.Column("owner", T),
        sa.Column("purpose", T, nullable=False),
        sa.Column("role", T, nullable=False),
        sa.Column("filename", T, nullable=False),
        sa.Column("bytes", B, nullable=False),
        sa.Column("lines", I),
        sa.Column("job_id", T, sa.ForeignKey("jobs.id", ondelete="CASCADE")),
        sa.Column("created_at", D, nullable=False),
        sa.Column("expires_at", D),
    )
    op.create_index("files_owner", "files", ["owner", "created_at"])
    op.create_table(
        "batches",
        sa.Column("id", T, primary_key=True),
        sa.Column("job_id", T, sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("owner", T),
        sa.Column("endpoint", T, nullable=False),
        sa.Column("input_file_id", T, nullable=False),
        sa.Column("output_file_id", T, nullable=False),
        sa.Column("error_file_id", T, nullable=False),
        sa.Column("completion_window", T, nullable=False),
        sa.Column("metadata", T),
        sa.Column("errors", T),
        sa.Column("created_at", D, nullable=False),
        sa.Column("cancelling_at", D),
    )


def downgrade() -> None:
    raise NotImplementedError("clef migrations are forward-only")

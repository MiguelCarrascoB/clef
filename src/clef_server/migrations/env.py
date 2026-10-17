"""Alembic environment. ``db.py`` builds the Config in code and passes the open connection.

Online only: ``config.attributes["connection"]`` is a SQLAlchemy Connection already inside the caller's
transaction (``Database.tx()``), so the whole upgrade commits or rolls back as one.
"""

from __future__ import annotations

from alembic import context

from clef_server.dbschema import metadata

connection = context.config.attributes.get("connection")
if connection is None:  # pragma: no cover - db.py always passes one
    raise RuntimeError("run migrations through clef_server.db (clef db upgrade), not the alembic CLI")

context.configure(connection=connection, target_metadata=metadata, render_as_batch=True)
with context.begin_transaction():
    context.run_migrations()

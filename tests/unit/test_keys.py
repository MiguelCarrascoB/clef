"""Keys and principals (D1): key format, store, cache, quotas, auth and scopes over HTTP, the admin API.

The default-deny test at the end walks the routes: a route without a scope dependency fails it.
"""

from __future__ import annotations

from typing import Any

from clef_server.keystore import Actor, KeyStore


def add_key(app: Any, name: str, scopes: str = "classify,jobs,classifiers,ops", **kw: Any) -> str:
    """A DB principal on a test app (the cache is reloaded at once); returns its secret."""
    ctx = app.state.ctx
    _, secret = KeyStore(ctx.db).create_principal(name, scopes, actor=Actor("test"), **kw)
    ctx.keys.reload()
    return secret

"""Export the FastAPI OpenAPI schema to openapi.json at the repo root.

    python scripts/export_openapi.py            # (re)write openapi.json
    python scripts/export_openapi.py --check    # exit 1 when openapi.json is stale (CI)

The app is built with a fake engine and a throw-away state dir, so no model, GPU or user state is touched.
Output is deterministic: sorted keys, 2-space indent, LF line endings, trailing newline.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "openapi.json"
sys.path.insert(0, str(ROOT / "src"))


class FakeEngine:
    status, error, load_seconds = "ready", None, None

    def start(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def info(self) -> dict[str, Any]:
        return {}

    async def decide(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError


def render() -> str:
    from clef_server.config import Config
    from clef_server.main import create_app

    with tempfile.TemporaryDirectory(prefix="clef-openapi-") as state:
        cfg = Config(api_key=None, api_keys_raw=None, state_dir=state, cors_origins=())
        spec = create_app(cfg, FakeEngine()).openapi()
    return json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if openapi.json is out of date")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = TARGET.read_bytes().decode("utf-8").replace("\r\n", "\n") if TARGET.exists() else None
        if current != text:
            print("openapi.json is stale: run `python scripts/export_openapi.py`", file=sys.stderr)
            return 1
        print("openapi.json is up to date")
        return 0
    TARGET.write_bytes(text.encode("utf-8"))
    print(f"wrote {TARGET} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

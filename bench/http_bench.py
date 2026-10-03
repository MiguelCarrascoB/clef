"""Shim: the HTTP load test lives in clef_server.httpbench (`clef bench`). Same flags as before.

python bench/http_bench.py --concurrency 1,2,4,8,16 --requests 100 [--mixed] [--url http://127.0.0.1:8910]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clef_server.httpbench import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

## What and why

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pytest tests/unit` passes
- [ ] `python scripts/export_openapi.py --check` passes (run it without `--check` and commit `openapi.json` if the API changed)
- [ ] Docs and CHANGELOG updated when behaviour changes (`docs/ARCHITECTURE.md` is the contract)
- [ ] Backend-specific change: which hardware did you run it on? (write "not run on hardware" if none)

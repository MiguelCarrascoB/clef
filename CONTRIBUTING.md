# Contributing

Thanks for helping. Architecture and the API contract live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): change the
contract first, then the code.

## Dev setup

Python 3.10 or newer (the code must stay 3.10 compatible: no `typing.Self`, `tomllib`, `ExceptionGroup`). The unit tests
need only CPU torch; they never load the model weights. [uv](https://docs.astral.sh/uv/) is recommended.

**Linux / macOS**

```bash
git clone https://github.com/MiguelCarrascoB/clef && cd clef
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/cpu.txt        # macOS: requirements/macos.txt
uv pip install -e ".[server,dev]"
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/MiguelCarrascoB/clef; cd clef
uv venv --python 3.12; .venv\Scripts\activate
uv pip install -r requirements/cpu.txt
uv pip install -e ".[server,dev]"
```

**Windows + WSL2 / AMD ROCm (the maintainer's setup)**: see the README quick start, install `requirements/rocm.txt`
inside WSL, then `pip install -e ".[server,dev]"`. Use `.\clef.ps1` from Windows.

**Node** (JS client only): Node 18+, no install step.

## Checks (run them before a PR)

```bash
ruff check .
ruff format --check .
pytest tests/unit                       # CPU, no weights
python scripts/export_openapi.py --check   # openapi.json up to date; run without --check after API changes
node --test clients/js/test/client.test.js           # JS client
shellcheck scripts/*.sh
```

If you touch the docs, build the site strictly ( any broken link or warning fails). Python code blocks in
Markdown are formatted by `ruff format`, so keep them formatted and free of aligned trailing comments:

```bash
python -m venv .venv-docs && .venv-docs/bin/pip install -r requirements/docs.txt   # Windows: .venv-docs\Scripts\pip
NO_MKDOCS_2_WARNING=true .venv-docs/bin/mkdocs build --strict
```

The site at <https://miguelcarrascob.github.io/clef/> is served by GitHub Pages from the `gh-pages` branch. GitHub
Actions is disabled for this repository, so publish by hand after committing docs changes:
`bash scripts/publish_docs.sh` (strict build, then one commit on `gh-pages`).

Run `ruff format` only on files you changed; a repo-wide format in a multi-person change touches everyone's files.
Python line length is 110.

The repository has no CI (GitHub Actions is disabled), so these checks are the gate: run all of them before opening
a PR, on Python 3.10 if you can (the oldest supported version). Nothing is published to PyPI, npm or a registry.

## Tests

- `tests/unit`: CPU, no weights, no GPU, mocked backends and engine. Add tests with every change; backend-specific
  logic is tested with fake `Backend` objects.
- `tests/integration` (`pytest -m gpu`): runs against a live server (`CLEF_URL`, `CLEF_API_KEY`) and skips when none
  answers.

## Rules of the code

- `backend.py` is the only place with device-specific code. No `torch.cuda.*` elsewhere (`offload.py` is device-agnostic
  and gets its stream and pinned-memory helpers from `backend.py`).
- New endpoints are **feature modules**: a file in `src/clef_server/` exposing `router(ctx: AppContext) -> APIRouter`,
  added to `main.FEATURES` (see `appctx.py` and the "Feature modules" section of
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)). Call the model through `ctx.infer` / `ctx.decide`, put `ctx.auth` on
  the router and `ctx.limited` on routes that run inference, and add pydantic models so `openapi.json` covers the
  route.
- One uvicorn worker, one GPU worker thread.
- Never edit the model directory; wrap or re-implement in the engine.
- No hardcoded user paths (`/home/<name>`, `C:\Users\<name>`); use `~`, `%LOCALAPPDATA%` or the state directory helpers.
- Do not log request contents or keys.

## Hardware checklist

The unit tests cannot exercise GPUs. If your change touches a backend (dtype, memory, fast paths,
quantization, telemetry), say in the PR which hardware you ran it on. Maintainers with a real NVIDIA or Mac machine:
follow [docs/hardware-validation.md](docs/hardware-validation.md) and paste the results. Performance changes need
before/after numbers from `clef bench`; a regression of more than 5% on the verified ROCm setup blocks a change.

## Pull requests

Small, focused PRs with the checklist in the template filled in. Update the CHANGELOG (`## Unreleased`) for
user-visible changes, and the guide in `docs/` when you change behaviour it describes.

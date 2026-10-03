# Examples

The same job in four languages: triage a support ticket ("Checkout is down, orders blocked") into
`billing` / `technical` / `account`, then a multi-label call, a score call and a saved classifier
(`support-triage`).

| File | Needs |
| --- | --- |
| `curl.sh` | bash, curl |
| `classify.py` | `pip install -e .` (or `PYTHONPATH=src`) |
| `classify.mjs` | Node 18+ |
| `classify.ps1` | PowerShell 5.1+ |
| `classify_csv.py` | same as `classify.py`; classifies a CSV column in chunks |
| `openai_sdk.py` | `pip install openai`; the official OpenAI SDK against clef ([guide](../docs/openai-compat.md)) |
| `hf_zero_shot.py` | `pip install huggingface_hub`; `InferenceClient.zero_shot_classification` against clef |
| `jobs.py` | same as `classify.py`; classifies a CSV column as an async job (submit, poll, page results), see [docs/jobs.md](../docs/jobs.md) |

All read two environment variables:

- `CLEF_URL` - server address, default `http://127.0.0.1:8910`
- `CLEF_API_KEY` - sent as `X-API-Key`; only needed when the server has auth on

Start the server first (`clef serve`), then for example:

```bash
bash examples/curl.sh
python examples/classify.py
node examples/classify.mjs
pwsh examples/classify.ps1        # or: powershell -File examples\classify.ps1
python examples/classify_csv.py examples/tickets.csv -o out.csv --column text --labels billing,technical,account
python examples/jobs.py examples/tickets.csv -o out.csv --column text --labels billing,technical,account
```

`classify_csv.py` writes the input columns plus `label`, `confidence` and one `score_<label>` column per label
(`--multi-label` joins every hit with `|`). Remote server: `CLEF_URL=https://clef.example.com CLEF_API_KEY=... python examples/classify.py`.

The saved-classifier steps create then delete `support-triage`; remove the delete line to keep it.

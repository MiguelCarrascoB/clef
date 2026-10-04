# First request

Start the server and wait until it is ready (the model loads in the background):

```bash
clef serve --detach        # or plain `clef serve` in a spare terminal
clef status                # backend, device, status
curl -s http://127.0.0.1:8910/health
```

Then ask which label fits an input.

=== "curl"

    ```bash
    curl -s http://127.0.0.1:8910/v1/classify -H 'Content-Type: application/json' \
      -d '{"input": "Checkout is down", "labels": ["billing", "technical"]}'
    # {"label":"technical","confidence":0.96,"scores":{"billing":0.04,"technical":0.96}, ...}
    ```

=== "Python"

    ```python
    from clef_client import ClefClient  # AsyncClefClient has the same methods

    c = ClefClient("http://127.0.0.1:8910", api_key=None)

    r = c.classify("Checkout is down", ["billing", "technical"])
    r.label, r.confidence, r.scores  # 'technical', 0.96, {...}

    # multi-label: every label whose P(true) >= threshold
    c.classify("Refund me, the app also crashes", ["billing", "technical", "feature"], multi_label=True).labels

    c.classify_many(["...", "..."], ["billing", "technical"])  # shared label set, results in input order
    c.score("Server has been down for two hours", ["low", "medium", "high"]).score  # expected level index

    c.classifier("support-triage").classify("My invoice is wrong")  # a saved classifier
    ```

=== "JavaScript"

    Node 18+ and browsers, no dependencies (`clients/js`).

    ```js
    import { ClefClient } from "./clients/js/src/index.js";

    const c = new ClefClient({ baseUrl: "http://127.0.0.1:8910" });
    const r = await c.classify("Checkout is down", ["billing", "technical"]);
    console.log(r.label, r.confidence, r.scores);
    await c.classifier("support-triage").classify("My invoice is wrong");
    ```

Saved classifiers (`PUT /v1/classifiers/{name}`) store labels and instructions on the server so callers send only the
input. More runnable examples live in
[`examples/`](https://github.com/MiguelCarrascoB/clef/tree/main/examples) (curl, Python, JavaScript, PowerShell, a
CSV classifier, the OpenAI and Hugging Face clients, an async job).

The original SystemOne API (`POST /v1/systemone`, `POST /v1/batch`, typed `choice` / `score` / `noul` questions,
images and videos) is unchanged; see [Architecture](ARCHITECTURE.md).

## Where to go from here

- [Classification guide](classification.md): labels, instructions, multi-label, scoring, saved classifiers.
- [OpenAI & Hugging Face compatibility](openai-compat.md): use the OpenAI SDK or `huggingface_hub` against your server.
- [Evaluation](evaluation.md): measure accuracy and calibration on your own labelled data.
- [Jobs & webhooks](jobs.md): run 100,000 rows in the background.
- [Smaller GPUs](memory.md): fit the model into 16, 12 or 8 GB.
- [HTTP API](api-reference.md): every endpoint, interactive.
- [Console](console.md): open <http://127.0.0.1:8910/> to try inputs in the browser.
- Something wrong? Run `clef doctor` and see [troubleshooting](troubleshooting.md).

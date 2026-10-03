"""Support-ticket triage with the Python client. Reads CLEF_URL and CLEF_API_KEY from the environment."""

from clef_client import ClefClient

TICKET = "Checkout is down, orders blocked"
LABELS = ["billing", "technical", "account"]

with ClefClient() as clef:  # base_url / api_key default to CLEF_URL / CLEF_API_KEY
    # Single label
    r = clef.classify(TICKET, LABELS)
    print(f"single : {r.label} ({r.confidence:.2f})  scores={r.scores}")

    # Multi label: every label whose score is >= threshold, best first
    r = clef.classify(
        "I was charged twice and the app crashes on login",
        {
            "billing": "Payments, invoices, refunds",
            "technical": "Bugs and outages",
            "account": "Login, profile, permissions",
        },
        multi_label=True,
        threshold=0.5,
    )
    print(f"multi  : {r.labels}  scores={r.scores}")

    # Ordinal score
    s = clef.score(TICKET, ["low", "medium", "high"], instructions="How urgent is this ticket?")
    print(f"score  : {s.level} (score={s.score:.2f}, confidence={s.confidence:.2f})")

    # Many inputs in one request
    texts = ["Refund my invoice", "Cannot reset my password"]
    for text, res in zip(texts, clef.classify_many(texts, LABELS), strict=True):
        print(f"batch  : {text!r} -> {res.label}")

    # Saved classifier
    triage = clef.classifier("support-triage")
    triage.save(labels=LABELS, description="Support ticket routing")
    print("saved  :", triage.classify(TICKET).label)
    triage.delete()

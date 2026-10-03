# End-to-end test of the local clef-flash SystemOne server
$ErrorActionPreference = 'Stop'
$base = 'http://localhost:8910'

# --- 1) Single SystemOne request: text/JSON state, choice + score + noul types
$body = @{
    model = 'clef-flash'
    state = 'Our checkout started returning errors and orders are blocked.'
    questions = [ordered]@{
        department = [ordered]@{
            type = 'choice'
            instructions = 'Which team should handle the message?'
            criteria = [ordered]@{ billing = 'Payments or invoices'; technical = 'Bugs or outages' }
        }
        urgency = [ordered]@{ type = 'score'; criteria = @('Can wait', 'This week', 'Today') }
        outage  = [ordered]@{ type = 'noul'; instructions = 'Is a service down?' }
    }
} | ConvertTo-Json -Depth 6
$r1 = Invoke-RestMethod -Uri "$base/v1/systemone" -Method Post -Body $body -ContentType 'application/json'
"== /v1/systemone =="
$r1 | ConvertTo-Json -Depth 6

# --- 2) Batched: three requests, ONE forward pass
$mk = { param($txt, $n) @{
    model = 'clef-flash'
    state = @{ ticket = @{ text = $txt; customers_affected = $n } }
    questions = [ordered]@{
        department = [ordered]@{ type = 'choice'; instructions = 'Which team owns this?'
            criteria = [ordered]@{ billing = 'Payments or invoices'; technical = 'Bugs or outages' } }
        outage = [ordered]@{ type = 'noul'; instructions = 'Is a service down?' }
    }
} }
$batch = @{
    batch = @(
        & $mk 'Customers are double-charged on invoices since the payment update.' 300
        & $mk 'Orders blocked at checkout, 1200 customers affected.' 1200
        & $mk 'Wordmark in the footer is 2px off-center.' 1
    )
} | ConvertTo-Json -Depth 6
$r2 = Invoke-RestMethod -Uri "$base/v1/batch" -Method Post -Body $batch -ContentType 'application/json'
"`n== /v1/batch (3 records, one forward pass) =="
"batch_ms: $($r2.batch_ms)"
foreach ($res in $r2.results) {
    $conf = $res.answers.department.confidence
    "department: $($res.answers.department.choice)  (p=$conf)  outage_p_true: $($res.answers.outage.probabilities.true)"
}

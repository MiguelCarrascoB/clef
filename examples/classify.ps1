# Support-ticket triage with PowerShell 5.1+ / 7. Reads CLEF_URL and CLEF_API_KEY.
$ErrorActionPreference = 'Stop'

$Url = if ($env:CLEF_URL) { $env:CLEF_URL.TrimEnd('/') } else { 'http://127.0.0.1:8910' }
$Headers = @{}
if ($env:CLEF_API_KEY) { $Headers['X-API-Key'] = $env:CLEF_API_KEY }

function Invoke-Clef([string]$Method, [string]$Path, $Body = $null) {
    $req = @{ Method = $Method; Uri = "$Url$Path"; Headers = $Headers; ContentType = 'application/json' }
    if ($null -ne $Body) { $req.Body = ($Body | ConvertTo-Json -Depth 10 -Compress) }
    Invoke-RestMethod @req
}

$ticket = 'Checkout is down, orders blocked'
$labels = @('billing', 'technical', 'account')

Write-Host '== single label'
$r = Invoke-Clef POST '/v1/classify' @{ input = $ticket; labels = $labels }
"{0} ({1:N2})" -f $r.label, $r.confidence

Write-Host '== multi label'
$r = Invoke-Clef POST '/v1/classify' @{
    input = 'I was charged twice and the app crashes on login'
    labels = @{ billing = 'Payments, invoices, refunds'; technical = 'Bugs and outages'; account = 'Login, profile, permissions' }
    multi_label = $true
    threshold = 0.5
}
$r.labels -join ', '

Write-Host '== score'
$r = Invoke-Clef POST '/v1/score' @{ input = $ticket; levels = @('low', 'medium', 'high'); instructions = 'How urgent is this ticket?' }
"{0} (score {1:N2})" -f $r.level, $r.score

Write-Host '== saved classifier'
$null = Invoke-Clef PUT '/v1/classifiers/support-triage' @{ kind = 'classify'; labels = $labels; description = 'Support ticket routing' }
(Invoke-Clef POST '/v1/classifiers/support-triage' @{ input = $ticket }).label
$null = Invoke-Clef DELETE '/v1/classifiers/support-triage'

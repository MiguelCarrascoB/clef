<#
.SYNOPSIS
  Register (or remove) a Task Scheduler task that starts clef when you log on to Windows.
.DESCRIPTION
  Runs `clef.ps1 start` (WSL backend with the hidden keep-alive session) hidden, at logon, as the current user.
  Use -Native to run `clef serve --detach` from a native Windows install instead (uv tool / venv on PATH).
.EXAMPLE
  .\deploy\windows-task.ps1              # register
  .\deploy\windows-task.ps1 -Remove      # unregister
  .\deploy\windows-task.ps1 -Native      # native Windows (CPU / CUDA) instead of WSL
#>
param(
    [switch]$Remove,
    [switch]$Native,
    [string]$TaskName = 'clef'
)
$ErrorActionPreference = 'Stop'

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "removed task '$TaskName'"
    return
}

$repo = Split-Path -Parent $PSScriptRoot
if ($Native) {
    $clef = (Get-Command clef -ErrorAction Stop).Source
    $action = New-ScheduledTaskAction -Execute $clef -Argument 'serve --detach'
} else {
    $script = Join-Path $repo 'clef.ps1'
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`" start"
}
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description 'Starts the clef local decision server at logon' -Force | Out-Null
Write-Host "registered task '$TaskName' (runs at logon). Test now: Start-ScheduledTask -TaskName $TaskName"

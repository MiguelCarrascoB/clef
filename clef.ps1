<#
.SYNOPSIS
  Windows wrapper for the clef server running in WSL2 (Ubuntu): a thin layer over the `clef` CLI inside the venv.
  Keeps a hidden WSL session alive so the detached server survives. (Native Windows install: use `clef` directly.)
.EXAMPLE
  .\clef.ps1 start | stop | restart | status | logs | test | bench [http] [args] | doctor | open
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'restart', 'status', 'logs', 'test', 'bench', 'doctor', 'open', 'help')]
    [string]$Command = 'help',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = 'Stop'
$Distro = if ($env:CLEF_WSL_DISTRO) { $env:CLEF_WSL_DISTRO } else { 'Ubuntu' }
$Port = if ($env:CLEF_PORT) { $env:CLEF_PORT } else { '8910' }
$BaseUrl = "http://localhost:$Port"

# C:\x\y -> /mnt/c/x/y
$RepoWin = (Resolve-Path $PSScriptRoot).Path.TrimEnd('\')
$RepoWsl = '/mnt/' + $RepoWin.Substring(0, 1).ToLower() + $RepoWin.Substring(2).Replace('\', '/')

function Invoke-Wsl([string]$Script) {
    # A LOGIN shell, so /etc/profile.d hooks from the ROCm-on-WSL install (e.g. LD_LIBRARY_PATH for amdsmi,
    # HSA_ENABLE_DXG_DETECTION) reach the server. Measured: with `bash -c` GPU telemetry was unavailable.
    & wsl.exe -d $Distro -- bash -lc $Script | Out-Host
    return $LASTEXITCODE
}

function Invoke-Launch([string]$Sub) {
    $code = Invoke-Wsl "bash '$RepoWsl/scripts/launch_server.sh' $Sub $Port"
    return $code
}

# WSL terminates a distro (and the detached server with it) ~15 s after the last wsl.exe session exits.
# A hidden `wsl.exe ... sleep infinity` session keeps it alive for as long as the server should run.
$KeepAliveDir = Join-Path $env:LOCALAPPDATA 'clef'
$KeepAlivePid = Join-Path $KeepAliveDir 'keepalive.pid'

function Get-KeepAlive {
    if (-not (Test-Path $KeepAlivePid)) { return $null }
    $id = Get-Content $KeepAlivePid -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $id) { return $null }
    $proc = Get-Process -Id ([int]$id) -ErrorAction SilentlyContinue
    if ($proc -and $proc.ProcessName -eq 'wsl') { return $proc }
    return $null
}

function Start-KeepAlive {
    if (Get-KeepAlive) { return }
    New-Item -ItemType Directory -Force -Path $KeepAliveDir | Out-Null
    $proc = Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d', $Distro, '--exec', 'sleep', 'infinity') `
        -WindowStyle Hidden -PassThru
    Set-Content -Path $KeepAlivePid -Value $proc.Id -Encoding ascii
    Write-Host "WSL keep-alive started (pid $($proc.Id))"
}

function Stop-KeepAlive {
    $proc = Get-KeepAlive
    if ($proc) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "WSL keep-alive stopped (pid $($proc.Id))"
    }
    Remove-Item -Path $KeepAlivePid -ErrorAction SilentlyContinue
}

function Show-Health {
    try {
        $h = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 5
    } catch {
        Write-Host "health: unreachable ($BaseUrl) - $($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }
    $rows = @(
        [pscustomobject]@{ Field = 'status';  Value = $h.status },
        [pscustomobject]@{ Field = 'version'; Value = $h.version },
        [pscustomobject]@{ Field = 'gpu';     Value = "$($h.gpu.name) ($($h.gpu.vram_allocated_gb) / $($h.gpu.vram_total_gb) GB)" },
        [pscustomobject]@{ Field = 'torch';   Value = $h.torch },
        [pscustomobject]@{ Field = 'dtype';   Value = $h.dtype },
        [pscustomobject]@{ Field = 'load_s';  Value = $h.load_seconds },
        [pscustomobject]@{ Field = 'uptime_s'; Value = $h.uptime_s },
        [pscustomobject]@{ Field = 'error';   Value = $h.error }
    )
    $rows | Format-Table -AutoSize | Out-String | Write-Host
    return ($h.status -eq 'ready' -or $h.status -eq 'warming')
}

switch ($Command) {
    'start'   { Start-KeepAlive; exit (Invoke-Launch 'start') }
    'stop'    { $code = Invoke-Launch 'stop'; Stop-KeepAlive; exit $code }
    'restart' { Start-KeepAlive; exit (Invoke-Launch 'restart') }
    'logs'    { exit (Invoke-Launch 'logs') }
    'status' {
        [void](Invoke-Launch 'status')
        $ka = Get-KeepAlive
        if ($ka) { Write-Host "keep-alive: running (pid $($ka.Id))" } else { Write-Host 'keep-alive: not running (WSL may idle-stop the server)' -ForegroundColor Yellow }
        $ok = Show-Health
        if ($ok) { exit 0 } else { exit 1 }
    }
    'doctor' {
        exit (Invoke-Wsl ". '$RepoWsl/scripts/env.sh' && cd '$RepoWsl' && clef_run doctor $($Rest -join ' ')")
    }
    'test' {
        $extra = $Rest -join ' '
        Write-Host '== unit tests (WSL, no GPU) ==' -ForegroundColor Cyan
        $code = Invoke-Wsl ". '$RepoWsl/scripts/env.sh' && cd '$RepoWsl' && python -m pytest tests/unit -q $extra"
        if ($code -ne 0) { exit $code }
        Write-Host "== integration tests against $BaseUrl ==" -ForegroundColor Cyan
        $code = Invoke-Wsl ". '$RepoWsl/scripts/env.sh' && cd '$RepoWsl' && CLEF_URL=http://127.0.0.1:$Port python -m pytest tests/integration -m gpu -q -rs $extra"
        exit $code
    }
    'bench' {
        $http = ($Rest.Count -gt 0 -and $Rest[0] -eq 'http')
        if ($http) {
            $args2 = ($Rest | Select-Object -Skip 1) -join ' '
            exit (Invoke-Wsl ". '$RepoWsl/scripts/env.sh' && cd '$RepoWsl' && CLEF_URL=http://127.0.0.1:$Port clef_run bench $args2")
        }
        Write-Host 'In-process bench needs the GPU: stop the server first (.\clef.ps1 stop). Use "bench http" for the live server.' -ForegroundColor Yellow
        exit (Invoke-Wsl ". '$RepoWsl/scripts/env.sh' && cd '$RepoWsl' && python bench/bench.py $($Rest -join ' ')")
    }
    'open' {
        Start-Process "$BaseUrl/"
    }
    default {
        Write-Host 'usage: .\clef.ps1 start|stop|restart|status|logs|test|bench [http] [args]|doctor|open'
    }
}

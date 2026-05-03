param(
    [Parameter(Mandatory = $true)]
    [string]$Token,
    [int]$Port = 7862,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

if (-not $Python) {
    $localVenv = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    $trexVenv = Join-Path $PSScriptRoot "..\T-Rex\.venv\Scripts\python.exe"
    if (Test-Path $localVenv) {
        $Python = $localVenv
    } elseif (Test-Path $trexVenv) {
        $Python = $trexVenv
    } else {
        $Python = "python"
    }
}

$args = @("app.py", "--trex2_api_token", $Token, "--server_port", $Port)

Push-Location $PSScriptRoot
try {
    & $Python @args
}
finally {
    Pop-Location
}

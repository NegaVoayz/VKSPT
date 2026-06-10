<#
.SYNOPSIS
    Append a timestamped entry to the project dev record document.

.PARAMETER Message
    The log entry text. Lines after the first are indented.

.PARAMETER DocPath
    Path to the dev record file. Auto-detected from git root if omitted.

.EXAMPLE
    ./tools/append_log.ps1 "Fix Vulkan validation error"

.EXAMPLE
    ./tools/append_log.ps1 "Phase 10: Motion blur`n  - Camera velocity`n  - Per-pixel vectors"
#>

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Message,

    [string]$DocPath
)

$ErrorActionPreference = 'Stop'

# Resolve project root from git, falling back to script parent
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (git -C $ScriptDir rev-parse --show-toplevel 2>$null)
if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $ScriptDir
}

# Auto-detect dev record file (avoids encoding issues with Chinese chars in script source)
if (-not $DocPath) {
    $Candidates = Get-ChildItem -Path (Join-Path $ProjectRoot 'docs') -Filter '*开发记录*' -File
    if ($Candidates.Count -eq 0) {
        Write-Error "Dev record not found under $ProjectRoot/docs/"
        exit 1
    }
    $DocPath = $Candidates[0].FullName
}

if (-not (Test-Path $DocPath)) {
    Write-Error "Dev record not found: $DocPath"
    exit 1
}

$Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm'
$Lines = $Message -split "`n"
$Entry = "$Timestamp $($Lines[0])"
for ($i = 1; $i -lt $Lines.Length; $i++) {
    if ($Lines[$i].Trim() -ne '') {
        $Entry += "`n  $($Lines[$i])"
    } else {
        $Entry += "`n"
    }
}

$Existing = Get-Content $DocPath -Raw -Encoding UTF8
if ($Existing -and -not $Existing.EndsWith("`n")) {
    $Entry = "`n$Entry"
}

Add-Content $DocPath -Value $Entry -Encoding UTF8
Write-Host "Appended: $Timestamp $($Lines[0])"

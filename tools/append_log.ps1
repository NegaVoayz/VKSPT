# Append a timestamped entry to the dev record.
param([string]$message)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$doc = Join-Path $scriptDir "..\docs\项目开发记录文档.md" | Resolve-Path
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$entry = "$timestamp " + $message -replace "\\n", "`n"

Add-Content -Path $doc -Value ""
Add-Content -Path $doc -Value $entry
Write-Host "Logged: $timestamp" -ForegroundColor Green

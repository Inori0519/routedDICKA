[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Output = Join-Path $Root "reproduction\plots"
$Tables = Join-Path $Output "tables"

if (Test-Path -LiteralPath $Python -PathType Leaf) {
    $PythonExe = (Resolve-Path -LiteralPath $Python).Path
} else {
    $PythonExe = (Get-Command $Python -ErrorAction Stop).Source
}

New-Item -ItemType Directory -Path $Tables -Force | Out-Null
foreach ($Name in @("results.csv", "ideal_rates.csv", "protocol_envelopes.csv")) {
    Copy-Item -LiteralPath (Join-Path $Root "outputs\tables\$Name") -Destination (Join-Path $Tables $Name)
}

$env:PYTHONDONTWRITEBYTECODE = "1"
& $PythonExe (Join-Path $Root "main\plot_figures.py") --config (Join-Path $Root "config\paper_suite.json") --output $Output
if ($LASTEXITCODE -ne 0) {
    throw "Figure generation failed with exit code $LASTEXITCODE"
}

Write-Host "Figures: $(Join-Path $Output 'figures')"
Write-Host "Published outputs and manuscript figures were not changed."

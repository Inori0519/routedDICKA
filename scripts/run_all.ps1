[CmdletBinding()]
param(
    [string]$Python = "python",
    [ValidateSet("smoke0", "smoke1", "smoke2", "core", "paper")]
    [string]$Profile = "paper",
    [ValidateRange(0, 4)]
    [int]$M = 0,
    [ValidateRange(1, 5)]
    [int]$Level = 2,
    [string]$Output = "",
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $Root "config\paper_suite.json"
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $Root "outputs"
} elseif (-not [System.IO.Path]::IsPathRooted($Output)) {
    $Output = Join-Path $Root $Output
}
$Output = [System.IO.Path]::GetFullPath($Output)

if (Test-Path -LiteralPath $Python -PathType Leaf) {
    $PythonExe = (Resolve-Path -LiteralPath $Python).Path
} else {
    $PythonExe = (Get-Command $Python -ErrorAction Stop).Source
}

$env:PYTHONPATH = (Join-Path $Root "src")

function Format-Command {
    param([string]$Executable, [string[]]$Arguments)
    $quoted = foreach ($item in $Arguments) {
        if ($item -match '[\s"]') {
            '"' + ($item -replace '"', '\"') + '"'
        } else {
            $item
        }
    }
    return ($Executable + " " + ($quoted -join " "))
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host "[$Name]" -ForegroundColor Cyan
    Write-Host (Format-Command -Executable $PythonExe -Arguments $Arguments)
    if ($DryRun) {
        return
    }
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Name' failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Unified routed DICKA suite" -ForegroundColor White
Write-Host "  root:    $Root"
Write-Host "  output:  $Output"
Write-Host "  profile: $Profile"
Write-Host "  NPA:     level $Level"
if ($M -gt 0) {
    Write-Host "  Radau:   m=$M override for experiments that explicitly allow it"
} else {
    Write-Host "  Radau:   per-experiment orders from config"
}
Write-Host "  solver:  MOSEK for numerical profiles only"

Invoke-PythonStep -Name "Lightweight dependency preflight" -Arguments @(
    "-c",
    "import numpy, scipy; print('lightweight dependency preflight: OK')"
)

Invoke-PythonStep -Name "Lightweight unit tests" -Arguments @(
    "-m", "unittest", "discover",
    "-s", (Join-Path $Root "tests"),
    "-p", "test_*.py",
    "-v"
)

Invoke-PythonStep -Name "Protocol and BFF model contract" -Arguments @(
    (Join-Path $Root "verification\verify_model_contract.py"),
    "--config", $Config,
    "--output", (Join-Path $Output "audits\model_contract.json")
)

Invoke-PythonStep -Name "Exact long-only attack and converse" -Arguments @(
    (Join-Path $Root "verification\verify_v21_attack.py"),
    "--config", $Config,
    "--output", (Join-Path $Output "audits\v21_attack.json")
)

Invoke-PythonStep -Name "Independent routed-certificate verification" -Arguments @(
    (Join-Path $Root "verification\verify_v21_certificate.py"),
    "--config", $Config,
    "--output", (Join-Path $Output "audits\v21_routed_certificate.json")
)

Invoke-PythonStep -Name "Primary m=4 routed-certificate verification" -Arguments @(
    (Join-Path $Root "verification\verify_m4_reference_certificate.py"),
    "--config", $Config,
    "--output", (Join-Path $Output "audits\m4_same_point_certificate.json")
)

Invoke-PythonStep -Name "Analytic and strict-separation reference tables" -Arguments @(
    (Join-Path $Root "main\build_reference_data.py"),
    "--config", $Config,
    "--output", $Output
)

Invoke-PythonStep -Name "Supplement proof tables and manifest" -Arguments @(
    (Join-Path $Root "main\export_supplement_tables.py"),
    "--root", $Root
)

if ($Profile -ne "smoke0") {
    Invoke-PythonStep -Name "MOSEK numerical dependency preflight" -Arguments @(
        "-c",
        "import matplotlib, mosek, ncpol2sdpa; print('MOSEK numerical preflight: OK')"
    )

    $ExperimentArguments = @(
        (Join-Path $Root "main\run_experiments.py"),
        "--config", $Config,
        "--profile", $Profile,
        "--output", $Output,
        "--level", "$Level"
    )
    if ($M -gt 0) {
        $ExperimentArguments += @("--m", "$M")
    }
    if ($Resume) {
        $ExperimentArguments += "--resume"
    }
    Invoke-PythonStep -Name "Reduced BFF/NPA experiment suite" -Arguments $ExperimentArguments

    Invoke-PythonStep -Name "Independent numerical audit" -Arguments @(
        (Join-Path $Root "verification\verify_outputs.py"),
        "--config", $Config,
        "--output", $Output
    )

    if ($Profile -eq "paper") {
        Invoke-PythonStep -Name "Dense verified-certificate envelopes" -Arguments @(
            (Join-Path $Root "main\evaluate_witness_envelopes.py"),
            "--config", $Config,
            "--output", $Output
        )

        Invoke-PythonStep -Name "Independent envelope audit" -Arguments @(
            (Join-Path $Root "verification\verify_witness_envelopes.py"),
            "--config", $Config,
            "--output", $Output
        )

        Invoke-PythonStep -Name "Unified manuscript numerical tables" -Arguments @(
            (Join-Path $Root "main\export_manuscript_numerics.py"),
            "--output", $Output,
            "--paper", (Join-Path $Root "paper")
        )

        Invoke-PythonStep -Name "Publication figures and figure data" -Arguments @(
            (Join-Path $Root "main\plot_figures.py"),
            "--config", $Config,
            "--output", $Output
        )

        $FinalAuditArguments = @(
            (Join-Path $Root "verification\verify_outputs.py"),
            "--config", $Config,
            "--output", $Output,
            "--require-figures",
            "--require-scientific-gates"
        )
        Invoke-PythonStep -Name "Final artifact audit" -Arguments $FinalAuditArguments

    }
}

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run complete; no Python command was executed." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "Complete. Audited artifacts are under $Output" -ForegroundColor Green
}

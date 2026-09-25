param(
    [int]$Repeats = 1,
    [int]$Rounds = 3,
    [double[]]$NonIIDAlphas = @(0.5)
)
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_real_fl_matrix.ps1") `
    -Repeats $Repeats -Rounds $Rounds -NonIIDAlphas $NonIIDAlphas `
    -Variants @("full_shadow_no_dedup", "full_shadow_provenance") `
    -AttackScenarios @("shadow_exact_replay")
if ($LASTEXITCODE -ne 0) { throw "Shadow observation paired experiment failed" }

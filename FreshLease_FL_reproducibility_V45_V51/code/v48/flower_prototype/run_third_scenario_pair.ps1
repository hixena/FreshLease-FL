param(
    [int]$Repeats = 1,
    [int]$Rounds = 3,
    [int]$AttackStartRound = 2,
    [double[]]$NonIIDAlphas = @(0.5),
    [switch]$IncludeBackdoor
)

$ErrorActionPreference = "Stop"
if ($AttackStartRound -lt 2 -or $AttackStartRound -gt $Rounds) {
    throw "AttackStartRound must be between 2 and Rounds for delayed betrayal"
}
$scriptDir = $PSScriptRoot
$runId = (New-Guid).ToString("N").Substring(0, 6)
$resultDir = Join-Path $scriptDir "real_fl_results-$(Get-Date -Format 'yyyyMMdd-HHmmss')-third-$runId"
$scenarios = @("diverse_then_repeat_farming")
if ($IncludeBackdoor) { $scenarios += "diverse_then_repeat_backdoor" }

& (Join-Path $scriptDir "run_real_fl_matrix.ps1") `
    -Repeats $Repeats -Rounds $Rounds -AttackStartRound $AttackStartRound `
    -NonIIDAlphas $NonIIDAlphas -Variants @("full", "no_online_revalidation") `
    -AttackScenarios $scenarios -ResultDirectory $resultDir
if (-not $?) { throw "Delayed-betrayal Flower run failed" }

docker run --rm `
    --mount "type=bind,source=$resultDir,target=/results" `
    node-access-flower-prototype:0.19 `
    python -m flower_prototype.analyze_third_scenario `
    /results/real_fl_node_results.csv /results/real_fl_round_node_events.csv `
    /results/real_fl_round_metrics.csv /results/third_scenario_run_results.csv `
    /results/third_scenario_paired_effects.csv $AttackStartRound
if ($LASTEXITCODE -ne 0) { throw "Delayed-betrayal stage analysis failed" }
Write-Host "Delayed-betrayal results: $resultDir"

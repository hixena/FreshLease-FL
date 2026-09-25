param(
    [int]$Repeats = 1,
    [int]$Rounds = 6,
    [int]$AttackStartRound = 2,
    [int]$LimitedCleanUpdates = 4,
    [double]$LimitedAggregationWeight = 0.25,
    [double]$CumulativeRiskDecay = 0.50,
    [double]$CumulativeRiskThreshold = 0.90,
    [double]$SelfReversalGate = 0.20,
    [double[]]$NonIIDAlphas = @(0.5),
    [switch]$IncludeBackdoor
)

$ErrorActionPreference = "Stop"
if ($AttackStartRound -lt 2 -or $AttackStartRound -gt $Rounds) {
    throw "AttackStartRound must be between 2 and Rounds"
}
if ($LimitedCleanUpdates -lt 2) {
    throw "Use at least two clean updates so LIMITED is a real intermediate state"
}
$scriptDir = $PSScriptRoot
$runId = (New-Guid).ToString("N").Substring(0, 6)
$resultDir = Join-Path $scriptDir "real_fl_results-$(Get-Date -Format 'yyyyMMdd-HHmmss')-progressive-$runId"
$scenarios = @("gradual_drift_betrayal")
if ($IncludeBackdoor) { $scenarios += "diverse_then_repeat_backdoor" }

& (Join-Path $scriptDir "run_real_fl_matrix.ps1") `
    -Repeats $Repeats -Rounds $Rounds -AttackStartRound $AttackStartRound `
    -LimitedCleanUpdates $LimitedCleanUpdates -NonIIDAlphas $NonIIDAlphas `
    -LimitedAggregationWeight $LimitedAggregationWeight `
    -CumulativeRiskDecay $CumulativeRiskDecay `
    -CumulativeRiskThreshold $CumulativeRiskThreshold `
    -SelfReversalGate $SelfReversalGate `
    -Variants @("progressive_full", "progressive_no_cumulative", "full", "no_online_revalidation") `
    -AttackScenarios $scenarios -ResultDirectory $resultDir
if (-not $?) { throw "Progressive-access Flower run failed" }

docker run --rm `
    --mount "type=bind,source=$resultDir,target=/results" `
    node-access-flower-prototype:0.19 `
    python -m flower_prototype.analyze_progressive_third_scenario `
    /results/real_fl_node_results.csv /results/real_fl_round_node_events.csv `
    /results/real_fl_round_metrics.csv /results/progressive_run_results.csv `
    /results/progressive_comparison.csv $AttackStartRound
if ($LASTEXITCODE -ne 0) { throw "Progressive-access analysis failed" }
Write-Host "Progressive-access results: $resultDir"

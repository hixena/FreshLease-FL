param(
    [int]$Repeats = 3,
    [int]$Rounds = 10,
    [int]$DriftStartRound = 2,
    [int]$DriftTransitionRounds = 4,
    [int]$LimitedCleanUpdates = 4,
    [double]$LimitedAggregationWeight = 0.25,
    [double]$CumulativeRiskDecay = 0.50,
    [double]$CumulativeRiskThreshold = 0.90,
    [double]$SelfReversalGate = 0.20,
    [double[]]$NonIIDAlphas = @(0.1, 0.5, 10.0)
)

$ErrorActionPreference = "Stop"
if ($DriftStartRound -lt 2 -or $DriftStartRound -gt $Rounds) {
    throw "DriftStartRound must be between 2 and Rounds"
}
if ($DriftTransitionRounds -lt 1) {
    throw "DriftTransitionRounds must be positive"
}
$scriptDir = $PSScriptRoot
$runId = (New-Guid).ToString("N").Substring(0, 6)
$resultDir = Join-Path $scriptDir "real_fl_results-$(Get-Date -Format 'yyyyMMdd-HHmmss')-benign-drift-$runId"
$previousTransition = $env:BENIGN_DRIFT_TRANSITION_ROUNDS
$env:BENIGN_DRIFT_TRANSITION_ROUNDS = "$DriftTransitionRounds"

try {
    & (Join-Path $scriptDir "run_real_fl_matrix.ps1") `
        -Repeats $Repeats -Rounds $Rounds -AttackStartRound $DriftStartRound `
        -LimitedCleanUpdates $LimitedCleanUpdates -NonIIDAlphas $NonIIDAlphas `
        -LimitedAggregationWeight $LimitedAggregationWeight `
        -CumulativeRiskDecay $CumulativeRiskDecay `
        -CumulativeRiskThreshold $CumulativeRiskThreshold `
        -SelfReversalGate $SelfReversalGate `
        -Variants @("progressive_full", "progressive_no_cumulative", "full", "no_online_revalidation") `
        -AttackScenarios @("benign_concept_drift") -ResultDirectory $resultDir
    if (-not $?) { throw "Benign concept-drift Flower run failed" }

    docker run --rm `
        --mount "type=bind,source=$resultDir,target=/results" `
        node-access-flower-prototype:0.19 `
        python -m flower_prototype.analyze_benign_drift_control `
        /results/real_fl_node_results.csv /results/real_fl_round_node_events.csv `
        /results/real_fl_round_metrics.csv /results/benign_drift_run_results.csv `
        /results/benign_drift_summary.csv $DriftStartRound
    if ($LASTEXITCODE -ne 0) { throw "Benign concept-drift analysis failed" }
}
finally {
    if ($previousTransition) {
        $env:BENIGN_DRIFT_TRANSITION_ROUNDS = $previousTransition
    } else {
        Remove-Item "Env:BENIGN_DRIFT_TRANSITION_ROUNDS" -ErrorAction SilentlyContinue
    }
}

Write-Host "Benign concept-drift results: $resultDir"
Write-Host "Per-run control results: $(Join-Path $resultDir 'benign_drift_run_results.csv')"
Write-Host "False-revocation summary: $(Join-Path $resultDir 'benign_drift_summary.csv')"

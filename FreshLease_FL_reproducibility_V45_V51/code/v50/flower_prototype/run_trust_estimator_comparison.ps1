param(
    [int]$Repeats = 1,
    [int]$Rounds = 3,
    [int]$AttackStartRound = 2,
    [double[]]$NonIIDAlphas = @(0.5),
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$runId = (New-Guid).ToString("N").Substring(0, 6)
$resultDir = Join-Path $scriptDir "real_fl_results-$(Get-Date -Format 'yyyyMMdd-HHmmss')-estimators-$runId"

& (Join-Path $scriptDir "run_real_fl_matrix.ps1") `
    -Repeats $Repeats -Rounds $Rounds -AttackStartRound $AttackStartRound `
    -LimitedCleanUpdates 3 -LimitedAggregationWeight 0.75 `
    -CumulativeRiskDecay 0.50 -CumulativeRiskThreshold 0.72 `
    -SelfReversalGate 0.20 -NonIIDAlphas $NonIIDAlphas `
    -TrustEstimators @("dirichlet_mean", "beta_mean") `
    -Variants @("progressive_full") `
    -AttackScenarios @(
        "single_type_farming",
        "diverse_then_repeat_farming",
        "attestation_camouflage"
    ) -ResultDirectory $resultDir -SkipBuild:$SkipBuild
if (-not $?) { throw "Trust-estimator Flower comparison failed" }

docker run --rm `
    --mount "type=bind,source=$resultDir,target=/results" `
    node-access-flower-prototype:0.19 `
    python -m flower_prototype.analyze_trust_estimator_comparison `
    /results/real_fl_node_results.csv `
    /results/trust_estimator_run_results.csv `
    /results/trust_estimator_summary.csv `
    /results/trust_estimator_paired_effects.csv
if ($LASTEXITCODE -ne 0) { throw "Trust-estimator analysis failed" }

Write-Host "Trust-estimator comparison results: $resultDir"
Write-Host "Primary summary: $(Join-Path $resultDir 'trust_estimator_summary.csv')"
Write-Host "Paired effects: $(Join-Path $resultDir 'trust_estimator_paired_effects.csv')"

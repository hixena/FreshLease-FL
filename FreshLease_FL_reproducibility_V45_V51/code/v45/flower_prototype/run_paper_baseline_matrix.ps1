param(
    [int]$Repeats = 1,
    [int]$Rounds = 3,
    [int]$AttackStartRound = 2,
    [double[]]$NonIIDAlphas = @(0.5),
    [string[]]$Datasets = @("digits"),
    [switch]$SkipBuild,
    [string]$ResultDirectory = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $scriptDir "run_real_fl_matrix.ps1"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resolvedResultDirectory = if ($ResultDirectory) {
    $ResultDirectory
} else {
    Join-Path $scriptDir "paper_baseline_results-$stamp"
}
$arguments = @{
    Repeats = $Repeats
    Rounds = $Rounds
    AttackStartRound = $AttackStartRound
    LimitedCleanUpdates = 3
    LimitedAggregationWeight = 0.75
    CumulativeRiskDecay = 0.50
    CumulativeRiskThreshold = 0.72
    SelfReversalGate = 0.20
    NonIIDAlphas = $NonIIDAlphas
    Datasets = $Datasets
    TrustEstimators = @("dirichlet_lcb")
    Variants = @(
        "no_online_revalidation",
        "full",
        "progressive_no_cumulative",
        "progressive_full"
    )
    AttackScenarios = @(
        "gradual_drift_betrayal",
        "benign_concept_drift"
    )
    ResultDirectory = $resolvedResultDirectory
}
if ($SkipBuild) { $arguments.SkipBuild = $true }

Write-Host "Paper baseline matrix: datasets=$($Datasets -join ',') repeats=$Repeats rounds=$Rounds"
& $runner @arguments
if ($LASTEXITCODE -ne 0) { throw "Underlying real-FL matrix failed" }

python (Join-Path $scriptDir "analyze_paper_baselines.py") `
    (Join-Path $resolvedResultDirectory "real_fl_node_results.csv") `
    (Join-Path $resolvedResultDirectory "real_fl_round_node_events.csv") `
    (Join-Path $resolvedResultDirectory "real_fl_round_metrics.csv") `
    (Join-Path $resolvedResultDirectory "paper_baseline_runs.csv") `
    (Join-Path $resolvedResultDirectory "paper_baseline_summary.csv") `
    $AttackStartRound
if ($LASTEXITCODE -ne 0) { throw "Paper baseline summary generation failed" }
Write-Host "Paper baseline summary: $(Join-Path $resolvedResultDirectory 'paper_baseline_summary.csv')"

param(
    [int]$Repeats = 1,
    [int]$Rounds = 3,
    [int]$AttackStartRound = 2,
    [double[]]$NonIIDAlphas = @(0.5),
    [string[]]$Datasets = @("fashion_mnist"),
    [int]$ClientCount = 10,
    [int]$TaskDeadlineSeconds = 30,
    [int]$BarrierTimeoutSeconds = 600,
    [int]$ClientJoinTimeoutSeconds = 300,
    [string[]]$Methods = @("no_online_revalidation", "fltrust", "trimmed_mean", "progressive_full"),
    [string[]]$AttackScenarios = @("gradual_drift_betrayal", "benign_concept_drift"),
    [switch]$SkipBuild,
    [string]$ResultDirectory = ""
)

$ErrorActionPreference = "Stop"
$allowedMethods = @(
    "no_online_revalidation", "full", "fltrust", "trimmed_mean",
    "coordinate_median", "progressive_no_cumulative", "progressive_full"
)
foreach ($method in $Methods) {
    if ($method -notin $allowedMethods) { throw "Unknown V35 method: $method" }
}
$allowedAttackScenarios = @(
    "gradual_drift_betrayal", "benign_concept_drift",
    "diverse_then_repeat_backdoor"
)
foreach ($scenario in $AttackScenarios) {
    if ($scenario -notin $allowedAttackScenarios) {
        throw "Unknown V35 attack scenario: $scenario"
    }
}
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $scriptDir "run_real_fl_matrix.ps1"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resolvedResultDirectory = if ($ResultDirectory) {
    $ResultDirectory
} else {
    Join-Path $scriptDir "external_baseline_results-$stamp"
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
    ClientCount = $ClientCount
    TaskDeadlineSeconds = $TaskDeadlineSeconds
    BarrierTimeoutSeconds = $BarrierTimeoutSeconds
    ClientJoinTimeoutSeconds = $ClientJoinTimeoutSeconds
    TrustEstimators = @("dirichlet_lcb")
    Variants = $Methods
    AttackScenarios = $AttackScenarios
    ResultDirectory = $resolvedResultDirectory
}
if ($SkipBuild) { $arguments.SkipBuild = $true }

Write-Host "V35 external baselines: clients=$ClientCount methods=$($Methods -join ',') datasets=$($Datasets -join ',') scenarios=$($AttackScenarios -join ',')"
& $runner @arguments
if ($LASTEXITCODE -ne 0) { throw "V35 external-baseline matrix failed" }

python (Join-Path $scriptDir "analyze_paper_baselines.py") `
    (Join-Path $resolvedResultDirectory "real_fl_node_results.csv") `
    (Join-Path $resolvedResultDirectory "real_fl_round_node_events.csv") `
    (Join-Path $resolvedResultDirectory "real_fl_round_metrics.csv") `
    (Join-Path $resolvedResultDirectory "external_baseline_runs.csv") `
    (Join-Path $resolvedResultDirectory "external_baseline_summary.csv") `
    $AttackStartRound
if ($LASTEXITCODE -ne 0) { throw "V35 external-baseline summary generation failed" }
Write-Host "V35 summary: $(Join-Path $resolvedResultDirectory 'external_baseline_summary.csv')"

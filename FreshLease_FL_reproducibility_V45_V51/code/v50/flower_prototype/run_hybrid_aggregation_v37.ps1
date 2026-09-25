param(
    [int]$Repeats = 5,
    [int]$Rounds = 20,
    [int]$AttackStartRound = 2,
    [double[]]$NonIIDAlphas = @(0.1, 0.5, 10.0),
    [string[]]$Datasets = @("fashion_mnist"),
    [int]$ClientCount = 10,
    [ValidateSet("synchronous_cold_start", "mixed_maturity")]
    [string]$CohortMode = "mixed_maturity",
    [int]$TaskDeadlineSeconds = 30,
    [int]$BarrierTimeoutSeconds = 600,
    [int]$ClientJoinTimeoutSeconds = 300,
    [string[]]$Methods = @(
        "no_online_revalidation", "trimmed_mean",
        "progressive_full", "progressive_trimmed_mean"
    ),
    [string[]]$AttackScenarios = @(
        "diverse_then_repeat_backdoor", "benign_concept_drift"
    ),
    [switch]$SkipBuild,
    [string]$ResultDirectory = ""
)

$ErrorActionPreference = "Stop"
$allowedMethods = @(
    "no_online_revalidation", "full", "fltrust", "trimmed_mean",
    "coordinate_median", "progressive_no_cumulative", "progressive_full",
    "progressive_fltrust", "progressive_trimmed_mean"
)
foreach ($method in $Methods) {
    if ($method -notin $allowedMethods) { throw "Unknown V37 method: $method" }
}
$allowedAttackScenarios = @(
    "gradual_drift_betrayal", "benign_concept_drift",
    "diverse_then_repeat_backdoor"
)
foreach ($scenario in $AttackScenarios) {
    if ($scenario -notin $allowedAttackScenarios) {
        throw "Unknown V37 attack scenario: $scenario"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $scriptDir "run_real_fl_matrix.ps1"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resolvedResultDirectory = if ($ResultDirectory) {
    $ResultDirectory
} else {
    Join-Path $scriptDir "hybrid_aggregation_v37_results-$stamp"
}

# V37 changes the experiment matrix and cohort initial condition only.  The
# selected V34/V36 access-control working point remains frozen.
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
    CohortMode = $CohortMode
    TaskDeadlineSeconds = $TaskDeadlineSeconds
    BarrierTimeoutSeconds = $BarrierTimeoutSeconds
    ClientJoinTimeoutSeconds = $ClientJoinTimeoutSeconds
    TrustEstimators = @("dirichlet_lcb")
    Variants = $Methods
    AttackScenarios = $AttackScenarios
    ResultDirectory = $resolvedResultDirectory
}
if ($SkipBuild) { $arguments.SkipBuild = $true }

Write-Host "V37 robustness matrix: clients=$ClientCount cohort=$CohortMode methods=$($Methods -join ',') datasets=$($Datasets -join ',') scenarios=$($AttackScenarios -join ',') alphas=$($NonIIDAlphas -join ',') repeats=$Repeats rounds=$Rounds"
& $runner @arguments
if ($LASTEXITCODE -ne 0) { throw "V37 robustness matrix failed" }

$projectRoot = Split-Path -Parent $scriptDir
Push-Location $projectRoot
try {
    python -m flower_prototype.analyze_hybrid_v37 `
        (Join-Path $resolvedResultDirectory "real_fl_node_results.csv") `
        (Join-Path $resolvedResultDirectory "real_fl_round_node_events.csv") `
        (Join-Path $resolvedResultDirectory "real_fl_round_metrics.csv") `
        (Join-Path $resolvedResultDirectory "hybrid_aggregation_v37_runs.csv") `
        (Join-Path $resolvedResultDirectory "hybrid_aggregation_v37_summary.csv") `
        $AttackStartRound
} finally {
    Pop-Location
}
if ($LASTEXITCODE -ne 0) { throw "V37 summary generation failed" }
Write-Host "V37 summary: $(Join-Path $resolvedResultDirectory 'hybrid_aggregation_v37_summary.csv')"

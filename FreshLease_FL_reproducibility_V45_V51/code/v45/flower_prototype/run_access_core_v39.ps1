# Historical V39 entry point retained for reproducibility inside the V40 package.
param(
    [ValidateSet("core", "estimators")]
    [string]$Stage = "core",
    [int]$Repeats = 5,
    [int]$Rounds = 5,
    [double[]]$NonIIDAlphas = @(0.5),
    [string[]]$Datasets = @("fashion_mnist"),
    [int]$ClientCount = 10,
    [int]$TaskDeadlineSeconds = 30,
    [int]$BarrierTimeoutSeconds = 600,
    [int]$ClientJoinTimeoutSeconds = 300,
    [switch]$SkipBuild,
    [string]$ResultDirectory = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$runner = Join-Path $scriptDir "run_real_fl_matrix.ps1"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resolvedResultDirectory = if ($ResultDirectory) {
    $ResultDirectory
} else {
    Join-Path $scriptDir "access_core_v39_results-$Stage-$stamp"
}

$scenarios = @(
    "single_type_farming",
    "false_quality_reporting",
    "attestation_camouflage",
    "diverse_then_repeat_farming"
)

if ($Stage -eq "core") {
    $variants = @(
        "access_full",
        "access_no_diversity",
        "access_no_repeat_decay",
        "access_no_result_verification",
        "access_no_limited",
        "access_naive"
    )
    $estimators = @("dirichlet_lcb")
} else {
    # Estimator sensitivity is deliberately separate from the mechanism
    # ablation so the same comparison is not counted as additional methods.
    $variants = @("access_full")
    $estimators = @("dirichlet_lcb", "dirichlet_mean", "beta_mean")
}

$arguments = @{
    Repeats = $Repeats
    Rounds = $Rounds
    AttackStartRound = 2
    LimitedCleanUpdates = 3
    LimitedAggregationWeight = 0.75
    NormEscalationMode = "moderate_shadow"
    NonIIDAlphas = $NonIIDAlphas
    Datasets = $Datasets
    ClientCount = $ClientCount
    CohortMode = "synchronous_cold_start"
    TaskDeadlineSeconds = $TaskDeadlineSeconds
    BarrierTimeoutSeconds = $BarrierTimeoutSeconds
    ClientJoinTimeoutSeconds = $ClientJoinTimeoutSeconds
    TrustEstimators = $estimators
    Variants = $variants
    AttackScenarios = $scenarios
    ResultDirectory = $resolvedResultDirectory
}
if ($SkipBuild) { $arguments.SkipBuild = $true }

$configurationCount = (
    $variants.Count * $estimators.Count * $scenarios.Count *
    $NonIIDAlphas.Count * $Datasets.Count * $Repeats
)
Write-Host "V39 access-core stage=$Stage configurations=$configurationCount rounds=$Rounds clients=$ClientCount"
Write-Host "Primary outcomes: initial access state, training eligibility, onboarding cost, LIMITED exposure"

& $runner @arguments
if ($LASTEXITCODE -ne 0) { throw "V39 access-core matrix failed" }

Push-Location $projectRoot
try {
    python -m flower_prototype.analyze_access_v39 `
        (Join-Path $resolvedResultDirectory "real_fl_node_results.csv") `
        (Join-Path $resolvedResultDirectory "access_core_v39_runs.csv") `
        (Join-Path $resolvedResultDirectory "access_core_v39_summary.csv")
    if ($LASTEXITCODE -ne 0) { throw "V39 access-core summary generation failed" }
} finally {
    Pop-Location
}

Write-Host "Completed V39 access-core results: $resolvedResultDirectory"
Write-Host "Run-level access decisions: $(Join-Path $resolvedResultDirectory 'access_core_v39_runs.csv')"
Write-Host "Cluster-aware summary: $(Join-Path $resolvedResultDirectory 'access_core_v39_summary.csv')"

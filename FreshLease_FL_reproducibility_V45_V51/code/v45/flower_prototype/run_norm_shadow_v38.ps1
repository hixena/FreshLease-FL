param(
    [int]$Repeats = 5,
    [int]$Rounds = 20,
    [int]$AttackStartRound = 2,
    [double[]]$NonIIDAlphas = @(0.1),
    [string[]]$Datasets = @("fashion_mnist"),
    [int]$ClientCount = 10,
    [ValidateSet("synchronous_cold_start", "mixed_maturity")]
    [string]$CohortMode = "mixed_maturity",
    [int]$TaskDeadlineSeconds = 30,
    [int]$BarrierTimeoutSeconds = 600,
    [int]$ClientJoinTimeoutSeconds = 300,
    [string[]]$Methods = @(
        "progressive_full", "progressive_trimmed_mean"
    ),
    [string[]]$AttackScenarios = @(
        "diverse_then_repeat_backdoor", "benign_concept_drift"
    ),
    [switch]$SkipBuild,
    [string]$ResultDirectory = ""
)

$ErrorActionPreference = "Stop"
$allowedMethods = @("progressive_full", "progressive_trimmed_mean")
foreach ($method in $Methods) {
    if ($method -notin $allowedMethods) {
        throw "V38 shadow diagnosis accepts only progressive methods: $method"
    }
}
$allowedScenarios = @("diverse_then_repeat_backdoor", "benign_concept_drift")
foreach ($scenario in $AttackScenarios) {
    if ($scenario -notin $allowedScenarios) {
        throw "Unknown V38 shadow scenario: $scenario"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$runner = Join-Path $scriptDir "run_real_fl_matrix.ps1"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resolvedResultDirectory = if ($ResultDirectory) {
    $ResultDirectory
} else {
    Join-Path $scriptDir "norm_shadow_v38_results-$stamp"
}

# V38 is diagnostic.  It freezes the selected V34/V37 working point and changes
# only repeated finite ordinary norm excesses from hard revocation to shadow
# rejection.  Extreme and non-finite updates remain hard failures.
$arguments = @{
    Repeats = $Repeats
    Rounds = $Rounds
    AttackStartRound = $AttackStartRound
    LimitedCleanUpdates = 3
    LimitedAggregationWeight = 0.75
    CumulativeRiskDecay = 0.50
    CumulativeRiskThreshold = 0.72
    SelfReversalGate = 0.20
    NormEscalationMode = "moderate_shadow"
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

Write-Host "V38 norm shadow: clients=$ClientCount methods=$($Methods -join ',') scenarios=$($AttackScenarios -join ',') alphas=$($NonIIDAlphas -join ',') repeats=$Repeats rounds=$Rounds"
& $runner @arguments
if ($LASTEXITCODE -ne 0) { throw "V38 norm-shadow matrix failed" }

Push-Location $projectRoot
try {
    python -m flower_prototype.analyze_hybrid_v38 `
        (Join-Path $resolvedResultDirectory "real_fl_node_results.csv") `
        (Join-Path $resolvedResultDirectory "real_fl_round_node_events.csv") `
        (Join-Path $resolvedResultDirectory "real_fl_round_metrics.csv") `
        (Join-Path $resolvedResultDirectory "hybrid_aggregation_v38_runs.csv") `
        (Join-Path $resolvedResultDirectory "hybrid_aggregation_v38_summary.csv") `
        $AttackStartRound
    if ($LASTEXITCODE -ne 0) { throw "V38 hybrid summary generation failed" }

    python -m flower_prototype.analyze_norm_shadow_v38 `
        (Join-Path $resolvedResultDirectory "real_fl_node_results.csv") `
        (Join-Path $resolvedResultDirectory "real_fl_round_node_events.csv") `
        (Join-Path $resolvedResultDirectory "norm_shadow_v38_nodes.csv") `
        (Join-Path $resolvedResultDirectory "norm_shadow_v38_summary.csv")
    if ($LASTEXITCODE -ne 0) { throw "V38 norm-shadow diagnosis failed" }
} finally {
    Pop-Location
}

Write-Host "Completed V38 diagnostic results: $resolvedResultDirectory"
Write-Host "Hybrid summary: $(Join-Path $resolvedResultDirectory 'hybrid_aggregation_v38_summary.csv')"
Write-Host "Norm-shadow summary: $(Join-Path $resolvedResultDirectory 'norm_shadow_v38_summary.csv')"

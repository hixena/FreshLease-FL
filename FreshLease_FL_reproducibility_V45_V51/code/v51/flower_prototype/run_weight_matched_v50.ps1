param(
    [ValidateSet("smoke", "formal")]
    [string]$Stage = "smoke",
    [int]$TaskDeadlineSeconds = 30,
    [int]$BarrierTimeoutSeconds = 2400,
    [int]$ClientJoinTimeoutSeconds = 1200,
    [int]$RoundProgressPollSeconds = 5,
    [string]$V49ResultDirectory = "",
    [string]$ResultDirectory = "",
    [switch]$Resume,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$runner = Join-Path $scriptDir "run_real_fl_matrix.ps1"
$compose = Join-Path $scriptDir "docker-compose.evidence-farming.yml"
if (-not (Test-Path (Join-Path $scriptDir "data\cifar10.npz"))) {
    throw "Missing CIFAR-10 data. Copy flower_prototype\data\cifar10.npz from V49 or run prepare_cifar10.py"
}
$rounds = if ($Stage -eq "smoke") { 16 } else { 50 }
$repeats = if ($Stage -eq "smoke") { 1 } else { 10 }
$total = 2 * $repeats
if ($Stage -eq "formal") {
    if (-not $V49ResultDirectory) { throw "Formal V50 requires -V49ResultDirectory pointing to the completed V49 formal root" }
    $referenceRoot = (Resolve-Path $V49ResultDirectory -ErrorAction Stop).Path
    foreach ($name in @(
        "post_promotion_v49_runs.csv", "post_promotion_v49_promotion_guard.csv",
        "real_fl_node_results.csv", "real_fl_round_node_events.csv"
    )) {
        if (-not (Test-Path (Join-Path $referenceRoot $name))) {
            throw "V49 reference is missing $name in $referenceRoot"
        }
    }
}
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$root = if ($ResultDirectory) { $ResultDirectory } else {
    Join-Path $scriptDir "weight_matched_v50_results-$Stage-$stamp"
}
if (-not [IO.Path]::IsPathRooted($root)) { $root = Join-Path (Get-Location).Path $root }
$root = [IO.Path]::GetFullPath($root)
if (Test-Path $root) {
    if (-not $Resume) { throw "Result directory already exists: $root; use -Resume to reuse it" }
} else {
    New-Item -ItemType Directory -Path $root | Out-Null
}
$batch = Join-Path $root "batch-01-weight-matched-n20-k2-cifar10-a0p5"
$required = @(
    "real_fl_node_results.csv", "real_fl_round_node_events.csv",
    "real_fl_round_metrics.csv", "real_fl_configuration_metrics.csv"
)
$reuse = $false
if ($Resume -and (Test-Path $batch)) {
    $complete = @($required | Where-Object { -not (Test-Path (Join-Path $batch $_)) }).Count -eq 0
    if ($complete) {
        $configs = @(Import-Csv (Join-Path $batch "real_fl_configuration_metrics.csv"))
        $nodes = @(Import-Csv (Join-Path $batch "real_fl_node_results.csv"))
        $reuse = ($configs.Count -eq $total -and $nodes.Count -eq 20 * $total -and
            @($nodes | Where-Object { $_.attack_start_round -ne "6" }).Count -eq 0)
    }
    if (-not $reuse) {
        $batch = Join-Path $root "batch-01-weight-matched-n20-k2-cifar10-a0p5-retry-$stamp"
        Write-Host "Previous batch incomplete; running a fresh batch: $batch" -ForegroundColor Yellow
    }
}
Write-Host "V50 matched-weight control: $Stage $total configurations, $rounds rounds (CIFAR-10, 20 clients, 2 newcomers)" -ForegroundColor Cyan
Write-Host "Constant newcomer weight 23/30 after round 5; baseline V49 lease L5/O5"
# Build even on resume: the analysis module may have changed after the matrix finished.
if (-not $SkipBuild) {
    Push-Location $projectRoot
    try { docker compose -f $compose build }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw "V50 Docker build failed" }
}
if (-not $reuse) {
    $matrixArguments = @{
        Repeats=$repeats; Rounds=$rounds; AttackStartRound=6
        LimitedObservationUpdates=5; LimitedAggregationWeight=0.5
        AccessLeaseFullUpdates=5; MaxProbationTasks=20
        MinProbationCompletedTasks=9; MinEvidenceMass=1.5
        NormEscalationMode="moderate_shadow"; NonIIDAlphas=@(0.5)
        Datasets=@("cifar10"); ClientCount=20; TargetCount=2
        UseGenericAttackers=$true; CohortMode="mixed_maturity"
        TrustEstimators=@("dirichlet_lcb"); Variants=@("access_weight_matched")
        AttackScenarios=@("diverse_then_repeat_backdoor", "benign_concept_drift")
        ResultDirectory=$batch; MaxLocalTrainSamples=512
        ValidationSamplesPerClass=100; TestSamplesPerClass=200
        TaskDeadlineSeconds=$TaskDeadlineSeconds
        BarrierTimeoutSeconds=$BarrierTimeoutSeconds
        ClientJoinTimeoutSeconds=$ClientJoinTimeoutSeconds
        RoundProgressPollSeconds=$RoundProgressPollSeconds
        ProgressTotal=$total; ProgressActivity="V50 matched-weight matrix"
        SkipBuild=$true
    }
    & $runner @matrixArguments
    if ($LASTEXITCODE -ne 0) { throw "V50 matched-weight matrix failed" }
} else {
    Write-Host "Resume: reusing complete matrix from $batch" -ForegroundColor Cyan
}
foreach ($name in $required) {
    Copy-Item (Join-Path $batch $name) (Join-Path $root $name) -Force
}
$resolvedRoot = (Resolve-Path $root).Path
$analysisArgs = @(
    "python", "-m", "flower_prototype.analyze_weight_matched_v50", $Stage,
    "/results/real_fl_node_results.csv",
    "/results/real_fl_round_node_events.csv",
    "/results/real_fl_round_metrics.csv",
    "/results/real_fl_configuration_metrics.csv",
    "/results/weight_matched_v50_runs.csv",
    "/results/weight_matched_v50_summary.csv",
    "/results/weight_matched_v50_promotion_guard.csv",
    "/results/weight_matched_v50_paired_runs.csv",
    "/results/weight_matched_v50_paired_summary.csv", "$rounds"
)
$mounts = @("--mount", "type=bind,source=$resolvedRoot,target=/results")
if ($Stage -eq "formal") {
    $mounts += @("--mount", "type=bind,source=$referenceRoot,target=/reference,readonly")
    $analysisArgs += @(
        "/reference/post_promotion_v49_runs.csv",
        "/reference/post_promotion_v49_promotion_guard.csv",
        "/reference/real_fl_node_results.csv",
        "/reference/real_fl_round_node_events.csv"
    )
}
docker run --rm @mounts node-access-flower-prototype:0.23 @analysisArgs
if ($LASTEXITCODE -ne 0) { throw "V50 summary or V49 paired analysis failed" }
Write-Progress -Activity "V50 matched-weight matrix" -Completed
Write-Host "Completed V50: $root" -ForegroundColor Green
Write-Host "Audit: $(Join-Path $root 'weight_matched_v50_promotion_guard.csv')"
if ($Stage -eq "formal") {
    Write-Host "Paired effects: $(Join-Path $root 'weight_matched_v50_paired_summary.csv')"
}

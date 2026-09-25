param(
    [ValidateSet("smoke", "formal")]
    [string]$Stage = "smoke",
    [int]$Repeats = 5,
    [int]$Rounds = 50,
    [int]$TaskDeadlineSeconds = 30,
    [int]$BarrierTimeoutSeconds = 2400,
    [int]$ClientJoinTimeoutSeconds = 1200,
    [int]$RoundProgressPollSeconds = 5,
    [switch]$SkipBuild,
    [switch]$Resume,
    [string]$ResultDirectory = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$runner = Join-Path $scriptDir "run_real_fl_matrix.ps1"
$datasetFile = Join-Path $scriptDir "data\cifar10.npz"
if (-not (Test-Path $datasetFile)) {
    throw "CIFAR-10 data is missing. Run: python .\flower_prototype\prepare_cifar10.py"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$root = if ($ResultDirectory) { $ResultDirectory } else {
    Join-Path $scriptDir "cifar10_v47_results-$Stage-$stamp"
}
if (-not [IO.Path]::IsPathRooted($root)) {
    $root = Join-Path (Get-Location).Path $root
}
$root = [IO.Path]::GetFullPath($root)
if (Test-Path $root) {
    if (-not $Resume) { throw "Result directory already exists: $root" }
} else {
    New-Item -ItemType Directory -Path $root | Out-Null
}

$effectiveRepeats = if (
    $Stage -eq "smoke" -and -not $PSBoundParameters.ContainsKey("Repeats")
) { 1 } else { $Repeats }
$effectiveRounds = if (
    $Stage -eq "smoke" -and -not $PSBoundParameters.ContainsKey("Rounds")
) { 16 } else { $Rounds }
if ($effectiveRounds -lt 16) {
    throw "V47 requires at least 16 rounds to observe an L5 lease cycle"
}

$scenarios = @("diverse_then_repeat_backdoor", "benign_concept_drift")
$variants = @(
    @{ Name = "access_freshness_lease"; Tag = "lease-L5" },
    @{ Name = "access_full"; Tag = "progressive-no-lease" },
    @{ Name = "rffl_reputation"; Tag = "rffl-style" }
)
$total = $variants.Count * $scenarios.Count * $effectiveRepeats
$started = Get-Date
$offset = 0
$built = $SkipBuild.IsPresent
$batches = @()
$batchNumber = 0

Write-Host "V47 stage=$Stage configurations=$total rounds=$effectiveRounds"
Write-Host "Frozen scope: CIFAR-10, clients=20, attackers=2, alpha=0.5"
Write-Host "Comparators: L5 freshness lease, progressive access without lease, RFFL-style"
Write-Host "Compute controls: 512 local examples/round, 100 validation and 200 test examples/class"

foreach ($variant in $variants) {
    $batchNumber++
    $batchDir = Join-Path $root (
        "batch-{0:D2}-{1}-n20-k2-cifar10-a0p5" -f $batchNumber, $variant.Tag
    )
    $perBatch = $scenarios.Count * $effectiveRepeats
    if ($Resume -and (Test-Path $batchDir)) {
        $required = @(
            "real_fl_node_results.csv", "real_fl_round_node_events.csv",
            "real_fl_round_metrics.csv", "real_fl_configuration_metrics.csv"
        )
        $hasAllFiles = ($required | Where-Object {
            -not (Test-Path (Join-Path $batchDir $_))
        }).Count -eq 0
        $completedConfigurations = 0
        if ($hasAllFiles) {
            $completedConfigurations = @(
                Import-Csv (Join-Path $batchDir "real_fl_configuration_metrics.csv")
            ).Count
        }
        if ($hasAllFiles -and $completedConfigurations -eq $perBatch) {
            Write-Host "Resume: reusing complete batch $batchDir ($perBatch configurations)" -ForegroundColor Cyan
            $batches += $batchDir
            $offset += $perBatch
            continue
        }
        $batchDir = Join-Path $root (
            "batch-{0:D2}-{1}-n20-k2-cifar10-a0p5-retry-{2}" -f `
            $batchNumber, $variant.Tag, $stamp
        )
        Write-Host "Resume: prior batch incomplete; retrying as $batchDir" -ForegroundColor Yellow
    }

    $arguments = @{
        Repeats=$effectiveRepeats; Rounds=$effectiveRounds; AttackStartRound=2
        LimitedObservationUpdates=5; LimitedAggregationWeight=0.50
        AccessLeaseFullUpdates=5
        MaxProbationTasks=20; MinProbationCompletedTasks=9; MinEvidenceMass=1.50
        NormEscalationMode="moderate_shadow"; NonIIDAlphas=@(0.5)
        Datasets=@("cifar10"); ClientCount=20; TargetCount=2
        UseGenericAttackers=$true; CohortMode="mixed_maturity"
        TrustEstimators=@("dirichlet_lcb"); Variants=@($variant.Name)
        AttackScenarios=$scenarios; ResultDirectory=$batchDir
        MaxLocalTrainSamples=512
        ValidationSamplesPerClass=100; TestSamplesPerClass=200
        TaskDeadlineSeconds=$TaskDeadlineSeconds
        BarrierTimeoutSeconds=$BarrierTimeoutSeconds
        ClientJoinTimeoutSeconds=$ClientJoinTimeoutSeconds
        ProgressOffset=$offset; ProgressTotal=$total; ProgressStartedAt=$started
        ProgressActivity="V47 CIFAR-10 matrix"
        RoundProgressPollSeconds=$RoundProgressPollSeconds
    }
    if ($built) { $arguments.SkipBuild = $true }
    & $runner @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "V47 batch failed: $($variant.Name)"
    }
    $built = $true
    $batches += $batchDir
    $offset += $perBatch
}

foreach ($name in @(
    "real_fl_node_results.csv", "real_fl_round_node_events.csv",
    "real_fl_round_metrics.csv", "real_fl_configuration_metrics.csv"
)) {
    $destination = Join-Path $root $name
    $rows = foreach ($batch in $batches) { Import-Csv (Join-Path $batch $name) }
    $rows | Export-Csv -Path $destination -NoTypeInformation -Encoding UTF8
}

$resolvedRoot = (Resolve-Path $root).Path
docker run --rm `
    --mount "type=bind,source=$resolvedRoot,target=/results" `
    node-access-flower-prototype:0.23 `
    python -m flower_prototype.analyze_cifar10_v47 $Stage `
    /results/real_fl_node_results.csv `
    /results/real_fl_round_node_events.csv `
    /results/real_fl_round_metrics.csv `
    /results/real_fl_configuration_metrics.csv `
    /results/cifar10_v47_runs.csv `
    /results/cifar10_v47_summary.csv `
    /results/cifar10_v47_paired_effects.csv `
    /results/cifar10_v47_paired_summary.csv `
    /results/cifar10_v47_round_runs.csv `
    /results/cifar10_v47_round_summary.csv
if ($LASTEXITCODE -ne 0) { throw "V47 CIFAR-10 summary generation failed" }

Write-Progress -Id 41 -Activity "V47 CIFAR-10 matrix" -Completed
Write-Host "Completed V47 stage=$Stage results: $root" -ForegroundColor Green
Write-Host "Summary: $(Join-Path $root 'cifar10_v47_summary.csv')"
Write-Host "Paired summary: $(Join-Path $root 'cifar10_v47_paired_summary.csv')"
Write-Host "Round trajectory: $(Join-Path $root 'cifar10_v47_round_summary.csv')"

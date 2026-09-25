param(
    [ValidateSet("generalization", "sensitivity")]
    [string]$Experiment = "generalization",
    [ValidateSet("smoke", "formal")]
    [string]$Stage = "smoke",
    [int]$Repeats = 5,
    [int]$Rounds = 50,
    [int]$TaskDeadlineSeconds = 30,
    [int]$BarrierTimeoutSeconds = 1800,
    [int]$ClientJoinTimeoutSeconds = 900,
    [int]$RoundProgressPollSeconds = 5,
    [switch]$SkipBuild,
    [switch]$Resume,
    [string]$ResultDirectory = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $scriptDir "run_real_fl_matrix.ps1"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$root = if ($ResultDirectory) { $ResultDirectory } else {
    Join-Path $scriptDir "v46_results-$Experiment-$Stage-$stamp"
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
$defaultSmokeRounds = if ($Experiment -eq "sensitivity") { 21 } else { 16 }
$effectiveRounds = if (
    $Stage -eq "smoke" -and -not $PSBoundParameters.ContainsKey("Rounds")
) { $defaultSmokeRounds } else { $Rounds }
if ($effectiveRounds -lt $defaultSmokeRounds) {
    throw "V46 $Experiment requires at least $defaultSmokeRounds rounds"
}

$scenarios = @("diverse_then_repeat_backdoor", "benign_concept_drift")
$batchSpecs = @()
if ($Experiment -eq "generalization") {
    $conditions = @(
        @{ Dataset = "fashion_mnist"; Alpha = 0.1; Tag = "fashion-a0p1" },
        @{ Dataset = "fashion_mnist"; Alpha = 10.0; Tag = "fashion-a10" },
        @{ Dataset = "digits"; Alpha = 0.5; Tag = "digits-a0p5" }
    )
    $variants = @(
        @{ Name = "access_freshness_lease"; Weight = 0.50; Lease = 5 },
        @{ Name = "access_full"; Weight = 0.50; Lease = 5 }
    )
    foreach ($condition in $conditions) {
        foreach ($variant in $variants) {
            $batchSpecs += @{
                Dataset = $condition.Dataset; Alpha = $condition.Alpha
                Tag = "$($condition.Tag)-$($variant.Name)"
                Variant = $variant.Name; Weight = $variant.Weight
                Lease = $variant.Lease; Scenarios = $scenarios
            }
        }
    }
} else {
    foreach ($lease in @(3, 5, 10)) {
        $batchSpecs += @{
            Dataset = "fashion_mnist"; Alpha = 0.5
            Tag = "fashion-a0p5-access_freshness_lease-L$lease"
            Variant = "access_freshness_lease"; Weight = 0.50
            Lease = $lease; Scenarios = $scenarios
        }
    }
    $batchSpecs += @{
        Dataset = "fashion_mnist"; Alpha = 0.5
        Tag = "fashion-a0p5-access_full"
        Variant = "access_full"; Weight = 0.50
        Lease = 5; Scenarios = $scenarios
    }
}

$total = 0
foreach ($spec in $batchSpecs) {
    $total += $spec.Scenarios.Count * $effectiveRepeats
}
$started = Get-Date
$offset = 0
$built = $SkipBuild.IsPresent
$batches = @()
$batchNumber = 0

Write-Host "V46 experiment=$Experiment stage=$Stage configurations=$total rounds=$effectiveRounds"
Write-Host "Frozen mechanism: V45 freshness lease; no risk-threshold retuning"
if ($Experiment -eq "generalization") {
    Write-Host "New conditions: Fashion-MNIST alpha=0.1/10 and Digits alpha=0.5"
    Write-Host "Comparators: access_freshness_lease vs access_full; backdoor plus benign drift"
} else {
    Write-Host "Lease lengths: 3/5/10 full-weight updates; Fashion-MNIST alpha=0.5"
    Write-Host "Control: access_full; backdoor plus benign drift"
}

foreach ($spec in $batchSpecs) {
    $batchNumber++
    $batchDir = Join-Path $root (
        "batch-{0:D2}-{1}-n20-k2" -f $batchNumber, $spec.Tag
    )
    $perBatch = $spec.Scenarios.Count * $effectiveRepeats
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
            "batch-{0:D2}-{1}-n20-k2-retry-{2}" -f `
            $batchNumber, $spec.Tag, $stamp
        )
        Write-Host "Resume: prior batch incomplete; retrying as $batchDir" -ForegroundColor Yellow
    }

    $arguments = @{
        Repeats=$effectiveRepeats; Rounds=$effectiveRounds; AttackStartRound=2
        LimitedObservationUpdates=5; LimitedAggregationWeight=$spec.Weight
        AccessLeaseFullUpdates=$spec.Lease
        MaxProbationTasks=20; MinProbationCompletedTasks=9; MinEvidenceMass=1.50
        NormEscalationMode="moderate_shadow"; NonIIDAlphas=@($spec.Alpha)
        Datasets=@($spec.Dataset); ClientCount=20; TargetCount=2
        UseGenericAttackers=$true; CohortMode="mixed_maturity"
        TrustEstimators=@("dirichlet_lcb"); Variants=@($spec.Variant)
        AttackScenarios=$spec.Scenarios; ResultDirectory=$batchDir
        TaskDeadlineSeconds=$TaskDeadlineSeconds
        BarrierTimeoutSeconds=$BarrierTimeoutSeconds
        ClientJoinTimeoutSeconds=$ClientJoinTimeoutSeconds
        ProgressOffset=$offset; ProgressTotal=$total; ProgressStartedAt=$started
        ProgressActivity="V46 $Experiment matrix"
        RoundProgressPollSeconds=$RoundProgressPollSeconds
    }
    if ($built) { $arguments.SkipBuild = $true }
    & $runner @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "V46 batch failed: $($spec.Tag)"
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

$prefix = "v46_$Experiment"
$resolvedRoot = (Resolve-Path $root).Path
docker run --rm `
    --mount "type=bind,source=$resolvedRoot,target=/results" `
    node-access-flower-prototype:0.23 `
    python -m flower_prototype.analyze_v46_evidence $Experiment $Stage `
    /results/real_fl_node_results.csv `
    /results/real_fl_round_node_events.csv `
    /results/real_fl_round_metrics.csv `
    /results/real_fl_configuration_metrics.csv `
    "/results/${prefix}_runs.csv" `
    "/results/${prefix}_summary.csv" `
    "/results/${prefix}_paired_effects.csv" `
    "/results/${prefix}_paired_summary.csv" `
    "/results/${prefix}_round_runs.csv" `
    "/results/${prefix}_round_summary.csv"
if ($LASTEXITCODE -ne 0) { throw "V46 $Experiment summary generation failed" }

Write-Progress -Id 41 -Activity "V46 $Experiment matrix" -Completed
Write-Host "Completed V46 experiment=$Experiment stage=$Stage results: $root" -ForegroundColor Green
Write-Host "Summary: $(Join-Path $root "${prefix}_summary.csv")"
Write-Host "Paired summary: $(Join-Path $root "${prefix}_paired_summary.csv")"
Write-Host "Round trajectory: $(Join-Path $root "${prefix}_round_summary.csv")"

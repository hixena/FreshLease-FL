param(
    [Parameter(Mandatory = $true)]
    [string]$BaseResultDirectory,
    [int]$RepeatStart = 5,
    [int]$AdditionalRepeats = 5,
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
$runner = Join-Path $scriptDir "run_real_fl_matrix.ps1"
$datasetFile = Join-Path $scriptDir "data\cifar10.npz"

if (-not (Test-Path $datasetFile)) {
    throw "CIFAR-10 data is missing. Copy cifar10.npz into flower_prototype\data or run prepare_cifar10.py"
}
if ($RepeatStart -lt 1) {
    throw "RepeatStart must be at least 1 because V48 extends an existing result"
}
if ($AdditionalRepeats -lt 1) { throw "AdditionalRepeats must be positive" }
if ($Rounds -lt 16) { throw "At least 16 rounds are required for an L5 lease cycle" }

$baseRoot = [IO.Path]::GetFullPath($BaseResultDirectory)
if (-not (Test-Path $baseRoot -PathType Container)) {
    throw "Base result directory does not exist: $baseRoot"
}

$rawFiles = @(
    "real_fl_node_results.csv",
    "real_fl_round_node_events.csv",
    "real_fl_round_metrics.csv",
    "real_fl_configuration_metrics.csv"
)
foreach ($name in $rawFiles) {
    $path = Join-Path $baseRoot $name
    if (-not (Test-Path $path -PathType Leaf)) {
        throw "Base result is missing $name`: $baseRoot"
    }
}

$scenarios = @("diverse_then_repeat_backdoor", "benign_concept_drift")
$variants = @(
    @{ Name = "access_freshness_lease"; Tag = "lease-L5" },
    @{ Name = "access_full"; Tag = "progressive-no-lease" },
    @{ Name = "rffl_reputation"; Tag = "rffl-style" }
)

$baseConfigurations = @(Import-Csv (Join-Path $baseRoot "real_fl_configuration_metrics.csv"))
$expectedBaseCount = $variants.Count * $scenarios.Count * $RepeatStart
if ($baseConfigurations.Count -ne $expectedBaseCount) {
    throw "Expected $expectedBaseCount base configurations (seeds 0-$($RepeatStart - 1)); found $($baseConfigurations.Count)"
}
$baseRepeats = @(
    $baseConfigurations |
        ForEach-Object { [int]$_.repeat } |
        Sort-Object -Unique
)
$expectedBaseRepeats = @(0..($RepeatStart - 1))
if (($baseRepeats -join ",") -ne ($expectedBaseRepeats -join ",")) {
    throw "Base result must contain exactly seeds 0-$($RepeatStart - 1); found $($baseRepeats -join ',')"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$root = if ($ResultDirectory) { $ResultDirectory } else {
    Join-Path $scriptDir "cifar10_v48_results-formal10-$stamp"
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

$total = $variants.Count * $scenarios.Count * $AdditionalRepeats
$started = Get-Date
$offset = 0
$built = $SkipBuild.IsPresent
$batches = @()
$batchNumber = 0

Write-Host "V48 precision extension: new configurations=$total rounds=$Rounds" -ForegroundColor Cyan
Write-Host "Reusing frozen V47 seeds 0-$($RepeatStart - 1) from: $baseRoot"
Write-Host "Running additional seeds $RepeatStart-$($RepeatStart + $AdditionalRepeats - 1)"
Write-Host "No policy, model, attack, dataset, or hyperparameter changes"

foreach ($variant in $variants) {
    $batchNumber++
    $batchDir = Join-Path $root (
        "batch-{0:D2}-{1}-seeds-{2}-to-{3}" -f `
        $batchNumber, $variant.Tag, $RepeatStart,
        ($RepeatStart + $AdditionalRepeats - 1)
    )
    $perBatch = $scenarios.Count * $AdditionalRepeats
    if ($Resume -and (Test-Path $batchDir)) {
        $hasAllFiles = ($rawFiles | Where-Object {
            -not (Test-Path (Join-Path $batchDir $_))
        }).Count -eq 0
        $completedConfigurations = 0
        if ($hasAllFiles) {
            $completedConfigurations = @(
                Import-Csv (Join-Path $batchDir "real_fl_configuration_metrics.csv")
            ).Count
        }
        if ($hasAllFiles -and $completedConfigurations -eq $perBatch) {
            Write-Host "Resume: reusing complete extension batch $batchDir" -ForegroundColor Cyan
            $batches += $batchDir
            $offset += $perBatch
            continue
        }
        $batchDir = Join-Path $root (
            "batch-{0:D2}-{1}-retry-{2}" -f $batchNumber, $variant.Tag, $stamp
        )
        Write-Host "Resume: incomplete extension batch; retrying as $batchDir" -ForegroundColor Yellow
    }

    $arguments = @{
        Repeats=$AdditionalRepeats; RepeatStart=$RepeatStart
        Rounds=$Rounds; AttackStartRound=2
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
        ProgressActivity="V48 CIFAR-10 precision extension"
        RoundProgressPollSeconds=$RoundProgressPollSeconds
    }
    if ($built) { $arguments.SkipBuild = $true }
    & $runner @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "V48 extension batch failed: $($variant.Name)"
    }
    $built = $true
    $batches += $batchDir
    $offset += $perBatch
}

foreach ($name in $rawFiles) {
    $destination = Join-Path $root $name
    $rows = @(
        Import-Csv (Join-Path $baseRoot $name)
        foreach ($batch in $batches) {
            Import-Csv (Join-Path $batch $name)
        }
    )
    $rows | Export-Csv -Path $destination -NoTypeInformation -Encoding UTF8
}

$combinedConfigurations = @(
    Import-Csv (Join-Path $root "real_fl_configuration_metrics.csv")
)
$expectedTotal = $variants.Count * $scenarios.Count * (
    $RepeatStart + $AdditionalRepeats
)
if ($combinedConfigurations.Count -ne $expectedTotal) {
    throw "Combined matrix expected $expectedTotal configurations; found $($combinedConfigurations.Count)"
}
$identityFields = @(
    "dataset", "client_count", "attacker_count", "cohort_mode", "variant",
    "trust_estimator", "attack_scenario", "noniid_alpha", "repeat"
)
$configurationKeys = @($combinedConfigurations | ForEach-Object {
    $row = $_
    ($identityFields | ForEach-Object {
        [string]$row.PSObject.Properties[$_].Value
    }) -join "|"
})
if (@($configurationKeys | Sort-Object -Unique).Count -ne $configurationKeys.Count) {
    throw "Combined configuration matrix contains duplicate experiment keys"
}
$combinedRepeats = @(
    $combinedConfigurations |
        ForEach-Object { [int]$_.repeat } |
        Sort-Object -Unique
)
$expectedRepeats = @(0..($RepeatStart + $AdditionalRepeats - 1))
if (($combinedRepeats -join ",") -ne ($expectedRepeats -join ",")) {
    throw "Combined matrix has unexpected seeds: $($combinedRepeats -join ',')"
}

$resolvedRoot = (Resolve-Path $root).Path
docker run --rm `
    --mount "type=bind,source=$resolvedRoot,target=/results" `
    node-access-flower-prototype:0.23 `
    python -m flower_prototype.analyze_cifar10_v47 formal10 `
    /results/real_fl_node_results.csv `
    /results/real_fl_round_node_events.csv `
    /results/real_fl_round_metrics.csv `
    /results/real_fl_configuration_metrics.csv `
    /results/cifar10_v48_runs.csv `
    /results/cifar10_v48_summary.csv `
    /results/cifar10_v48_paired_effects.csv `
    /results/cifar10_v48_paired_summary.csv `
    /results/cifar10_v48_round_runs.csv `
    /results/cifar10_v48_round_summary.csv
if ($LASTEXITCODE -ne 0) { throw "V48 combined summary generation failed" }

Write-Progress -Id 41 -Activity "V48 CIFAR-10 precision extension" -Completed
Write-Host "Completed V48 combined 10-seed results: $root" -ForegroundColor Green
Write-Host "Summary: $(Join-Path $root 'cifar10_v48_summary.csv')"
Write-Host "Paired summary: $(Join-Path $root 'cifar10_v48_paired_summary.csv')"
Write-Host "Round trajectory: $(Join-Path $root 'cifar10_v48_round_summary.csv')"

param(
    [ValidateSet("repeat", "progressive", "upgrade", "baselines")]
    [string]$Stage = "repeat",
    [int]$Repeats = 5,
    [int]$Rounds = 8,
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
$root = if ($ResultDirectory) { $ResultDirectory } else {
    Join-Path $scriptDir "access_robustness_v40_results-$Stage-$stamp"
}
if (Test-Path $root) { throw "Result directory already exists: $root" }
New-Item -ItemType Directory -Path $root | Out-Null

$built = $SkipBuild.IsPresent
$batchNumber = 0
$batches = @()

function Invoke-V40Batch {
    param(
        [string]$Label, [string[]]$Variants, [string[]]$Scenarios,
        [int]$MinTasks, [int]$MaxTasks, [double]$EvidenceMass,
        [int]$CleanUpdates, [double]$LimitedWeight
    )
    $script:batchNumber++
    $batchDir = Join-Path $root ("batch-{0:D2}-{1}" -f $script:batchNumber, $Label)
    $arguments = @{
        Repeats = $Repeats; Rounds = $Rounds; AttackStartRound = 2
        LimitedCleanUpdates = $CleanUpdates; LimitedAggregationWeight = $LimitedWeight
        MaxProbationTasks = $MaxTasks; MinProbationCompletedTasks = $MinTasks
        MinEvidenceMass = $EvidenceMass; NormEscalationMode = "moderate_shadow"
        NonIIDAlphas = $NonIIDAlphas; Datasets = $Datasets; ClientCount = $ClientCount
        CohortMode = "synchronous_cold_start"; TrustEstimators = @("dirichlet_lcb")
        Variants = $Variants; AttackScenarios = $Scenarios; ResultDirectory = $batchDir
        TaskDeadlineSeconds = $TaskDeadlineSeconds
        BarrierTimeoutSeconds = $BarrierTimeoutSeconds
        ClientJoinTimeoutSeconds = $ClientJoinTimeoutSeconds
    }
    if ($script:built) { $arguments.SkipBuild = $true }
    & $runner @arguments
    if ($LASTEXITCODE -ne 0) { throw "V40 batch failed: $Label" }
    $script:built = $true
    $script:batches += $batchDir
}

if ($Stage -eq "repeat") {
    # Pre-registered structural contrast: three evidence types, six completed
    # tasks, and a 2.70 evidence-mass floor. Repeat decay stays below the floor;
    # the no-decay ablation crosses it before the 20-task budget expires.
    Invoke-V40Batch "repeat" `
        @("access_full", "access_no_repeat_decay") `
        @("three_type_repeat_farming") 6 20 2.70 3 0.50
} elseif ($Stage -eq "progressive") {
    $scenarios = @("diverse_then_repeat_farming", "diverse_then_repeat_backdoor")
    foreach ($weight in @(0.25, 0.50, 0.75)) {
        Invoke-V40Batch "full-w$($weight.ToString().Replace('.', 'p'))" `
            @("access_full") $scenarios 9 20 1.50 3 $weight
    }
    # Run the direct-admission control once: its behavior is independent of
    # LIMITED weight, so duplicating it would create pseudo-replication.
    Invoke-V40Batch "no-limited-control" `
        @("access_no_limited") $scenarios 9 20 1.50 3 1.0
} elseif ($Stage -eq "upgrade") {
    foreach ($clean in @(2, 3, 4)) {
        Invoke-V40Batch "clean-$clean" @("access_full") `
            @("diverse_then_repeat_farming", "diverse_then_repeat_backdoor") `
            9 20 1.50 $clean 0.50
    }
} else {
    Invoke-V40Batch "baselines" `
        @("access_full", "access_attestation_only", "access_static_multisource", "access_naive") `
        @("single_type_farming", "false_quality_reporting", "attestation_camouflage", "diverse_then_repeat_farming") `
        9 20 1.50 3 0.75
}

foreach ($name in @("real_fl_node_results.csv", "real_fl_round_node_events.csv", "real_fl_round_metrics.csv")) {
    $destination = Join-Path $root $name
    $allRows = foreach ($batch in $batches) { Import-Csv (Join-Path $batch $name) }
    $allRows | Export-Csv -Path $destination -NoTypeInformation -Encoding UTF8
}

Push-Location $projectRoot
try {
    python -m flower_prototype.analyze_access_v40 $Stage `
        (Join-Path $root "real_fl_node_results.csv") `
        (Join-Path $root "real_fl_round_node_events.csv") `
        (Join-Path $root "real_fl_round_metrics.csv") `
        (Join-Path $root "access_robustness_v40_runs.csv") `
        (Join-Path $root "access_robustness_v40_summary.csv") `
        (Join-Path $root "access_robustness_v40_paired_effects.csv")
    if ($LASTEXITCODE -ne 0) { throw "V40 summary generation failed" }
} finally {
    Pop-Location
}

Write-Host "Completed V40 stage=$Stage results: $root"
Write-Host "Summary: $(Join-Path $root 'access_robustness_v40_summary.csv')"
Write-Host "Paired effects: $(Join-Path $root 'access_robustness_v40_paired_effects.csv')"

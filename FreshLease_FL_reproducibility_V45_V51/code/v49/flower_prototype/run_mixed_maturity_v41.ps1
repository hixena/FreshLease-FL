param(
    [ValidateSet("primary", "observation_sensitivity")]
    [string]$Stage = "primary",
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
    Join-Path $scriptDir "mixed_maturity_v41_results-$Stage-$stamp"
}
if (Test-Path $root) { throw "Result directory already exists: $root" }
New-Item -ItemType Directory -Path $root | Out-Null

$scenarios = @("diverse_then_repeat_farming", "diverse_then_repeat_backdoor")
$fullSettings = if ($Stage -eq "primary") {
    @(
        @{ Label = "full-w0p25"; ObservationUpdates = 5; Weight = 0.25 },
        @{ Label = "full-w0p50"; ObservationUpdates = 5; Weight = 0.50 },
        @{ Label = "full-w0p75"; ObservationUpdates = 5; Weight = 0.75 }
    )
} else {
    @(
        @{ Label = "observe-3"; ObservationUpdates = 3; Weight = 0.50 },
        @{ Label = "observe-5"; ObservationUpdates = 5; Weight = 0.50 },
        @{ Label = "observe-7"; ObservationUpdates = 7; Weight = 0.50 }
    )
}

$settings = @($fullSettings) + @(
    @{ Label = "no-limited-control"; ObservationUpdates = 5; Weight = 1.0; Control = $true }
)
$perBatch = $scenarios.Count * $NonIIDAlphas.Count * $Datasets.Count * $Repeats
$totalConfigurations = $settings.Count * $perBatch
$progressStartedAt = Get-Date
$progressOffset = 0
$built = $SkipBuild.IsPresent
$batchNumber = 0
$batches = @()

Write-Host "V41 stage=$Stage configurations=$totalConfigurations rounds=$Rounds clients=$ClientCount"
Write-Host "Cohort: 9 mature ADMITTED normal nodes + 1 LIMITED newcomer target"
Write-Host "Progress shows completed/total, elapsed time, and estimated remaining time."

foreach ($setting in $settings) {
    $batchNumber++
    $batchDir = Join-Path $root ("batch-{0:D2}-{1}" -f $batchNumber, $setting.Label)
    $variant = if ($setting.Control) { "access_no_limited" } else { "access_full" }
    $arguments = @{
        Repeats = $Repeats
        Rounds = $Rounds
        AttackStartRound = 2
        LimitedObservationUpdates = $setting.ObservationUpdates
        LimitedAggregationWeight = $setting.Weight
        MaxProbationTasks = 20
        MinProbationCompletedTasks = 9
        MinEvidenceMass = 1.50
        NormEscalationMode = "moderate_shadow"
        NonIIDAlphas = $NonIIDAlphas
        Datasets = $Datasets
        ClientCount = $ClientCount
        CohortMode = "mixed_maturity"
        TrustEstimators = @("dirichlet_lcb")
        Variants = @($variant)
        AttackScenarios = $scenarios
        ResultDirectory = $batchDir
        TaskDeadlineSeconds = $TaskDeadlineSeconds
        BarrierTimeoutSeconds = $BarrierTimeoutSeconds
        ClientJoinTimeoutSeconds = $ClientJoinTimeoutSeconds
        ProgressOffset = $progressOffset
        ProgressTotal = $totalConfigurations
        ProgressStartedAt = $progressStartedAt
    }
    if ($built) { $arguments.SkipBuild = $true }
    & $runner @arguments
    if ($LASTEXITCODE -ne 0) { throw "V41 batch failed: $($setting.Label)" }
    $built = $true
    $batches += $batchDir
    $progressOffset += $perBatch
}

foreach ($name in @("real_fl_node_results.csv", "real_fl_round_node_events.csv", "real_fl_round_metrics.csv")) {
    $destination = Join-Path $root $name
    $allRows = foreach ($batch in $batches) { Import-Csv (Join-Path $batch $name) }
    $allRows | Export-Csv -Path $destination -NoTypeInformation -Encoding UTF8
}

Push-Location $projectRoot
try {
    python -m flower_prototype.analyze_mixed_maturity_v41 $Stage `
        (Join-Path $root "real_fl_node_results.csv") `
        (Join-Path $root "real_fl_round_node_events.csv") `
        (Join-Path $root "real_fl_round_metrics.csv") `
        (Join-Path $root "mixed_maturity_v41_runs.csv") `
        (Join-Path $root "mixed_maturity_v41_summary.csv") `
        (Join-Path $root "mixed_maturity_v41_paired_effects.csv")
    if ($LASTEXITCODE -ne 0) { throw "V41 summary generation failed" }
} finally {
    Pop-Location
}

Write-Host "Completed V41 stage=$Stage results: $root" -ForegroundColor Green
Write-Host "Progress summary: succeeded=$progressOffset failed=0 skipped=0 total=$totalConfigurations elapsed=$(((Get-Date) - $progressStartedAt).ToString('hh\:mm\:ss'))"
Write-Host "Summary: $(Join-Path $root 'mixed_maturity_v41_summary.csv')"
Write-Host "Paired effects: $(Join-Path $root 'mixed_maturity_v41_paired_effects.csv')"

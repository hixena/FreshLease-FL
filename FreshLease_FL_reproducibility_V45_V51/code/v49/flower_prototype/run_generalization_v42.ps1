param(
    [ValidateSet("smoke", "dataset_noniid", "client_scale")]
    [string]$Stage = "smoke",
    [int]$Repeats = 5,
    [int]$Rounds = 8,
    [int]$TaskDeadlineSeconds = 30,
    [int]$BarrierTimeoutSeconds = 900,
    [int]$ClientJoinTimeoutSeconds = 600,
    [switch]$SkipBuild,
    [string]$ResultDirectory = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$runner = Join-Path $scriptDir "run_real_fl_matrix.ps1"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$root = if ($ResultDirectory) { $ResultDirectory } else {
    Join-Path $scriptDir "generalization_v42_results-$Stage-$stamp"
}
if (Test-Path $root) { throw "Result directory already exists: $root" }
New-Item -ItemType Directory -Path $root | Out-Null

$effectiveRepeats = if ($Stage -eq "smoke" -and -not $PSBoundParameters.ContainsKey("Repeats")) { 1 } else { $Repeats }
$effectiveRounds = if ($Stage -eq "smoke" -and -not $PSBoundParameters.ContainsKey("Rounds")) { 3 } else { $Rounds }
$scenarios = @("diverse_then_repeat_farming", "diverse_then_repeat_backdoor")

$scopes = switch ($Stage) {
    "smoke" {
        @(@{ Label = "n10-digits-fashion-a0p5"; ClientCount = 10;
             Datasets = @("digits", "fashion_mnist"); Alphas = @(0.5) })
    }
    "dataset_noniid" {
        @(@{ Label = "n10-digits-fashion-a0p1-a0p5-a10"; ClientCount = 10;
             Datasets = @("digits", "fashion_mnist"); Alphas = @(0.1, 0.5, 10.0) })
    }
    "client_scale" {
        @(
            @{ Label = "n10-fashion-a0p5"; ClientCount = 10;
               Datasets = @("fashion_mnist"); Alphas = @(0.5) },
            @{ Label = "n20-fashion-a0p5"; ClientCount = 20;
               Datasets = @("fashion_mnist"); Alphas = @(0.5) }
        )
    }
}

$settings = foreach ($scope in $scopes) {
    [PSCustomObject]@{
        Label = "full-$($scope.Label)"; ClientCount = $scope.ClientCount
        Datasets = $scope.Datasets; Alphas = $scope.Alphas
        Variant = "access_full"; Weight = 0.50
    }
    [PSCustomObject]@{
        Label = "control-$($scope.Label)"; ClientCount = $scope.ClientCount
        Datasets = $scope.Datasets; Alphas = $scope.Alphas
        Variant = "access_no_limited"; Weight = 1.0
    }
}

$totalConfigurations = 0
foreach ($setting in $settings) {
    $totalConfigurations += (
        $setting.Datasets.Count * $setting.Alphas.Count *
        $scenarios.Count * $effectiveRepeats
    )
}
$progressStartedAt = Get-Date
$progressOffset = 0
$built = $SkipBuild.IsPresent
$batchNumber = 0
$batches = @()

Write-Host "V42 stage=$Stage configurations=$totalConfigurations rounds=$effectiveRounds clients=$(($scopes.ClientCount -join ','))"
Write-Host "Frozen work point: observation_updates=5 limited_weight=0.50; paired control weight=1.00"
Write-Host "Progress shows completed/total, elapsed time, and estimated remaining time."

foreach ($setting in $settings) {
    $batchNumber++
    $batchDir = Join-Path $root ("batch-{0:D2}-{1}" -f $batchNumber, $setting.Label)
    $perBatch = (
        $setting.Datasets.Count * $setting.Alphas.Count *
        $scenarios.Count * $effectiveRepeats
    )
    $arguments = @{
        Repeats = $effectiveRepeats
        Rounds = $effectiveRounds
        AttackStartRound = 2
        LimitedObservationUpdates = 5
        LimitedAggregationWeight = $setting.Weight
        MaxProbationTasks = 20
        MinProbationCompletedTasks = 9
        MinEvidenceMass = 1.50
        NormEscalationMode = "moderate_shadow"
        NonIIDAlphas = $setting.Alphas
        Datasets = $setting.Datasets
        ClientCount = $setting.ClientCount
        CohortMode = "mixed_maturity"
        TrustEstimators = @("dirichlet_lcb")
        Variants = @($setting.Variant)
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
    if ($LASTEXITCODE -ne 0) { throw "V42 batch failed: $($setting.Label)" }
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
    python -m flower_prototype.analyze_generalization_v42 $Stage `
        (Join-Path $root "real_fl_node_results.csv") `
        (Join-Path $root "real_fl_round_node_events.csv") `
        (Join-Path $root "real_fl_round_metrics.csv") `
        (Join-Path $root "generalization_v42_runs.csv") `
        (Join-Path $root "generalization_v42_summary.csv") `
        (Join-Path $root "generalization_v42_paired_effects.csv")
    if ($LASTEXITCODE -ne 0) { throw "V42 summary generation failed" }
} finally {
    Pop-Location
}

Write-Host "Completed V42 stage=$Stage results: $root" -ForegroundColor Green
Write-Host "Progress summary: succeeded=$progressOffset failed=0 skipped=0 total=$totalConfigurations elapsed=$(((Get-Date) - $progressStartedAt).ToString('hh\:mm\:ss'))"
Write-Host "Summary: $(Join-Path $root 'generalization_v42_summary.csv')"
Write-Host "Paired effects: $(Join-Path $root 'generalization_v42_paired_effects.csv')"

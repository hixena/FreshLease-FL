param(
    [ValidateSet("smoke", "external_baselines", "fixed_attacker_ratio")]
    [string]$Stage = "smoke",
    [int]$Repeats = 5,
    [int]$Rounds = 8,
    [int]$TaskDeadlineSeconds = 30,
    [int]$BarrierTimeoutSeconds = 900,
    [int]$ClientJoinTimeoutSeconds = 600,
    [switch]$SkipBuild,
    [switch]$Resume,
    [string]$ResultDirectory = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$runner = Join-Path $scriptDir "run_real_fl_matrix.ps1"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$root = if ($ResultDirectory) { $ResultDirectory } else {
    Join-Path $scriptDir "q3_v43_results-$Stage-$stamp"
}
if (Test-Path $root) {
    if (-not $Resume) { throw "Result directory already exists: $root" }
} else {
    New-Item -ItemType Directory -Path $root | Out-Null
}

$effectiveRepeats = if ($Stage -eq "smoke" -and -not $PSBoundParameters.ContainsKey("Repeats")) { 1 } else { $Repeats }
$effectiveRounds = if ($Stage -eq "smoke" -and -not $PSBoundParameters.ContainsKey("Rounds")) { 3 } else { $Rounds }
$scenarios = @("diverse_then_repeat_farming", "diverse_then_repeat_backdoor")
$variants = @(
    @{ Name = "access_full"; Weight = 0.50 },
    @{ Name = "rffl_reputation"; Weight = 1.00 },
    @{ Name = "access_no_limited"; Weight = 1.00 }
)
$scopes = switch ($Stage) {
    "smoke" { @(@{ Label="n10-k1-fashion-a0p5"; Clients=10; Attackers=1; Alphas=@(0.5) }) }
    "external_baselines" { @(@{ Label="n10-k1-fashion-all-alpha"; Clients=10; Attackers=1; Alphas=@(0.1,0.5,10.0) }) }
    "fixed_attacker_ratio" {
        @(
            @{ Label="n10-k1-fashion-a0p5"; Clients=10; Attackers=1; Alphas=@(0.5) },
            @{ Label="n20-k2-fashion-a0p5"; Clients=20; Attackers=2; Alphas=@(0.5) }
        )
    }
}

$total = 0
foreach ($scope in $scopes) {
    $total += $scope.Alphas.Count * $variants.Count * $scenarios.Count * $effectiveRepeats
}
$started = Get-Date
$offset = 0
$built = $SkipBuild.IsPresent
$batches = @()
$batchNumber = 0
Write-Host "V43 stage=$Stage configurations=$total rounds=$effectiveRounds"
Write-Host "Comparators: proposed access_full, RFFL-style reputation, direct admission/FedAvg"
Write-Host "Overhead: wall time, server processing time, update payload bytes, controller state bytes"

foreach ($scope in $scopes) {
  foreach ($variant in $variants) {
    $batchNumber++
    $batchDir = Join-Path $root ("batch-{0:D2}-{1}-{2}" -f $batchNumber, $variant.Name, $scope.Label)
    $perBatch = $scope.Alphas.Count * $scenarios.Count * $effectiveRepeats
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
            "batch-{0:D2}-{1}-{2}-retry-{3}" -f `
            $batchNumber, $variant.Name, $scope.Label, $stamp
        )
        Write-Host "Resume: prior batch incomplete; retrying as $batchDir" -ForegroundColor Yellow
    }
    $arguments = @{
        Repeats=$effectiveRepeats; Rounds=$effectiveRounds; AttackStartRound=2
        LimitedObservationUpdates=5; LimitedAggregationWeight=$variant.Weight
        MaxProbationTasks=20; MinProbationCompletedTasks=9; MinEvidenceMass=1.50
        NormEscalationMode="moderate_shadow"; NonIIDAlphas=$scope.Alphas
        Datasets=@("fashion_mnist"); ClientCount=$scope.Clients
        TargetCount=$scope.Attackers; UseGenericAttackers=$true
        CohortMode="mixed_maturity"; TrustEstimators=@("dirichlet_lcb")
        Variants=@($variant.Name); AttackScenarios=$scenarios
        ResultDirectory=$batchDir; TaskDeadlineSeconds=$TaskDeadlineSeconds
        BarrierTimeoutSeconds=$BarrierTimeoutSeconds
        ClientJoinTimeoutSeconds=$ClientJoinTimeoutSeconds
        ProgressOffset=$offset; ProgressTotal=$total; ProgressStartedAt=$started
    }
    if ($built) { $arguments.SkipBuild = $true }
    & $runner @arguments
    if ($LASTEXITCODE -ne 0) { throw "V43 batch failed: $($variant.Name)-$($scope.Label)" }
    $built = $true
    $batches += $batchDir
    $offset += $perBatch
  }
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
    python -m flower_prototype.analyze_q3_v43 $Stage `
    /results/real_fl_node_results.csv `
    /results/real_fl_round_node_events.csv `
    /results/real_fl_round_metrics.csv `
    /results/real_fl_configuration_metrics.csv `
    /results/q3_v43_runs.csv `
    /results/q3_v43_summary.csv `
    /results/q3_v43_paired_effects.csv
if ($LASTEXITCODE -ne 0) { throw "V43 summary generation failed" }

Write-Progress -Id 41 -Activity "V43 Q3 evidence matrix" -Completed
Write-Host "Completed V43 stage=$Stage results: $root" -ForegroundColor Green
Write-Host "Summary: $(Join-Path $root 'q3_v43_summary.csv')"
Write-Host "Paired effects: $(Join-Path $root 'q3_v43_paired_effects.csv')"

param(
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
    Join-Path $scriptDir "long_horizon_v44_results-$Stage-$stamp"
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

$effectiveRepeats = if ($Stage -eq "smoke" -and -not $PSBoundParameters.ContainsKey("Repeats")) { 1 } else { $Repeats }
$effectiveRounds = if ($Stage -eq "smoke" -and -not $PSBoundParameters.ContainsKey("Rounds")) { 8 } else { $Rounds }
if ($effectiveRounds -lt 8) { throw "V44 requires at least 8 rounds" }

$scenario = "diverse_then_repeat_backdoor"
$variants = @(
    @{ Name = "access_full"; Weight = 0.50 },
    @{ Name = "rffl_reputation"; Weight = 1.00 },
    @{ Name = "access_no_limited"; Weight = 1.00 }
)
$total = $variants.Count * $effectiveRepeats
$started = Get-Date
$offset = 0
$built = $SkipBuild.IsPresent
$batches = @()
$batchNumber = 0

Write-Host "V44 stage=$Stage configurations=$total rounds=$effectiveRounds"
Write-Host "Frozen scope: Fashion-MNIST, clients=20, attackers=2, alpha=0.5, backdoor"
Write-Host "Comparators: progressive access, RFFL-style reputation, direct admission/FedAvg"
Write-Host "Primary question: does the V43 benefit persist after round-5 promotion?"

foreach ($variant in $variants) {
    $batchNumber++
    $batchDir = Join-Path $root (
        "batch-{0:D2}-{1}-n20-k2-fashion-a0p5" -f $batchNumber, $variant.Name
    )
    $perBatch = $effectiveRepeats
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
            "batch-{0:D2}-{1}-n20-k2-fashion-a0p5-retry-{2}" -f `
            $batchNumber, $variant.Name, $stamp
        )
        Write-Host "Resume: prior batch incomplete; retrying as $batchDir" -ForegroundColor Yellow
    }

    $arguments = @{
        Repeats=$effectiveRepeats; Rounds=$effectiveRounds; AttackStartRound=2
        LimitedObservationUpdates=5; LimitedAggregationWeight=$variant.Weight
        MaxProbationTasks=20; MinProbationCompletedTasks=9; MinEvidenceMass=1.50
        NormEscalationMode="moderate_shadow"; NonIIDAlphas=@(0.5)
        Datasets=@("fashion_mnist"); ClientCount=20; TargetCount=2
        UseGenericAttackers=$true; CohortMode="mixed_maturity"
        TrustEstimators=@("dirichlet_lcb"); Variants=@($variant.Name)
        AttackScenarios=@($scenario); ResultDirectory=$batchDir
        TaskDeadlineSeconds=$TaskDeadlineSeconds
        BarrierTimeoutSeconds=$BarrierTimeoutSeconds
        ClientJoinTimeoutSeconds=$ClientJoinTimeoutSeconds
        ProgressOffset=$offset; ProgressTotal=$total; ProgressStartedAt=$started
        ProgressActivity="V44 long-horizon matrix"
        RoundProgressPollSeconds=$RoundProgressPollSeconds
    }
    if ($built) { $arguments.SkipBuild = $true }
    & $runner @arguments
    if ($LASTEXITCODE -ne 0) { throw "V44 batch failed: $($variant.Name)" }
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
    python -m flower_prototype.analyze_long_horizon_v44 $Stage `
    /results/real_fl_node_results.csv `
    /results/real_fl_round_node_events.csv `
    /results/real_fl_round_metrics.csv `
    /results/real_fl_configuration_metrics.csv `
    /results/long_horizon_v44_runs.csv `
    /results/long_horizon_v44_summary.csv `
    /results/long_horizon_v44_paired_effects.csv `
    /results/long_horizon_v44_round_runs.csv `
    /results/long_horizon_v44_round_summary.csv
if ($LASTEXITCODE -ne 0) { throw "V44 summary generation failed" }

Write-Progress -Id 41 -Activity "V44 long-horizon matrix" -Completed
Write-Host "Completed V44 stage=$Stage results: $root" -ForegroundColor Green
Write-Host "Summary: $(Join-Path $root 'long_horizon_v44_summary.csv')"
Write-Host "Paired effects: $(Join-Path $root 'long_horizon_v44_paired_effects.csv')"
Write-Host "Round trajectory: $(Join-Path $root 'long_horizon_v44_round_summary.csv')"

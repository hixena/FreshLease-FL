param(
    [ValidateSet("smoke", "formal")]
    [string]$Stage = "smoke",
    [ValidateSet("alpha_low", "alpha_high", "scale_10")]
    [string[]]$Conditions = @("alpha_low", "alpha_high", "scale_10"),
    [int]$Repeats = 10,
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
$dataFile = Join-Path $scriptDir "data\cifar10.npz"
if (-not (Test-Path $dataFile -PathType Leaf)) {
    throw "CIFAR-10 data missing: copy the frozen cifar10.npz to flower_prototype\data"
}
$effectiveRepeats = if ($Stage -eq "smoke" -and -not $PSBoundParameters.ContainsKey("Repeats")) { 1 } else { $Repeats }
$effectiveRounds = if ($Stage -eq "smoke" -and -not $PSBoundParameters.ContainsKey("Rounds")) { 16 } else { $Rounds }
if ($effectiveRepeats -lt 1 -or $effectiveRounds -lt 16) { throw "At least 1 seed and 16 rounds are required" }
if ($Stage -eq "formal" -and ($effectiveRepeats -ne 10 -or $effectiveRounds -ne 50)) {
    throw "V51 formal matrix is frozen to 10 paired seeds and 50 rounds"
}
if (@($Conditions | Sort-Object -Unique).Count -ne $Conditions.Count) { throw "Duplicate V51 condition" }

$profiles = @{
    alpha_low = @{ Alpha = 0.1; Clients = 20; Attackers = 2 }
    alpha_high = @{ Alpha = 10.0; Clients = 20; Attackers = 2 }
    scale_10 = @{ Alpha = 0.5; Clients = 10; Attackers = 1 }
}
$variants = @("access_freshness_lease", "access_full", "rffl_reputation")
$scenarios = @("diverse_then_repeat_backdoor", "benign_concept_drift")
$required = @("real_fl_node_results.csv", "real_fl_round_node_events.csv", "real_fl_round_metrics.csv", "real_fl_configuration_metrics.csv", "real_fl_audit_integrity.csv")
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$root = if ($ResultDirectory) { $ResultDirectory } else { Join-Path $scriptDir "q3_access_v51_results-$Stage-$stamp" }
if (-not [IO.Path]::IsPathRooted($root)) { $root = Join-Path (Get-Location).Path $root }
$root = [IO.Path]::GetFullPath($root)
if (Test-Path $root) {
    if (-not $Resume) { throw "Result directory exists: $root; use -Resume" }
} else { New-Item -ItemType Directory -Path $root | Out-Null }

$total = $Conditions.Count * $variants.Count * $scenarios.Count * $effectiveRepeats
$started = Get-Date
$offset = 0
$built = $SkipBuild.IsPresent
$batches = @()
$batchNumber = 0
Write-Host "V51 scope=$($Conditions -join ',') stage=$Stage configurations=$total rounds=$effectiveRounds" -ForegroundColor Cyan
Write-Host "CIFAR-10, attack from round 6; L5/O5/lambda0.5; paired backdoor and benign-newcomer cases"
foreach ($condition in $Conditions) {
    $spec = $profiles[$condition]
    foreach ($variant in $variants) {
        $batchNumber++
        $batchTag = "batch-{0:D2}-{1}-{2}" -f $batchNumber, $condition, $variant
        $batch = Join-Path $root $batchTag
        $perBatch = $scenarios.Count * $effectiveRepeats
        if ($Resume -and (Test-Path $batch)) {
            $complete = @($required | Where-Object { -not (Test-Path (Join-Path $batch $_) -PathType Leaf) }).Count -eq 0
            if ($complete) {
                $configs = @(Import-Csv (Join-Path $batch "real_fl_configuration_metrics.csv"))
                $complete = ($configs.Count -eq $perBatch -and @($configs | Where-Object {
                    $_.dataset -ne "cifar10" -or [int]$_.client_count -ne $spec.Clients -or
                    [int]$_.attacker_count -ne $spec.Attackers -or
                    [double]$_.noniid_alpha -ne [double]$spec.Alpha -or
                    [int]$_.rounds -ne $effectiveRounds -or $_.variant -ne $variant
                }).Count -eq 0)
                if ($complete) {
                    $nodeRows = @(Import-Csv (Join-Path $batch "real_fl_node_results.csv"))
                    $complete = ($nodeRows.Count -eq $perBatch * [int]$spec.Clients -and
                        @($nodeRows | Where-Object { [int]$_.attack_start_round -ne 6 }).Count -eq 0)
                }
            }
            if ($complete) {
                Write-Host "Resume: verified complete $batchTag" -ForegroundColor Cyan
                $batches += $batch
                $offset += $perBatch
                continue
            }
            $batch = Join-Path $root "$batchTag-retry-$stamp"
            Write-Host "Resume: incomplete $batchTag, retry in $batch" -ForegroundColor Yellow
        }
        $runnerArguments = @{
            Repeats=$effectiveRepeats; Rounds=$effectiveRounds; AttackStartRound=6
            CaptureAuditIntegrity=$true
            LimitedObservationUpdates=5; LimitedAggregationWeight=0.50
            AccessLeaseFullUpdates=5; MaxProbationTasks=20
            MinProbationCompletedTasks=9; MinEvidenceMass=1.50
            NormEscalationMode="moderate_shadow"; NonIIDAlphas=@([double]$spec.Alpha)
            Datasets=@("cifar10"); ClientCount=[int]$spec.Clients; TargetCount=[int]$spec.Attackers
            UseGenericAttackers=$true; CohortMode="mixed_maturity"
            TrustEstimators=@("dirichlet_lcb"); Variants=@($variant)
            AttackScenarios=$scenarios; ResultDirectory=$batch
            MaxLocalTrainSamples=512; ValidationSamplesPerClass=100; TestSamplesPerClass=200
            TaskDeadlineSeconds=$TaskDeadlineSeconds; BarrierTimeoutSeconds=$BarrierTimeoutSeconds
            ClientJoinTimeoutSeconds=$ClientJoinTimeoutSeconds
            ProgressOffset=$offset; ProgressTotal=$total; ProgressStartedAt=$started
            ProgressActivity="V51 Q3 access evidence"; RoundProgressPollSeconds=$RoundProgressPollSeconds
        }
        if ($built) { $runnerArguments.SkipBuild = $true }
        & $runner @runnerArguments
        if ($LASTEXITCODE -ne 0) { throw "V51 training failed: $batchTag" }
        $built = $true
        $batches += $batch
        $offset += $perBatch
    }
}

foreach ($name in $required) {
    $rows = foreach ($batch in $batches) { Import-Csv (Join-Path $batch $name) }
    $rows | Export-Csv -Path (Join-Path $root $name) -NoTypeInformation -Encoding UTF8
}
$combined = @(Import-Csv (Join-Path $root "real_fl_configuration_metrics.csv"))
if ($combined.Count -ne $total) { throw "Expected $total configuration records, found $($combined.Count)" }
$fields = @("dataset", "client_count", "attacker_count", "cohort_mode", "variant", "trust_estimator", "attack_scenario", "noniid_alpha", "repeat")
$keys = @($combined | ForEach-Object {
    $row = $_
    ($fields | ForEach-Object { [string]$row.PSObject.Properties[$_].Value }) -join "|"
})
if (@($keys | Sort-Object -Unique).Count -ne $total) { throw "Duplicate V51 configuration key" }

$resolvedRoot = (Resolve-Path $root).Path
docker run --rm --mount "type=bind,source=$resolvedRoot,target=/results" `
    node-access-flower-prototype:0.23 `
    python -m flower_prototype.analyze_q3_access_v51 $Stage `
    /results/real_fl_node_results.csv `
    /results/real_fl_round_node_events.csv `
    /results/real_fl_round_metrics.csv `
    /results/real_fl_configuration_metrics.csv `
    /results/real_fl_audit_integrity.csv `
    /results/q3_access_v51_runs.csv `
    /results/q3_access_v51_summary.csv `
    /results/q3_access_v51_paired_runs.csv `
    /results/q3_access_v51_paired_summary.csv `
    /results/q3_access_v51_promotion_guard.csv `
    /results/q3_access_v51_access_runs.csv `
    /results/q3_access_v51_access_summary.csv $effectiveRounds
if ($LASTEXITCODE -ne 0) { throw "V51 analysis or lifecycle audit failed" }
Write-Progress -Id 41 -Activity "V51 Q3 access evidence" -Completed
Write-Host "Completed V51 results: $root" -ForegroundColor Green
Write-Host "Access summary: $(Join-Path $root 'q3_access_v51_access_summary.csv')"
Write-Host "Paired outcomes: $(Join-Path $root 'q3_access_v51_paired_summary.csv')"

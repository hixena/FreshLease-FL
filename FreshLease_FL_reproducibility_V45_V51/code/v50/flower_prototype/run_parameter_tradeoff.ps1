param(
    [int]$Repeats = 1,
    [int]$Rounds = 10,
    [int]$AttackStartRound = 2,
    [int[]]$LimitedCleanUpdates = @(2, 3, 4),
    [double[]]$LimitedAggregationWeights = @(0.25, 0.50, 0.75),
    [double]$CumulativeRiskDecay = 0.50,
    [double[]]$CumulativeRiskThresholds = @(0.90),
    [double]$SelfReversalGate = 0.20,
    [double[]]$NonIIDAlphas = @(0.5),
    [string]$ResultDirectory = "",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
if ($Repeats -lt 1) { throw "Repeats must be positive" }
if ($Rounds -lt 3) { throw "Rounds must be at least 3" }
if ($AttackStartRound -lt 2 -or $AttackStartRound -gt $Rounds) {
    throw "AttackStartRound must be between 2 and Rounds"
}
foreach ($value in $LimitedCleanUpdates) {
    if ($value -lt 1 -or $value -gt $Rounds) {
        throw "Every LimitedCleanUpdates value must be between 1 and Rounds"
    }
}
foreach ($value in $LimitedAggregationWeights) {
    if ($value -le 0 -or $value -gt 1) {
        throw "Every LimitedAggregationWeights value must be in (0,1]"
    }
}
foreach ($value in $CumulativeRiskThresholds) {
    if ($value -le 0) { throw "Every CumulativeRiskThresholds value must be positive" }
}

$scriptDir = $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$root = if ($ResultDirectory) {
    [System.IO.Path]::GetFullPath($ResultDirectory)
} else {
    Join-Path $scriptDir "real_fl_results-$stamp-parameter-tradeoff"
}
if (-not (Test-Path $root)) { New-Item -ItemType Directory -Path $root | Out-Null }
$manifestPath = Join-Path $root "parameter_manifest.csv"
$manifestRows = @()

if (-not $SkipBuild) {
    docker compose -f (Join-Path $scriptDir "docker-compose.evidence-farming.yml") build
    if ($LASTEXITCODE -ne 0) { throw "Docker image build failed" }
}

function Invoke-Configuration {
    param(
        [string]$ConfigId,
        [string]$Variant,
        [string]$Scenario,
        [int]$CleanUpdates,
        [double]$Weight,
        [double]$Threshold
    )
    $childName = "$ConfigId-$($Scenario.Replace('_', '-'))"
    $child = Join-Path $root $childName
    $required = @(
        "real_fl_node_results.csv",
        "real_fl_round_node_events.csv",
        "real_fl_round_metrics.csv"
    )
    $complete = (Test-Path $child) -and (
        @($required | Where-Object { -not (Test-Path (Join-Path $child $_)) }).Count -eq 0
    )
    if ($complete) {
        Write-Host "Skipping completed configuration: $childName"
    } else {
        if (Test-Path $child) {
            throw "Incomplete result directory exists: $child. Remove only this child directory or choose a new ResultDirectory."
        }
        & (Join-Path $scriptDir "run_real_fl_matrix.ps1") `
            -Repeats $Repeats -Rounds $Rounds -AttackStartRound $AttackStartRound `
            -LimitedCleanUpdates $CleanUpdates -LimitedAggregationWeight $Weight `
            -CumulativeRiskDecay $CumulativeRiskDecay `
            -CumulativeRiskThreshold $Threshold -SelfReversalGate $SelfReversalGate `
            -NonIIDAlphas $NonIIDAlphas -Variants @($Variant) `
            -AttackScenarios @($Scenario) -ResultDirectory $child -SkipBuild
        if (-not $?) { throw "Configuration failed: $childName" }
    }
    $script:manifestRows += [PSCustomObject]@{
        config_id = $ConfigId
        variant = $Variant
        scenario = $Scenario
        attack_start_round = $AttackStartRound
        limited_clean_updates = $CleanUpdates
        limited_aggregation_weight = $Weight.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        cumulative_risk_decay = $CumulativeRiskDecay.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        cumulative_risk_threshold = $Threshold.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        self_reversal_gate = $SelfReversalGate.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        child_dir = $childName
    }
}

$baselineClean = $LimitedCleanUpdates[0]
$baselineWeight = $LimitedAggregationWeights[0]
$baselineThreshold = $CumulativeRiskThresholds[0]
foreach ($scenario in @("gradual_drift_betrayal", "benign_concept_drift")) {
    Invoke-Configuration "baseline_full" "full" $scenario `
        $baselineClean $baselineWeight $baselineThreshold
}

foreach ($clean in $LimitedCleanUpdates) {
    foreach ($weight in $LimitedAggregationWeights) {
        foreach ($threshold in $CumulativeRiskThresholds) {
            $weightTag = $weight.ToString("0.00", [System.Globalization.CultureInfo]::InvariantCulture).Replace(".", "p")
            $thresholdTag = $threshold.ToString("0.00", [System.Globalization.CultureInfo]::InvariantCulture).Replace(".", "p")
            $configId = "pf-c$clean-w$weightTag-t$thresholdTag"
            foreach ($scenario in @("gradual_drift_betrayal", "benign_concept_drift")) {
                Invoke-Configuration $configId "progressive_full" $scenario `
                    $clean $weight $threshold
            }
        }
    }
}

$manifestRows | Export-Csv -Path $manifestPath -NoTypeInformation -Encoding UTF8
docker run --rm `
    --mount "type=bind,source=$root,target=/results" `
    node-access-flower-prototype:0.19 `
    python -m flower_prototype.analyze_parameter_tradeoff `
    /results /results/parameter_manifest.csv `
    /results/parameter_tradeoff_runs.csv /results/parameter_tradeoff_summary.csv
if ($LASTEXITCODE -ne 0) { throw "Parameter-tradeoff analysis failed" }

Write-Host "Parameter-tradeoff results: $root"
Write-Host "Run-level pairs: $(Join-Path $root 'parameter_tradeoff_runs.csv')"
Write-Host "Safety-utility summary: $(Join-Path $root 'parameter_tradeoff_summary.csv')"

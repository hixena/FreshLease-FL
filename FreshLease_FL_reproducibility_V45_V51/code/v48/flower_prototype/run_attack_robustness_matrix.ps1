param(
    [int]$Repeats = 1,
    [int]$Rounds = 10,
    [int[]]$AttackStartRounds = @(2, 5, 8),
    [double[]]$GradualSignFlipScales = @(0.15, 0.35, 0.70),
    [double[]]$NonIIDAlphas = @(0.5),
    [int]$LimitedCleanUpdates = 3,
    [double]$LimitedAggregationWeight = 0.75,
    [double]$CumulativeRiskDecay = 0.50,
    [double]$CumulativeRiskThreshold = 0.72,
    [double]$SelfReversalGate = 0.20,
    [string]$ResultDirectory = "",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
if ($Repeats -lt 1) { throw "Repeats must be positive" }
if ($Rounds -lt 3) { throw "Rounds must be at least 3" }
foreach ($value in $AttackStartRounds) {
    if ($value -lt 2 -or $value -gt $Rounds) {
        throw "Every AttackStartRounds value must be between 2 and Rounds"
    }
}
foreach ($value in $GradualSignFlipScales) {
    if ($value -le 0) { throw "Every GradualSignFlipScales value must be positive" }
}

$scriptDir = $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$root = if ($ResultDirectory) {
    [System.IO.Path]::GetFullPath($ResultDirectory)
} else {
    Join-Path $scriptDir "real_fl_results-$stamp-attack-robustness"
}
if (-not (Test-Path $root)) { New-Item -ItemType Directory -Path $root | Out-Null }
$manifestPath = Join-Path $root "attack_robustness_manifest.csv"
$manifestRows = @()

if (-not $SkipBuild) {
    docker compose -f (Join-Path $scriptDir "docker-compose.evidence-farming.yml") build
    if ($LASTEXITCODE -ne 0) { throw "Docker image build failed" }
}

$previousScale = $env:GRADUAL_SIGN_FLIP_SCALE
try {
    foreach ($start in $AttackStartRounds) {
        foreach ($scale in $GradualSignFlipScales) {
            $scaleText = $scale.ToString(
                [System.Globalization.CultureInfo]::InvariantCulture
            )
            $scaleTag = $scale.ToString(
                "0.00", [System.Globalization.CultureInfo]::InvariantCulture
            ).Replace(".", "p")
            $configId = "start$start-scale$scaleTag"
            foreach ($variant in @("progressive_full", "progressive_no_cumulative", "full")) {
                $variantTag = $variant.Replace("_", "-")
                $childName = "$configId-$variantTag"
                $child = Join-Path $root $childName
                $required = @(
                    "real_fl_node_results.csv",
                    "real_fl_round_node_events.csv",
                    "real_fl_round_metrics.csv"
                )
                $complete = (Test-Path $child) -and (
                    @($required | Where-Object {
                        -not (Test-Path (Join-Path $child $_))
                    }).Count -eq 0
                )
                if ($complete) {
                    Write-Host "Skipping completed configuration: $childName"
                } else {
                    if (Test-Path $child) {
                        throw "Incomplete result directory exists: $child. Remove only this child directory or choose a new ResultDirectory."
                    }
                    $env:GRADUAL_SIGN_FLIP_SCALE = $scaleText
                    & (Join-Path $scriptDir "run_real_fl_matrix.ps1") `
                        -Repeats $Repeats -Rounds $Rounds -AttackStartRound $start `
                        -LimitedCleanUpdates $LimitedCleanUpdates `
                        -LimitedAggregationWeight $LimitedAggregationWeight `
                        -CumulativeRiskDecay $CumulativeRiskDecay `
                        -CumulativeRiskThreshold $CumulativeRiskThreshold `
                        -SelfReversalGate $SelfReversalGate `
                        -NonIIDAlphas $NonIIDAlphas -Variants @($variant) `
                        -AttackScenarios @("gradual_drift_betrayal") `
                        -ResultDirectory $child -SkipBuild
                    if (-not $?) { throw "Configuration failed: $childName" }
                }
                $manifestRows += [PSCustomObject]@{
                    config_id = $configId
                    variant = $variant
                    attack_start_round = $start
                    gradual_sign_flip_scale = $scaleText
                    child_dir = $childName
                }
            }
        }
    }
}
finally {
    if ($previousScale) { $env:GRADUAL_SIGN_FLIP_SCALE = $previousScale }
    else { Remove-Item "Env:GRADUAL_SIGN_FLIP_SCALE" -ErrorAction SilentlyContinue }
}

$manifestRows | Export-Csv -Path $manifestPath -NoTypeInformation -Encoding UTF8
docker run --rm `
    --mount "type=bind,source=$root,target=/results" `
    node-access-flower-prototype:0.19 `
    python -m flower_prototype.analyze_attack_robustness `
    /results /results/attack_robustness_manifest.csv `
    /results/attack_robustness_runs.csv /results/attack_robustness_summary.csv
if ($LASTEXITCODE -ne 0) { throw "Attack-robustness analysis failed" }

Write-Host "Attack-robustness results: $root"
Write-Host "Per-run results: $(Join-Path $root 'attack_robustness_runs.csv')"
Write-Host "Robustness summary: $(Join-Path $root 'attack_robustness_summary.csv')"

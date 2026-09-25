param(
    [ValidateSet("onboarding", "post_admission")]
    [string]$Stage = "onboarding",
    [int]$Repeats = 3,
    [int]$Rounds = 20,
    [double[]]$NonIIDAlphas = @(0.1, 0.5, 10.0)
)

$ErrorActionPreference = "Stop"
if ($Stage -eq "onboarding") {
    # Isolate probation checks: all non-naive variants retain identical online screening.
    $variants = @("full", "no_diversity", "no_repeat_decay", "no_result_verification", "naive")
    $scenarios = @("single_type_farming", "false_quality_reporting", "diverse_then_repeat_farming")
} else {
    # Hold onboarding fixed between full and no_online_revalidation.
    # A backdoor may evade the online screening; the outcome is an empirical question.
    $variants = @("full", "no_online_revalidation", "naive", "oracle_filter")
    $scenarios = @("diverse_then_repeat_farming", "diverse_then_repeat_backdoor")
}

& (Join-Path $PSScriptRoot "run_real_fl_matrix.ps1") `
    -Repeats $Repeats -Rounds $Rounds -NonIIDAlphas $NonIIDAlphas `
    -Variants $variants -AttackScenarios $scenarios
if (-not $?) { throw "Real-FL $Stage matrix failed" }

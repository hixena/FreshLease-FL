param(
    [int]$Repeats = 3
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$composeFile = Join-Path $scriptDir "docker-compose.evidence-farming.yml"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resultDir = Join-Path $scriptDir "container_results-$stamp"
$csvPath = Join-Path $resultDir "node_results.csv"
$variants = @(
    "full",
    "no_diversity",
    "no_repeat_decay",
    "no_semantic_separation",
    "naive",
    "no_budget"
)

New-Item -ItemType Directory -Path $resultDir | Out-Null
docker compose -f $composeFile build
if ($LASTEXITCODE -ne 0) {
    throw "Docker image build failed; experiment matrix was not started"
}

try {
    foreach ($variant in $variants) {
        for ($repeat = 0; $repeat -lt $Repeats; $repeat++) {
            $env:MECHANISM_VARIANT = $variant
            $env:EXPERIMENT_SEED = "$repeat"
            $safeVariant = $variant.Replace("_", "-")
            $projectName = "evidence-$safeVariant-r$repeat"
            $runCompleted = $false
            for ($attempt = 1; $attempt -le 2 -and -not $runCompleted; $attempt++) {
                Write-Host "Running variant=$variant repeat=$repeat attempt=$attempt"
                try {
                    docker compose -p $projectName -f $composeFile up -d --no-build
                    $serverId = docker compose -p $projectName -f $composeFile ps -q flower-server
                    if (-not $serverId) {
                        throw "Flower server container was not created"
                    }
                    $serverProcessExitCode = (docker wait $serverId).Trim()
                    Start-Sleep -Seconds 2
                    if ($serverProcessExitCode -ne "0") {
                        throw "Flower server exited with code $serverProcessExitCode"
                    }

                    $summary = Invoke-RestMethod -Uri "http://localhost:8000/experiment/summary"
                    if (-not $summary.barrier.released -or $summary.barrier.finished_nodes -ne 6) {
                        throw "Onboarding barrier did not finish all 6 nodes"
                    }
                    foreach ($node in $summary.nodes) {
                        if ($null -eq $node.onboarding_duration_seconds) {
                            throw "Node $($node.node_id) did not submit onboarding completion"
                        }
                        if ($node.access_state -eq "ADMITTED" -and $node.flower_fit_events -ne 20) {
                            throw "Admitted node $($node.node_id) did not finish 20 FIT rounds"
                        }
                    }

                    $jsonPath = Join-Path $resultDir "$variant-r$repeat-summary.json"
                    $summary | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $jsonPath
                    foreach ($node in $summary.nodes) {
                        [PSCustomObject]@{
                            variant = $variant
                            repeat = $repeat
                            node_id = $node.node_id
                            profile = $node.profile
                            access_state = $node.access_state
                            trust_score = $node.trust_score
                            completed_tasks = $node.completed_tasks
                            evidence_types = $node.evidence_types
                            evidence_mass = $node.evidence_mass
                            graduation_ready = $node.graduation_ready
                            budget_exhausted = $node.budget_exhausted
                            onboarding_duration_seconds = $node.onboarding_duration_seconds
                            flower_fit_events = $node.flower_fit_events
                            flower_evaluate_events = $node.flower_evaluate_events
                        } | Export-Csv -Path $csvPath -NoTypeInformation -Append -Encoding UTF8
                    }
                    docker compose -p $projectName -f $composeFile logs --no-color |
                        Set-Content -Encoding UTF8 (Join-Path $resultDir "$variant-r$repeat.log")
                    $runCompleted = $true
                }
                catch {
                    docker compose -p $projectName -f $composeFile logs --no-color 2>&1 |
                        Set-Content -Encoding UTF8 (Join-Path $resultDir "$variant-r$repeat-attempt$attempt-failed.log")
                    if ($attempt -ge 2) {
                        throw
                    }
                    Write-Warning "Run failed; retrying the same seed once: $($_.Exception.Message)"
                }
                finally {
                    docker compose -p $projectName -f $composeFile down -v
                }
            }
        }
    }
}
finally {
    Remove-Item Env:MECHANISM_VARIANT -ErrorAction SilentlyContinue
    Remove-Item Env:EXPERIMENT_SEED -ErrorAction SilentlyContinue
}

$summaryPath = Join-Path $resultDir "variant_summary.csv"
$data = Import-Csv $csvPath
$data | Group-Object variant, profile | ForEach-Object {
    $group = $_.Group
    [PSCustomObject]@{
        variant = $group[0].variant
        profile = $group[0].profile
        runs = $group.Count
        admitted_rate = @($group | Where-Object access_state -eq "ADMITTED").Count / $group.Count
        quarantine_rate = @($group | Where-Object access_state -eq "QUARANTINE").Count / $group.Count
        mean_completed_tasks = ($group | Measure-Object completed_tasks -Average).Average
        mean_flower_fit_events = ($group | Measure-Object flower_fit_events -Average).Average
    }
} | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8

$formalSummaryPath = Join-Path $resultDir "formal_summary.csv"
$pairedEffectsPath = Join-Path $resultDir "paired_effects.csv"
docker run --rm `
    --mount "type=bind,source=$resultDir,target=/results" `
    node-access-flower-prototype:0.13 `
    python -m flower_prototype.analyze_container_results `
    /results/node_results.csv /results/formal_summary.csv /results/paired_effects.csv

Write-Host "Completed. Results: $resultDir"
Write-Host "CSV: $csvPath"
Write-Host "Summary: $summaryPath"
Write-Host "Formal summary: $formalSummaryPath"
Write-Host "Paired effects: $pairedEffectsPath"

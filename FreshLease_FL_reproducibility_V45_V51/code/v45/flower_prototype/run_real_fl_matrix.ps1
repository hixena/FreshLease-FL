param(
    [int]$Repeats = 3,
    [int]$Rounds = 20,
    [int]$AttackStartRound = 1,
    [Alias("LimitedCleanUpdates")]
    [int]$LimitedObservationUpdates = 2,
    [double]$LimitedAggregationWeight = 0.25,
    [int]$AccessLeaseFullUpdates = 5,
    [int]$MaxProbationTasks = 20,
    [int]$MinProbationCompletedTasks = 9,
    [double]$MinEvidenceMass = 1.50,
    [double]$CumulativeRiskDecay = 0.50,
    [double]$CumulativeRiskThreshold = 0.90,
    [double]$SelfReversalGate = 0.20,
    [ValidateSet("enforce", "moderate_shadow")]
    [string]$NormEscalationMode = "enforce",
    [int]$TaskDeadlineSeconds = 30,
    [int]$BarrierTimeoutSeconds = 600,
    [int]$ClientJoinTimeoutSeconds = 300,
    [int]$ProgressOffset = 0,
    [int]$ProgressTotal = 0,
    [datetime]$ProgressStartedAt = [datetime]::MinValue,
    [string]$ProgressActivity = "Real-FL experiment matrix",
    [int]$RoundProgressPollSeconds = 5,
    [switch]$SkipBuild,
    [string]$ResultDirectory = "",
    [int]$ClientCount = 4,
    [ValidateRange(1, 2)]
    [int]$TargetCount = 1,
    [switch]$UseGenericAttackers,
    [ValidateSet("synchronous_cold_start", "mixed_maturity")]
    [string]$CohortMode = "synchronous_cold_start",
    [string[]]$Datasets = @("digits"),
    [double[]]$NonIIDAlphas = @(0.1, 0.5, 10.0),
    [string[]]$TrustEstimators = @("dirichlet_lcb"),
    [string[]]$Variants = @(
        "full", "no_online_revalidation", "naive", "oracle_filter"
    ),
    [string[]]$AttackScenarios = @(
        "single_type_farming",
        "diverse_then_repeat_farming",
        "attestation_camouflage"
    )
)

$ErrorActionPreference = "Stop"
if ($AttackStartRound -lt 1 -or $AttackStartRound -gt $Rounds) {
    throw "AttackStartRound must be between 1 and Rounds"
}
if ($LimitedObservationUpdates -lt 1) {
    throw "LimitedObservationUpdates must be positive"
}
if ($AccessLeaseFullUpdates -lt 1) {
    throw "AccessLeaseFullUpdates must be positive"
}
if ($MaxProbationTasks -lt 1 -or $MinProbationCompletedTasks -lt 1) {
    throw "Probation task limits must be positive"
}
if ($MinProbationCompletedTasks -gt $MaxProbationTasks) {
    throw "MinProbationCompletedTasks cannot exceed MaxProbationTasks"
}
if ($MinEvidenceMass -le 0) { throw "MinEvidenceMass must be positive" }
if ($ClientCount -lt 4 -or $ClientCount -gt 20) {
    throw "ClientCount must be between 4 and 20 for this compose file"
}
if (($ClientCount - $TargetCount) -lt 3) {
    throw "At least three honest clients are required"
}
if ($LimitedAggregationWeight -le 0 -or $LimitedAggregationWeight -gt 1) {
    throw "LimitedAggregationWeight must be in (0,1]"
}
if ($TaskDeadlineSeconds -lt 1 -or $BarrierTimeoutSeconds -lt 1 -or
    $ClientJoinTimeoutSeconds -lt 1) {
    throw "Task and barrier timeouts must be positive"
}
if ($RoundProgressPollSeconds -lt 1) {
    throw "RoundProgressPollSeconds must be positive"
}
$allowedTrustEstimators = @("dirichlet_lcb", "dirichlet_mean", "beta_mean")
$allowedDatasets = @("digits", "fashion_mnist")
foreach ($dataset in $Datasets) {
    if ($dataset -notin $allowedDatasets) { throw "Unknown Dataset: $dataset" }
}
foreach ($trustEstimator in $TrustEstimators) {
    if ($trustEstimator -notin $allowedTrustEstimators) {
        throw "Unknown TrustEstimator: $trustEstimator"
    }
}
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$composeFile = Join-Path $scriptDir "docker-compose.evidence-farming.yml"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resultDir = if ($ResultDirectory) { $ResultDirectory } else { Join-Path $scriptDir "real_fl_results-$stamp" }
if (-not [IO.Path]::IsPathRooted($resultDir)) {
    $resultDir = Join-Path (Get-Location).Path $resultDir
}
$resultDir = [IO.Path]::GetFullPath($resultDir)
$roundCsv = Join-Path $resultDir "real_fl_round_metrics.csv"
$roundNodeCsv = Join-Path $resultDir "real_fl_round_node_events.csv"
$nodeCsv = Join-Path $resultDir "real_fl_node_results.csv"
$configurationCsv = Join-Path $resultDir "real_fl_configuration_metrics.csv"
$localConfigurationCount = (
    $Variants.Count * $TrustEstimators.Count * $AttackScenarios.Count *
    $NonIIDAlphas.Count * $Datasets.Count * $Repeats
)
if ($ProgressTotal -lt 1) { $ProgressTotal = $localConfigurationCount }
if ($ProgressStartedAt -eq [datetime]::MinValue) { $ProgressStartedAt = Get-Date }
$localCompleted = 0

function Show-MatrixProgress {
    param([string]$Activity, [string]$Detail, [double]$Completed)
    $percent = if ($ProgressTotal -gt 0) {
        [math]::Min(100, [math]::Round(100 * $Completed / $ProgressTotal, 1))
    } else { 0 }
    $elapsed = (Get-Date) - $ProgressStartedAt
    $etaText = "estimating"
    if ($Completed -gt 0) {
        $secondsPerConfiguration = $elapsed.TotalSeconds / $Completed
        $remainingSeconds = [math]::Max(
            0, $secondsPerConfiguration * ($ProgressTotal - $Completed)
        )
        $etaText = [TimeSpan]::FromSeconds($remainingSeconds).ToString("hh\:mm\:ss")
    }
    $status = "$Completed/$ProgressTotal | elapsed=$($elapsed.ToString('hh\:mm\:ss')) | ETA=$etaText | $Detail"
    Write-Progress -Id 41 -Activity $Activity -Status $status -PercentComplete $percent
}

if (Test-Path $resultDir) { throw "Result directory already exists: $resultDir" }
New-Item -ItemType Directory -Path $resultDir | Out-Null
$env:DATA_MODE = "digits"
$env:FLOWER_ROUNDS = "$Rounds"
$previousAttackStartRound = $env:ATTACK_START_ROUND
$env:ATTACK_START_ROUND = "$AttackStartRound"
$previousLimitedCleanUpdates = $env:MIN_LIMITED_CLEAN_UPDATES
$previousLimitedObservationUpdates = $env:MIN_LIMITED_OBSERVATION_UPDATES
$env:MIN_LIMITED_CLEAN_UPDATES = "$LimitedObservationUpdates"
$env:MIN_LIMITED_OBSERVATION_UPDATES = "$LimitedObservationUpdates"
$previousAccessLeaseFullUpdates = $env:ACCESS_LEASE_FULL_UPDATES
$env:ACCESS_LEASE_FULL_UPDATES = "$AccessLeaseFullUpdates"
$previousMaxProbationTasks = $env:MAX_PROBATION_TASKS
$previousMinProbationCompletedTasks = $env:MIN_PROBATION_COMPLETED_TASKS
$previousMinEvidenceMass = $env:MIN_EVIDENCE_MASS
$env:MAX_PROBATION_TASKS = "$MaxProbationTasks"
$env:MIN_PROBATION_COMPLETED_TASKS = "$MinProbationCompletedTasks"
$env:MIN_EVIDENCE_MASS = $MinEvidenceMass.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$previousLimitedAggregationWeight = $env:LIMITED_AGGREGATION_WEIGHT
$previousCumulativeRiskDecay = $env:CUMULATIVE_RISK_DECAY
$previousCumulativeRiskThreshold = $env:CUMULATIVE_RISK_THRESHOLD
$previousSelfReversalGate = $env:SELF_REVERSAL_GATE
$previousNormEscalationMode = $env:NORM_ESCALATION_MODE
$env:LIMITED_AGGREGATION_WEIGHT = $LimitedAggregationWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$env:CUMULATIVE_RISK_DECAY = $CumulativeRiskDecay.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$env:CUMULATIVE_RISK_THRESHOLD = $CumulativeRiskThreshold.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$env:SELF_REVERSAL_GATE = $SelfReversalGate.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$env:NORM_ESCALATION_MODE = $NormEscalationMode
$env:EXPECTED_EXPERIMENT_NODES = "$ClientCount"
$env:NUM_DATA_PARTITIONS = "$ClientCount"
$env:FLOWER_MIN_CLIENTS = "$ClientCount"
$env:TARGET_PARTITION_ID = "$($ClientCount - 1)"
$env:TARGET_PARTITION_ID_1 = "$($ClientCount - $TargetCount)"
$env:TARGET_PARTITION_ID_2 = "$($ClientCount - 1)"
$previousIncumbentNodeIds = $env:INCUMBENT_NODE_IDS
$env:INCUMBENT_NODE_IDS = if ($CohortMode -eq "mixed_maturity") {
    ((1..($ClientCount - $TargetCount) | ForEach-Object { "honest-$_" }) -join ",")
} else {
    ""
}
$previousControlToken = $env:FL_CONTROL_TOKEN
$previousTrustEstimator = $env:TRUST_ESTIMATOR
$previousDatasetName = $env:DATASET_NAME
$previousTaskDeadline = $env:TASK_DEADLINE_SECONDS
$previousBarrierTimeout = $env:BARRIER_TIMEOUT_SECONDS
$previousClientJoinTimeout = $env:CLIENT_JOIN_TIMEOUT_SECONDS
$env:TASK_DEADLINE_SECONDS = "$TaskDeadlineSeconds"
$env:BARRIER_TIMEOUT_SECONDS = "$BarrierTimeoutSeconds"
$env:CLIENT_JOIN_TIMEOUT_SECONDS = "$ClientJoinTimeoutSeconds"
if (-not $previousControlToken) {
    $env:FL_CONTROL_TOKEN = [guid]::NewGuid().ToString("N")
}

try {
    if (-not $SkipBuild) {
        docker compose -f $composeFile build
        if ($LASTEXITCODE -ne 0) {
            throw "Docker image build failed; real-FL matrix was not started"
        }
    }
    foreach ($dataset in $Datasets) {
      $env:DATASET_NAME = $dataset
      foreach ($alpha in $NonIIDAlphas) {
        $env:NONIID_ALPHA = $alpha.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        foreach ($scenario in $AttackScenarios) {
            $env:ATTACK_SCENARIO = $scenario
            $env:ATTACKER_PROFILE = $scenario
            $targetService = switch ($scenario) {
                "single_type_farming" { "single-type-farming" }
                "three_type_repeat_farming" { "three-type-repeat-farming" }
                "diverse_then_repeat_farming" { "diverse-repeat-farming" }
                "gradual_drift_betrayal" { "gradual-drift-betrayal" }
                "benign_concept_drift" { "benign-concept-drift" }
                "diverse_then_repeat_backdoor" { "diverse-repeat-backdoor" }
                "attestation_camouflage" { "attestation-camouflage" }
                "false_quality_reporting" { "false-quality-reporting" }
                "shadow_exact_replay" { "shadow-exact-replay" }
                default { throw "Unknown target attack scenario: $scenario" }
            }
            $honestServices = 1..($ClientCount - $TargetCount) | ForEach-Object { "honest-$_" }
            $targetServices = if ($UseGenericAttackers -or $TargetCount -gt 1) {
                1..$TargetCount | ForEach-Object { "attacker-$_" }
            } else { @($targetService) }
            $services = @("access-controller", "flower-server") + $honestServices + $targetServices
            foreach ($variant in $Variants) {
              foreach ($trustEstimator in $TrustEstimators) {
                for ($repeat = 0; $repeat -lt $Repeats; $repeat++) {
                $env:MECHANISM_VARIANT = $variant
                $env:AGGREGATION_RULE = switch ($variant) {
                    "fltrust" { "fltrust" }
                    "progressive_fltrust" { "fltrust" }
                    "trimmed_mean" { "trimmed_mean" }
                    "progressive_trimmed_mean" { "trimmed_mean" }
                    "coordinate_median" { "coordinate_median" }
                    "rffl_reputation" { "rffl_reputation" }
                    default { "fedavg" }
                }
                $env:TRUST_ESTIMATOR = $trustEstimator
                $env:EXPERIMENT_SEED = "$repeat"
                $safeAlpha = $env:NONIID_ALPHA.Replace(".", "p")
                $safeVariant = $variant.Replace("_", "-")
                $safeScenario = $scenario.Replace("_", "-")
                $safeEstimator = $trustEstimator.Replace("_", "-")
                $safeDataset = $dataset.Replace("_", "-")
                $safeCohort = $CohortMode.Replace("_", "-")
                $projectName = "realfl-$safeDataset-$safeVariant-$safeEstimator-$safeScenario-$safeCohort-k$TargetCount-a$safeAlpha-r$repeat"
                $currentNumber = $ProgressOffset + $localCompleted + 1
                $detail = "dataset=$dataset variant=$variant attack=$scenario clients=$ClientCount attackers=$TargetCount seed=$repeat"
                Show-MatrixProgress $ProgressActivity $detail ($currentNumber - 1)
                Write-Host "Running dataset=$dataset variant=$variant estimator=$trustEstimator attack=$scenario cohort=$CohortMode alpha=$($env:NONIID_ALPHA) repeat=$repeat"
                try {
                    $configurationStartedAt = Get-Date
                    docker compose -p $projectName -f $composeFile up -d --no-build $services
                    $serverId = docker compose -p $projectName -f $composeFile ps -q flower-server
                    if (-not $serverId) {
                        throw "Flower server container was not created"
                    }
                    $lastReportedRound = 0
                    while ($true) {
                        $serverRunning = (docker inspect -f '{{.State.Running}}' $serverId).Trim()
                        if ($LASTEXITCODE -ne 0) {
                            throw "Could not inspect Flower server container"
                        }
                        if ($serverRunning -ne "true") { break }
                        $serverLogs = (& cmd.exe /d /c "docker logs $serverId 2>nul") -join "`n"
                        $roundMatches = [regex]::Matches(
                            [string]$serverLogs, 'online_access round=(\d+)'
                        )
                        if ($roundMatches.Count -gt 0) {
                            $maximumRound = $roundMatches |
                                ForEach-Object { [int]$_.Groups[1].Value } |
                                Measure-Object -Maximum
                            $currentRound = [int]$maximumRound.Maximum
                            if ($currentRound -gt $lastReportedRound) {
                                $lastReportedRound = $currentRound
                                $roundFraction = [Math]::Min(
                                    1.0, [double]$currentRound / [double]$Rounds
                                )
                                $fractionalCompleted = ($currentNumber - 1) + $roundFraction
                                $roundDetail = "$detail | round=$currentRound/$Rounds"
                                Show-MatrixProgress $ProgressActivity $roundDetail $fractionalCompleted
                                Write-Host (
                                    "Progress {0}/{1}: round {2}/{3} | {4}" -f `
                                    $currentNumber, $ProgressTotal, $currentRound, $Rounds, $detail
                                ) -ForegroundColor Cyan
                            }
                        }
                        Start-Sleep -Seconds $RoundProgressPollSeconds
                    }
                    $serverExit = (docker inspect -f '{{.State.ExitCode}}' $serverId).Trim()
                    Start-Sleep -Seconds 2
                    if ($serverExit -ne "0") {
                        throw "Flower server exited with code $serverExit"
                    }

                    $containerMetrics = Join-Path $resultDir "metrics-current.csv"
                    docker cp "${serverId}:/results/global_metrics.csv" $containerMetrics
                    if ($LASTEXITCODE -ne 0) {
                        throw "Could not copy centralized evaluation metrics"
                    }
                    $metrics = Import-Csv $containerMetrics
                    foreach ($row in $metrics) {
                        [PSCustomObject]@{
                            dataset = $dataset
                            client_count = $ClientCount
                            attacker_count = $TargetCount
                            cohort_mode = $CohortMode
                            variant = $variant
                            trust_estimator = $trustEstimator
                            attack_scenario = $scenario
                            noniid_alpha = $env:NONIID_ALPHA
                            repeat = $repeat
                            attack_start_round = $AttackStartRound
                            limited_clean_updates = $LimitedObservationUpdates
                            limited_observation_updates = $LimitedObservationUpdates
                            limited_aggregation_weight = $LimitedAggregationWeight
                            access_lease_full_updates = $AccessLeaseFullUpdates
                            max_probation_tasks = $MaxProbationTasks
                            min_probation_completed_tasks = $MinProbationCompletedTasks
                            min_evidence_mass = $MinEvidenceMass
                            round = $row.round
                            loss = $row.loss
                            accuracy = $row.accuracy
                            backdoor_asr = $row.backdoor_asr
                        } | Export-Csv -Path $roundCsv -NoTypeInformation -Append -Encoding UTF8
                    }
                    Remove-Item $containerMetrics

                    $containerRoundNodes = Join-Path $resultDir "round-nodes-current.csv"
                    docker cp "${serverId}:/results/round_node_events.csv" $containerRoundNodes
                    if ($LASTEXITCODE -ne 0) {
                        throw "Could not copy per-round node events"
                    }
                    $roundNodes = @(Import-Csv $containerRoundNodes)
                    foreach ($row in $roundNodes) {
                        [PSCustomObject]@{
                            dataset = $dataset
                            client_count = $ClientCount
                            attacker_count = $TargetCount
                            cohort_mode = $CohortMode
                            variant = $variant
                            trust_estimator = $trustEstimator
                            attack_scenario = $scenario
                            noniid_alpha = $env:NONIID_ALPHA
                            repeat = $repeat
                            attack_start_round = $AttackStartRound
                            limited_clean_updates = $LimitedObservationUpdates
                            limited_observation_updates = $LimitedObservationUpdates
                            limited_aggregation_weight = $LimitedAggregationWeight
                            access_lease_full_updates = $AccessLeaseFullUpdates
                            max_probation_tasks = $MaxProbationTasks
                            min_probation_completed_tasks = $MinProbationCompletedTasks
                            min_evidence_mass = $MinEvidenceMass
                            round = $row.round
                            cid = $row.cid
                            node_id = $row.node_id
                            selected = $row.selected
                            returned = $row.returned
                            signature_verified = $row.signature_verified
                            passed_screening = $row.passed_screening
                            accepted_for_aggregation = $row.accepted_for_aggregation
                            aggregated = $row.aggregated
                            reason = $row.reason
                            access_state_before = $row.access_state_before
                            access_state_after = $row.access_state_after
                            training_evidence_outcome = $row.training_evidence_outcome
                            access_transition = $row.access_transition
                            penalty_debt = $row.penalty_debt
                            aggregation_weight = $row.aggregation_weight
                            instantaneous_risk = $row.instantaneous_risk
                            cumulative_risk = $row.cumulative_risk
                            cohort_opposition = $row.cohort_opposition
                            self_reversal = $row.self_reversal
                            candidate_validation_loss = $row.candidate_validation_loss
                            global_validation_loss = $row.global_validation_loss
                            validation_loss_delta = $row.validation_loss_delta
                            validation_loss_relative_delta = $row.validation_loss_relative_delta
                            validation_loss_flag = $row.validation_loss_flag
                            update_norm = $row.update_norm
                            median_update_norm = $row.median_update_norm
                            norm_threshold = $row.norm_threshold
                            extreme_norm_threshold = $row.extreme_norm_threshold
                            norm_ratio = $row.norm_ratio
                            norm_strike_count = $row.norm_strike_count
                            norm_screening_outcome = $row.norm_screening_outcome
                            norm_escalation_mode = $row.norm_escalation_mode
                            benign_drift_fraction = $row.benign_drift_fraction
                            aggregation_rule = $row.aggregation_rule
                            aggregation_trust_score = $row.aggregation_trust_score
                            aggregation_norm_scale = $row.aggregation_norm_scale
                            aggregation_effective_weight = $row.aggregation_effective_weight
                            aggregation_coordinate_retention_rate = $row.aggregation_coordinate_retention_rate
                            aggregation_reputation_score = $row.aggregation_reputation_score
                            aggregation_reputation_contribution = $row.aggregation_reputation_contribution
                            aggregation_reputation_removed = $row.aggregation_reputation_removed
                            update_payload_bytes = $row.update_payload_bytes
                            server_round_processing_seconds = $row.server_round_processing_seconds
                        } | Export-Csv -Path $roundNodeCsv -NoTypeInformation -Append -Encoding UTF8
                    }
                    Remove-Item $containerRoundNodes

                    $summary = Invoke-RestMethod -Uri "http://localhost:8000/experiment/summary"
                    if (-not $summary.barrier.released -or $summary.barrier.finished_nodes -ne $ClientCount) {
                        throw "Onboarding barrier did not finish all $ClientCount nodes"
                    }
                    $firstRound = @($roundNodes | Where-Object { [int]$_.round -eq 1 })
                    if ($firstRound.Count -ne [int]$summary.barrier.admitted_nodes) {
                        throw "Round 1 sampled $($firstRound.Count) of $($summary.barrier.admitted_nodes) admitted clients"
                    }
                    $missingFirstRound = @($firstRound | Where-Object { [int]$_.returned -ne 1 })
                    if ($missingFirstRound.Count -gt 0) {
                        throw "Round 1 has $($missingFirstRound.Count) sampled clients without an update; inspect the log"
                    }
                    foreach ($node in $summary.nodes) {
                        [PSCustomObject]@{
                            dataset = $dataset
                            client_count = $ClientCount
                            attacker_count = $TargetCount
                            cohort_mode = $CohortMode
                            variant = $variant
                            trust_estimator = $trustEstimator
                            attack_scenario = $scenario
                            noniid_alpha = $env:NONIID_ALPHA
                            repeat = $repeat
                            attack_start_round = $AttackStartRound
                            limited_clean_updates = $LimitedObservationUpdates
                            limited_observation_updates = $LimitedObservationUpdates
                            limited_aggregation_weight = $LimitedAggregationWeight
                            max_probation_tasks = $MaxProbationTasks
                            min_probation_completed_tasks = $MinProbationCompletedTasks
                            min_evidence_mass = $MinEvidenceMass
                            configured_access_lease_full_updates = $AccessLeaseFullUpdates
                            node_id = $node.node_id
                            profile = $node.profile
                            cohort_role = $node.cohort_role
                            initial_access_state = $node.initial_access_state
                            access_state = $node.access_state
                            completed_tasks = $node.completed_tasks
                            credited_independent_tasks = $node.credited_independent_tasks
                            verified_source_records = $node.verified_source_records
                            evidence_types = $node.evidence_types
                            evidence_mass = $node.evidence_mass
                            minimum_evidence_mass_required = $node.minimum_evidence_mass_required
                            trust_score = $node.trust_score
                            history_mean = $node.history_mean
                            history_decision_score = $node.history_decision_score
                            history_lcb = $node.history_lcb
                            history_std = $node.history_std
                            history_evidence_maturity = $node.history_evidence_maturity
                            training_clean_evidence = $node.training_clean_evidence
                            structurally_accepted_training_updates = $node.structurally_accepted_training_updates
                            structurally_accepted_observation_updates = $node.structurally_accepted_observation_updates
                            limited_clean_updates_required = $node.limited_clean_updates_required
                            limited_observation_updates_required = $node.limited_observation_updates_required
                            freshness_lease_enabled = $node.freshness_lease_enabled
                            access_lease_full_updates = $node.access_lease_full_updates
                            penalty_debt = $node.penalty_debt
                            hard_failures = $node.hard_failures
                            onboarding_duration_seconds = $node.onboarding_duration_seconds
                            flower_fit_events = $node.flower_fit_events
                            aggregated_fit_events = $node.aggregated_fit_events
                            aggregated_weight_mass = $node.aggregated_weight_mass
                            revoked_round = $node.revoked_round
                            revocation_reason = $node.revocation_reason
                            norm_escalation_mode = $NormEscalationMode
                        } | Export-Csv -Path $nodeCsv -NoTypeInformation -Append -Encoding UTF8
                    }
                    $controllerStateBytes = docker compose -p $projectName -f $composeFile exec -T access-controller `
                        python -c "import os; print(sum(os.path.getsize('/data/'+n) for n in os.listdir('/data')))"
                    if ($LASTEXITCODE -ne 0) { throw "Could not measure controller state bytes" }
                    $configurationSeconds = ((Get-Date) - $configurationStartedAt).TotalSeconds
                    [PSCustomObject]@{
                        dataset = $dataset
                        client_count = $ClientCount
                        attacker_count = $TargetCount
                        attacker_fraction = $TargetCount / $ClientCount
                        cohort_mode = $CohortMode
                        variant = $variant
                        trust_estimator = $trustEstimator
                        attack_scenario = $scenario
                        noniid_alpha = $env:NONIID_ALPHA
                        repeat = $repeat
                        rounds = $Rounds
                        configuration_runtime_seconds = $configurationSeconds
                        controller_state_bytes = [long]($controllerStateBytes | Select-Object -Last 1)
                        access_lease_full_updates = $AccessLeaseFullUpdates
                        rffl_reputation_fade = 0.95
                        rffl_reputation_threshold_scale = 0.3333333333333333
                    } | Export-Csv -Path $configurationCsv -NoTypeInformation -Append -Encoding UTF8
                }
                catch {
                    Write-Host ("FAILED {0}/{1}: {2}" -f $currentNumber, $ProgressTotal, $detail) -ForegroundColor Red
                    Write-Host "Log: $(Join-Path $resultDir "$projectName.log")" -ForegroundColor Yellow
                    throw
                }
                finally {
                    New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
                    docker compose -p $projectName -f $composeFile logs --no-color |
                        Set-Content -Encoding UTF8 (Join-Path $resultDir "$projectName.log")
                    docker compose -p $projectName -f $composeFile down -v
                }
                $localCompleted++
                $completed = $ProgressOffset + $localCompleted
                Show-MatrixProgress $ProgressActivity $detail $completed
                Write-Host ("Completed {0}/{1}: {2}" -f $completed, $ProgressTotal, $detail) -ForegroundColor Green
                }
              }
            }
        }
      }
    }
}
finally {
    if ($previousAttackStartRound) {
        $env:ATTACK_START_ROUND = $previousAttackStartRound
    } else {
        Remove-Item "Env:ATTACK_START_ROUND" -ErrorAction SilentlyContinue
    }
    if ($previousLimitedCleanUpdates) {
        $env:MIN_LIMITED_CLEAN_UPDATES = $previousLimitedCleanUpdates
    } else {
        Remove-Item "Env:MIN_LIMITED_CLEAN_UPDATES" -ErrorAction SilentlyContinue
    }
    if ($previousLimitedObservationUpdates) {
        $env:MIN_LIMITED_OBSERVATION_UPDATES = $previousLimitedObservationUpdates
    } else {
        Remove-Item "Env:MIN_LIMITED_OBSERVATION_UPDATES" -ErrorAction SilentlyContinue
    }
    if ($previousAccessLeaseFullUpdates) {
        $env:ACCESS_LEASE_FULL_UPDATES = $previousAccessLeaseFullUpdates
    } else {
        Remove-Item "Env:ACCESS_LEASE_FULL_UPDATES" -ErrorAction SilentlyContinue
    }
    if ($previousMaxProbationTasks) { $env:MAX_PROBATION_TASKS = $previousMaxProbationTasks }
    else { Remove-Item "Env:MAX_PROBATION_TASKS" -ErrorAction SilentlyContinue }
    if ($previousMinProbationCompletedTasks) { $env:MIN_PROBATION_COMPLETED_TASKS = $previousMinProbationCompletedTasks }
    else { Remove-Item "Env:MIN_PROBATION_COMPLETED_TASKS" -ErrorAction SilentlyContinue }
    if ($previousMinEvidenceMass) { $env:MIN_EVIDENCE_MASS = $previousMinEvidenceMass }
    else { Remove-Item "Env:MIN_EVIDENCE_MASS" -ErrorAction SilentlyContinue }
    if ($previousLimitedAggregationWeight) { $env:LIMITED_AGGREGATION_WEIGHT = $previousLimitedAggregationWeight }
    else { Remove-Item "Env:LIMITED_AGGREGATION_WEIGHT" -ErrorAction SilentlyContinue }
    if ($previousCumulativeRiskDecay) { $env:CUMULATIVE_RISK_DECAY = $previousCumulativeRiskDecay }
    else { Remove-Item "Env:CUMULATIVE_RISK_DECAY" -ErrorAction SilentlyContinue }
    if ($previousCumulativeRiskThreshold) { $env:CUMULATIVE_RISK_THRESHOLD = $previousCumulativeRiskThreshold }
    else { Remove-Item "Env:CUMULATIVE_RISK_THRESHOLD" -ErrorAction SilentlyContinue }
    if ($previousSelfReversalGate) { $env:SELF_REVERSAL_GATE = $previousSelfReversalGate }
    else { Remove-Item "Env:SELF_REVERSAL_GATE" -ErrorAction SilentlyContinue }
    if ($previousNormEscalationMode) { $env:NORM_ESCALATION_MODE = $previousNormEscalationMode }
    else { Remove-Item "Env:NORM_ESCALATION_MODE" -ErrorAction SilentlyContinue }
    if ($previousIncumbentNodeIds) { $env:INCUMBENT_NODE_IDS = $previousIncumbentNodeIds }
    else { Remove-Item "Env:INCUMBENT_NODE_IDS" -ErrorAction SilentlyContinue }
    @(
        "DATA_MODE", "DATASET_NAME", "FLOWER_ROUNDS", "NONIID_ALPHA",
        "MECHANISM_VARIANT", "EXPERIMENT_SEED", "ATTACK_SCENARIO",
        "EXPECTED_EXPERIMENT_NODES", "NUM_DATA_PARTITIONS", "FLOWER_MIN_CLIENTS",
        "TARGET_PARTITION_ID", "TARGET_PARTITION_ID_1", "TARGET_PARTITION_ID_2",
        "ATTACKER_PROFILE", "AGGREGATION_RULE"
    ) | ForEach-Object {
        Remove-Item "Env:$_" -ErrorAction SilentlyContinue
    }
    if ($previousControlToken) {
        $env:FL_CONTROL_TOKEN = $previousControlToken
    } else {
        Remove-Item "Env:FL_CONTROL_TOKEN" -ErrorAction SilentlyContinue
    }
    if ($previousTrustEstimator) {
        $env:TRUST_ESTIMATOR = $previousTrustEstimator
    } else {
        Remove-Item "Env:TRUST_ESTIMATOR" -ErrorAction SilentlyContinue
    }
    if ($previousDatasetName) {
        $env:DATASET_NAME = $previousDatasetName
    }
    if ($previousTaskDeadline) { $env:TASK_DEADLINE_SECONDS = $previousTaskDeadline }
    else { Remove-Item "Env:TASK_DEADLINE_SECONDS" -ErrorAction SilentlyContinue }
    if ($previousBarrierTimeout) { $env:BARRIER_TIMEOUT_SECONDS = $previousBarrierTimeout }
    else { Remove-Item "Env:BARRIER_TIMEOUT_SECONDS" -ErrorAction SilentlyContinue }
    if ($previousClientJoinTimeout) { $env:CLIENT_JOIN_TIMEOUT_SECONDS = $previousClientJoinTimeout }
    else { Remove-Item "Env:CLIENT_JOIN_TIMEOUT_SECONDS" -ErrorAction SilentlyContinue }
}

$summaryCsv = Join-Path $resultDir "real_fl_summary.csv"
$accessSummaryCsv = Join-Path $resultDir "real_fl_access_summary.csv"
docker run --rm `
    --mount "type=bind,source=$resultDir,target=/results" `
    node-access-flower-prototype:0.23 `
    python -m flower_prototype.analyze_real_fl_results `
    /results/real_fl_round_metrics.csv /results/real_fl_node_results.csv `
    /results/real_fl_summary.csv /results/real_fl_access_summary.csv
if ($LASTEXITCODE -ne 0) { throw "Real-FL summary generation failed" }

Write-Host "Completed. Results: $resultDir"
Write-Host "Round metrics: $roundCsv"
Write-Host "Round-node ledger: $roundNodeCsv"
Write-Host "Node decisions: $nodeCsv"
Write-Host "Formal summary: $summaryCsv"
Write-Host "Access summary: $accessSummaryCsv"
if (($ProgressOffset + $localCompleted) -ge $ProgressTotal) {
    Write-Progress -Id 41 -Activity $ProgressActivity -Completed
}

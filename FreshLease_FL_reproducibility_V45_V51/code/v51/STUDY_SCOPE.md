# V51 access evidence and generalization, frozen before the new training matrix

## Research question and scope

This is an extension of the V49/V50 CIFAR-10 study. It does **not** change
Dirichlet trust thresholds, newcomer observation O=5, reduced weight=0.50,
lease length L=5, attack start=round 6, number of training rounds=50,
512 local examples per round, or fixed validation/test subsets.

Three new conditions, each evaluated separately:

| Condition | Dataset | Clients | Benign incumbents | Newcomers | Dirichlet alpha |
| --- | --- | ---: | ---: | ---: | ---: |
| `alpha_low` | CIFAR-10 | 20 | 18 | 2 | 0.1 |
| `alpha_high` | CIFAR-10 | 20 | 18 | 2 | 10 |
| `scale_10` | CIFAR-10 | 10 | 9 | 1 | 0.5 |

Each condition has `access_freshness_lease`, `access_full`, and independently
implemented `rffl_reputation` (RFFL-style), two scenarios (backdoor and benign
concept drift), ten **paired** seeds 0–9, and 50 rounds: 60 cases per condition,
180 cases across the three conditions. The existing V49 condition (20/2,
alpha=0.5) remains separately reported. The `benign_concept_drift` target is
a **legitimate newcomer**; incumbent honest nodes are an independent cohort.
The backdoor target is not expected to be rejected initially when it behaves
honestly during admission. Training-time RFFL removal is not initial rejection.

This is a *preselected* data heterogeneity and scale check, not a search for
the best alpha. V50 is the weight-matched test for the central condition only.
For the new conditions a V50 constant weight of 23/30 is **not** assumed to
match actual target mass, so no V51 result may be attributed specifically to
lease scheduling over an equal-total-weight control.

## Protocol and audit probe, executable locally without Docker

From the project root:

```powershell
python -m flower_prototype.audit_probe_v51 --output .\audit_probe_v51.csv
```

The CSV covers a clean control; modified audit payload; deleted middle and
latest event; forged client acknowledgment; repeated signed acknowledgment;
and a previously correct answer reused under a new nonce. This is seven
deterministic **protocol cases**, not seven independently sampled security
trials. It probes only these implementations and adversary actions. The
unanchored hash chain detects modified and deleted *middle* events, but
**cannot detect deletion of its newest event in an untrusted database**.
The last case is an explicit negative control and rules out a blanket claim
of tamper resistance. A separately preserved chain head or externally
anchored checkpoint would be required to address suffix deletion; it has not
been implemented or evaluated here. The toy attestation work product remains
a nonce-bound protocol challenge, not a TPM quote.

## Windows formal training

Copy the *same frozen* `flower_prototype\data\cifar10.npz` from the V49/V50
project into this package. With Docker Desktop running, from the V51 project
root:

```powershell
.\flower_prototype\run_q3_access_v51.ps1 -Stage smoke
.\flower_prototype\run_q3_access_v51.ps1 -Stage formal -SkipBuild
```

The smoke stage runs 18 configurations × 16 rounds, giving one lease cycle.
The formal stage runs 180 configurations × 50 rounds; it does not reuse the
smoke observations. `-SkipBuild` is safe only if the local image already
contains this V51 analyzer; otherwise omit it (default builds once, then
reuses the image). For a smaller first formal batch or to recover a condition:

```powershell
.\flower_prototype\run_q3_access_v51.ps1 `
    -Stage formal -Conditions alpha_low -SkipBuild
```

The CLI displays the inherited per-configuration progress bar. To resume
the **same** selected conditions after an interruption:

```powershell
.\flower_prototype\run_q3_access_v51.ps1 `
    -Stage formal -Conditions alpha_low `
    -ResultDirectory "D:\absolute\existing\q3_access_v51_results-formal-..." `
    -Resume -SkipBuild
```

Resume verifies the five raw CSV files (including the new per-configuration
`real_fl_audit_integrity.csv`) and configuration keys in each batch.
Incomplete batches restart as separately named retry batches; a Flower round
itself is not checkpointed. Keep each condition's root distinct when running
them separately, and never merge their means into one global effectiveness
number. Docker training is required; Python unit tests cannot replace it.

## Readout and decision rules

The runner emits seven V51 aggregate files plus five raw `real_fl_*.csv`:

- `q3_access_v51_promotion_guard.csv` must confirm all progressive newcomers
  reach full weight before round-6 attack and all 50 rounds exist in formal runs;
- `q3_access_v51_access_runs.csv` and `_access_summary.csv` report newcomer
  initial training eligibility and limited status, task count, onboarding
  time, preattack full access, and incumbents separately. Newcomer eligibility
  in benign concept drift is the observed legitimate newcomer pass rate,
  **not** a measured false-rejection rate across diverse legitimate devices;
- `real_fl_audit_integrity.csv` records an end-of-run audit check per
  configuration *before* the disposable controller container is removed.
  Its chain and signature validity do not defeat suffix deletion or establish
  an independent attacker-resistant log;
- `q3_access_v51_paired_runs.csv` and `_paired_summary.csv` compare exposure,
  ASR-AUC, accuracy, and time under paired seeds. Inspect confidence intervals,
  directions, and baseline attack effectiveness **within each condition**;
- `q3_access_v51_runs.csv` and `_summary.csv` retain unpaired method outcomes.

Do not interpret a near-zero baseline ASR as strong defense evidence. Do not
describe the result as favoring lease timing over equal-mass constant weighting
unless a separate empirical mass-matched control supports it. Report normal
newcomer participation costs and any declining accuracy, even if inconvenient.

## Execution status

Local protocol probe and Python tests can run in the Codex workspace. The
Flower/Docker formal matrix requires the user's Windows/Docker environment.
**No V51 formal effectiveness result is claimed in this package.**

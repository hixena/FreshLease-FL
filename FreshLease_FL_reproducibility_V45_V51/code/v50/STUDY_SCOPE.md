# V50: time-invariant, weight-matched comparison for V49

The V49 formal results already show lower ASR-AUC for the L=5/O=5 lease versus
unlimited full access. V50 checks a narrower question: does periodic permission
expiration help beyond an equal **total unnormalized aggregation weight**?

The completed V49 round-node ledger shows that, for every target and seed in
rounds 6–50, 24 accepted updates had weight 1 and 21 had weight 0.5:
34.5 weight units for 45 updates. The new `access_weight_matched` comparator
is identical to `access_full` before round 6. From round 6 onward *all*
newcomers, including benign newcomers, have constant weight 23/30. Its
45-update nominal weight mass is 34.5. Existing incumbent honest nodes retain
weight 1. The rule never accesses the attack label. There is no lease, renewal,
or permission expiration in the comparator.

Both conditions use the existing V49 CIFAR-10 logistic-regression model,
20 clients with 2 attackers, Dirichlet alpha 0.5, 50 rounds, round-6 attack,
10 paired seeds, backdoor and benign drift settings. This package runs only
20 new configurations, reusing the already completed V49 reference **read only**.
The formal analysis audits every target's actual aggregated weight, number
of aggregated updates, and round-5 promotion before calculating paired
ASR-AUC, final ASR, sustained target share, and final accuracy.

No V50 empirical outcome is yet claimed.

## Windows PowerShell

Unpack in a fresh directory. Copy `flower_prototype\data\cifar10.npz` from
the completed V49 project if the data is not included. From the V50 project
root:

```powershell
.\flower_prototype\run_weight_matched_v50.ps1 -Stage smoke
```

If the smoke output has `weight_matched_v50_promotion_guard.csv` and
`weight_matched_v50_summary.csv`, run the paired formal matrix:

```powershell
$v49Root = "C:\path\to\post_promotion_v49_results-formal-20260924-111731"
.\flower_prototype\run_weight_matched_v50.ps1 -Stage formal -V49ResultDirectory $v49Root
```

To resume a V50 output directory without repeating a **completed** 20-case
batch, run the same command with `-ResultDirectory <existing V50 root> -Resume`.
Partial batches restart in a sibling `retry` directory; the runner does not
checkpoint from the middle of an FL round. Rebuild is the default because the
new method is part of the container code. Only pass `-SkipBuild` when that
same V50 image has already been built on this machine.

The formal root contains `weight_matched_v50_paired_runs.csv` and
`weight_matched_v50_paired_summary.csv`. Inspect
`target_aggregation_mass_difference` and the summary's
`empirical_weight_mass_matched_within_1pct_rate` first. If weight mass differs,
the measured outcomes need this qualification; an unmatched comparison does
not isolate the temporal scheduling mechanism.

# FreshLease-FL: auditable progressive admission experiments

This repository is a reproducibility snapshot for the FreshLease-FL manuscript. It contains the V45 Fashion-MNIST auxiliary experiment, the V48 CIFAR-10 main matrix, the V49 post-promotion experiment, the V50 equal-total-weight control, and the V51 three-condition robustness matrix. The studies are **separate**: do not combine their independent seeds into one confidence interval.

## Research scope and interpretation

The implementation tests authorization state transitions and their effects on aggregation exposure. Newcomers receive limited initial influence, may earn expiring full-weight access, and can renew it after a fresh period of structurally valid updates. Incumbents are predeclared honest in these formal experiments. The mechanism does **not** infer that structurally valid updates are free of semantic backdoors.

The observed attack-period ASR-AUC reductions against non-expiring progressive access are conditional on the tested CIFAR-10 logistic-regression setup, newcomer attackers, approximately 10% attacker participation, 10 or 20 total clients, and fixed parameters L=5, O=5, and limited weight 0.5. V50 matches accepted target-update count and total target weight; it **does not** detect an additional ASR-AUC benefit from periodic leases over equal-mass constant downweighting. RFFL-style is an independent reproduction, not the original authors' released code. Internal hash-chain and signature consistency checks do not establish resistance to database suffix deletion or controller compromise.

## Repository contents

| Path | Contents |
| --- | --- |
| `code/v45/`, `code/v48/`, `code/v49/`, `code/v50/`, `code/v51/` | Source snapshots corresponding to each experiment version, including Docker and PowerShell runners. The frozen parameter documentation is in each version's `STUDY_SCOPE.md`. |
| `results/v45/` | Fashion-MNIST auxiliary matrix: 30 configurations × 50 rounds, including five paired attack seeds against progressive access. |
| `results/v48/` | CIFAR-10 original main matrix: 60 configurations × 50 rounds, 10 paired seeds per attack/comparator. This is the round-2 attack experiment behind the manuscript's main CIFAR-10 tables. |
| `results/v49/` | 60 independent configurations × 50 rounds, 20 clients with 2 newcomer targets, Dirichlet α=0.5, two scenarios and three variants. |
| `results/v50/` | 20 new configurations × 50 rounds for the equal-target-weight control, paired against V49 outputs. |
| `results/v51/` | 180 configurations × 50 rounds: 20 clients / 2 targets at α=0.1 and 10, and 10 clients / 1 target at α=0.5; two scenarios, three variants and 10 paired seeds per condition. |
| `verify_results.py` | Standard-library verifier: counts raw records, confirms version-specific paired means and numerical manuscript claims. |
| `SOURCE_MANIFEST.csv` | SHA-256 and original archive paths for included CSV files. |
| `ORIGINAL_ARCHIVES.csv` | SHA-256 of the historical source and formal result ZIP files used to assemble this package. |
| `SHA256SUMS` | SHA-256 for all files in this release tree, excluding this self-referential checksum list. |

CSV files larger than 2 MiB are stored as `*.csv.gz`. The verifier transparently reads `.csv` or `.csv.gz`. No training logs or third-party image archives are included. All result CSV values are byte-preserved after decompression; smaller CSVs are copied unchanged.

## Verify published observations without Docker

From the repository root, run:

```bash
python verify_results.py
```

This command uses only Python's standard library and verifies the SHA-256 file list, the V45, V48, V49, V50 and V51 result counts, V51 recorded controller integrity flags and promotion guards, the paired means, the equality of V50 target weight mass, and the principal manuscript effect sizes. It verifies the **supplied observations**; it does not rerun local training or demonstrate audit integrity against an attacker.

For a faster numerical check without streaming the full event ledgers:

```bash
python verify_results.py --skip-event-count
```

Re-running an original analysis script requires the raw CSVs: run `python extract_csv.py results/v48` (or another version's results directory) to decompress the few `.csv.gz` files, then run the corresponding `code/vXX/flower_prototype/analyze_*.py` from that version's project root. V50's formal analyzer additionally takes the four V49 reference files; see the frozen `STUDY_SCOPE.md` and analyzer usage message. The verifier is the simplest cross-version check because it reads published summaries and per-seed paired records without mixing source versions.

## Generate a new CIFAR-10 run on Windows with Docker Desktop

The original training was carried out on Windows with Docker Desktop. The source archive does not include `cifar10.npz`. Create it separately within the chosen version, using the project's pinned downloader and its official archive checksum:

```powershell
cd .\code\v51
python -m pip install -r .\flower_prototype\requirements.txt
python .\flower_prototype\prepare_cifar10.py
.\flower_prototype\run_q3_access_v51.ps1 -Stage smoke
.\flower_prototype\run_q3_access_v51.ps1 -Stage formal
```

`run_q3_access_v51.ps1` creates **new** results and does not re-create the records distributed here byte-for-byte: operating system and dependency versions, timing, and container scheduling may differ. The formal run is 180 configurations and 9,000 total configuration-round observations, not a single 9,000-round training run. If PowerShell execution is blocked, set an execution policy for the current process according to your local administrative policy. The commands for V49 and V50 are in their respective `STUDY_SCOPE.md` files and runners.

The CIFAR-10 data source and checksum are declared in `flower_prototype/prepare_cifar10.py`; refer to the [dataset publisher's page](https://cave.cs.toronto.edu/kriz/cifar.html) for the original material. No raw CIFAR-10 or Fashion-MNIST images are redistributed here.

## Versions, reuse and submission

This snapshot deliberately keeps five versioned code directories so that later adjustments to audit and analysis code are not misrepresented as the source used for earlier results. The `STUDY_SCOPE.md` files preserve the original experiment plans except that one V50 example Windows path has been replaced with a generic path. Their pre-run status statements describe the original plan, not the results included here. The root result files in each directory are original formal exports; there is no merged cross-experiment estimate. A source file's SHA-256 does not prove the binary container's identity or establish supply-chain attestation.

Before making the repository public, the authors should choose the code and result-data licenses, add author metadata and a citation file, and check the target journal's anonymous review requirements. This package deliberately includes no `LICENSE` or invented author identities. An archive/DOI can be attached to a fixed public release once those details are decided.

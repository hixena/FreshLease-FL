# V48 CIFAR-10 precision extension

V48 keeps the complete V47 configuration frozen and adds independent seeds
5--9. It reuses a completed V47 directory containing seeds 0--4, runs only the
30 missing configurations, merges both sets, validates uniqueness and seed
coverage, and produces combined 10-seed summaries.

No model, dataset, attack, access-policy, lease, aggregation, or training
hyperparameter is changed.

Example:

```powershell
.\flower_prototype\run_cifar10_v48.ps1 `
  -BaseResultDirectory "D:\path\to\cifar10_v47_results-formal-TIMESTAMP"
```

The default run executes seeds 5--9 for 50 rounds and writes a new
`cifar10_v48_results-formal10-*` directory. Do not point `ResultDirectory` at
the V47 base directory.

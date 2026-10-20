# How to reproduce and what can be reproduced

We distinguish **three separate tasks** rather than calling all of them a "reproduction".

### 1. Verify archived results without the raw CSV (fast; works in CI)

```bash
python scripts/verify_release.py
python -m pytest -q -m "not slow and not requires_data"
```

This validates the locked protocol SHA-256, the one-shot holdout receipt, the checksums of all 11 fixed-result files, and key article metrics. It does **not** retrain or claim independent test-set results. The notebook is a results viewer and has the same safe default.

### 2. Rerun development experiments from authorized source data (slow)

Download the original dataset and check its fingerprint:

```bash
python scripts/download_data.py
python scripts/verify_release.py --require-data
python -m pytest -q -m "not slow"
```

Development reruns are separated and write to git-ignored `artifacts/reproduced`:

```bash
python scripts/reproduce_development.py --mode screening
python scripts/reproduce_development.py --mode stacking
```

These commands deliberately do **not** replace the archived screening ranking/OOF results and do **not** open outer-holdout outcomes. Running both can take tens of minutes depending on hardware.

### 3. Archived frozen final assessment (already executed)

`LOCKED_PROTOCOL.json` and `results/batch/FINAL_HOLDOUT_RECEIPT.json` document the fixed candidates, protocol SHA, and immutable result checksums. `python run_batch.py` with the data present verifies/reports saved results and does **not** re-evaluate the final holdout. A new evaluation on the same inspected data must never be presented as new, independent test evidence or used for model tuning.

## Dependency and operating system considerations

Use Python 3.11 (recommended) and `pip install -r requirements.txt`. The archival environment recorded Python 3.13.5 on Linux and the package versions in `requirements-verified-environment.txt`. Exact numeric reproduction of stochastic tree ensembles can vary with operating system, binary builds and BLAS/OpenMP. The archival values are fixed records; new development reruns should be reported with their actual environment and numeric deltas, not forced to match by editing results.

Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` to limit resource contention if helpful.

## No-manual-cell principle

Keep model training out of notebook cells. CI runs dataset-free tests; local owners with a lawful data copy can opt in to computationally expensive training using dedicated CLI commands. The holdout receipt is intentionally preserved.

# Frequently asked questions

**Do I need to upload the paper?** No. The GitHub README, `docs/METHODS.md`, `docs/RESULTS.md` and the stored results are the software documentation. A paper link may be added when published; the manuscript Word file does not need to be committed.

**One notebook or many?** One deliberately lightweight results notebook (`PM2.5_Research.ipynb`) is enough. Heavy training, splitting, stacking and tests belong in `.py` modules and CLI scripts. This keeps notebooks from becoming hidden-state pipelines.

**Why does Run All finish quickly?** It verifies and visualizes archived, checksum-protected fixed-model outputs; it is not retraining models or silently scoring the already inspected holdout.

**Can GitHub Actions produce the full 4,707-row scores?** Not by default: CI has no raw Kaggle data and must never re-open the final test as part of pull-request checks. It checks the frozen result receipt and runs dataset-free leakage unit tests.

**Can I reproduce the training?** Yes, after downloading the exact dataset, run `python scripts/reproduce_development.py --mode screening` or `--mode stacking`; both write to `artifacts/`, do not overwrite the original frozen tables, and never score the holdout.

**Will results match to the last decimal?** Not guaranteed. Different Python versions, BLAS/OpenMP implementations and tree-learner binaries may change floating-point results. The frozen results and their hashes are reproducible as *archived records*; development experiments are independently rerunnable with numeric tolerance and disclosed environment.

**Should I automatically rerun the final holdout?** No. The outer holdout has already been evaluated and some earlier partitions were inspected too. `run_batch.py` recognizes its original receipt, verifies file hashes and refuses automatic rescoring. Do not delete the receipt, refit or retune on it to improve reported numbers.

**Can I claim zero data leakage?** No. Procedural guards and exact-pattern disjointness help, but the single cross-sectional source still has possible spatial/selection dependencies and cannot establish causal or out-of-country/time external validity.

**Why no NLP, RNN or RL?** This dataset lacks sufficient raw text, sequential station histories and control/reward processes to justify these methods for the published task.

**Where are the main metrics?** `results/batch/final_regression_fixed.csv`, `results/batch/final_classification_fixed.csv`, and `docs/RESULTS.md`.

**Should I include a model pickle?** Not required for scientific reproduction. It is often binary/version/platform dependent. The estimators and frozen hyperparameters are in `pm25aqi/` and `LOCKED_PROTOCOL.json`; use an explicit future inference release if publishing serialized artifacts.

**Can others use the dataset without permission?** Readers should follow Kaggle dataset terms. This repository deliberately does not grant the data license.

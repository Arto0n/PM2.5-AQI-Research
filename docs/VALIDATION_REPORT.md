# GitHub release validation record — 2026-10-20

## What was actually tested for this packaging

| Test | Observed result | Important limit |
|---|---|---|
| Immutable final-result hash and protocol check, without source CSV | PASS; 11 fixed-result checksums verified | No holdout model fitted or evaluated |
| Dataset-free synthetic integration/leakage suite | **55 passed, 5 deselected** | Covers procedural leakage guards and internal logic, not external validity |
| Full non-historical automated suite using the exact original CSV | **59 passed, 1 historical slow test deselected** | This does not execute every expensive experiment |
| CSV hash and development / screening / confirmation / holdout membership | SHA-256 match, sizes **18,756 / 14,112 / 4,644 / 4,707** | All are partitions of the same source dataset |
| Executed GitHub-facing results notebook in fresh Python kernel | **4 of 4 code cells completed without errors** | Displays archived results; no retraining |
| Full development-only screening reproduction, separate temporary workspace | PASS, ~160 s in the packaging environment | Outer holdout was not scored |
| Regression and classification ranking numerical reproduction | **Exactly same primary MAE 17.21731020026574 and macro-F1 0.5218770406322845** | Determinism is environment-dependent |
| Screening ranking, fold metrics, OOF prediction CSVs and country transfer CSV | **BYTE-IDENTICAL** against the archived files in this environment | Not proof of byte identity across OS / library revisions |
| Source-code guard against overwriting archived screening tables | PASS (direct runner refuses frozen checkout) | Only sanctioned `scripts/reproduce_development.py` uses isolated outputs |
| Nested seven-stacker full-data rerun during this GitHub packaging | **Not re-executed** | Archived nested stacking results are included and unit-tested |
| One-shot final-holdout full model refit during GitHub packaging | **Not re-executed, by design** | Hash-protected, already inspected archived results only |
| GitHub-hosted Actions execution | **Not yet run on GitHub** | Workflow syntax locally parsed; users should check first push |

## Research integrity conditions

- The newly packaged source dataset is **not included** in the upload-ready ZIP due unclear redistributable licensing terms.
- A Kaggle download helper exists, but access may depend on site authentication and its terms of service; manual authorized download is supported.
- `LOCKED_PROTOCOL.json` and `FINAL_HOLDOUT_RECEIPT.json` are unchanged from the scientific archive.
- The primary study outcome values have not been revised or optimized using the holdout in this packaging.
- The research article is maintained separately and is not included in this repository. Archived scientific results remain available.
- This report is an engineering reproducibility audit of the packaging environment, not a new scientific validation or external test.

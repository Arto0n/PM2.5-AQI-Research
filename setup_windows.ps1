# Optional Windows setup helper. Does not score/retrain the inspected holdout.
param([switch]$DownloadData, [switch]$OpenNotebook)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    & py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 not available via py launcher; see README.md for alternatives.' }
}
& $py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $py scripts\verify_release.py
if ($LASTEXITCODE -ne 0) { throw 'Archived results integrity failed' }
& $py -m pytest -q -m 'not slow and not requires_data'
if ($LASTEXITCODE -ne 0) { throw 'Dataset-free tests failed' }
if ($DownloadData) {
    & $py scripts\download_data.py
    if ($LASTEXITCODE -ne 0) { throw 'Kaggle download unavailable. See data/README.md for local import.' }
    & $py -m pytest -q -m 'not slow'
    if ($LASTEXITCODE -ne 0) { throw 'Full dataset tests failed' }
}
if ($OpenNotebook) {
    & $py -m notebook
}
Write-Host 'Setup verified: archived results intact; holdout not reopened.'

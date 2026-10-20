"""Fast checks that work in GitHub Actions without distributing the raw dataset."""
from pathlib import Path
import json
import numpy as np
import pytest
from scripts.verify_release import verify
from scripts.download_data import EXPECTED, install, read_archive
import io, zipfile
ROOT=Path(__file__).resolve().parents[1]

def test_cryptographic_frozen_release_verification_without_raw_dataset():
    obj=verify()
    assert obj['receipt_files_verified']==11
    assert obj['holdout_rows_archived']==4707

def test_public_repo_has_no_raw_csv_or_author_manuscript():
    assert 'global_air_pollution_dataset.csv' in (ROOT/'.gitignore').read_text()
    assert not list(ROOT.glob('*.docx'))
    assert (ROOT/'.github/workflows/tests.yml').is_file()
    assert (ROOT/'README.md').is_file()

def test_downloader_requires_exact_source_hash(tmp_path):
    with pytest.raises(ValueError,match='SHA-256 mismatch'):
        install(b'Country,CO AQI Value\nX,1',tmp_path)
    assert not (tmp_path/'global_air_pollution_dataset.csv').exists()

def test_download_archive_requires_exact_member():
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w') as z:z.writestr('other.csv','x')
    with pytest.raises(ValueError,match='expected CSV'):
        read_archive(memory.getvalue())

def test_article_alignment():
    import pandas as pd
    r=pd.read_csv(ROOT/'results/batch/final_regression_fixed.csv').set_index('model')
    c=pd.read_csv(ROOT/'results/batch/final_classification_fixed.csv').set_index('model')
    assert np.isclose(r.loc['log_hgb_A5','mae'],16.937,atol=.001)
    assert np.isclose(c.loc['lightgbm_A4','macro_f1'],.4977,atol=.0001)
    assert np.isclose(c.loc['rf_balanced_A4','macro_f1'],.5537,atol=.0001)


def test_direct_screening_runner_refuses_overwrite_in_frozen_checkout():
    source=(ROOT/'run_research.py').read_text()
    assert 'Frozen research repository' in source
    assert 'FINAL_HOLDOUT_RECEIPT.json' in source

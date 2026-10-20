"""Fetch or install an authorized copy of the exact Kaggle CSV; reject hash mismatch."""
from __future__ import annotations
import argparse, hashlib, io, shutil, sys, urllib.request, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILENAME='global_air_pollution_dataset.csv'
EXPECTED='e1fc6e682edb8a2102d87e182b844fc8cde49e10763c0ef4293e510bbe240f69'
API_URL='https://www.kaggle.com/api/v1/datasets/download/hasibalmuzdadid/global-air-pollution-dataset'
SOURCE_URL='https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset'

def sha(blob):return hashlib.sha256(blob).hexdigest()

def read_archive(blob):
    if zipfile.is_zipfile(io.BytesIO(blob)):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            hits=[n for n in z.namelist() if n.rsplit('/',1)[-1].lower()==FILENAME.lower()]
            if len(hits)!=1:raise ValueError('Kaggle archive did not contain one expected CSV')
            return z.read(hits[0])
    if blob.startswith(b'Country,') or blob.startswith(b'\xef\xbb\xbfCountry,'):
        return blob
    raise ValueError('Kaggle download was not the expected CSV or ZIP')

def install(blob, root=ROOT):
    root=Path(root);destination=root/FILENAME
    if sha(blob)!=EXPECTED:
        raise ValueError('Dataset SHA-256 mismatch; refusing to install a different version')
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()==EXPECTED:
        return destination
    tmp=destination.with_suffix('.csv.partial')
    try:
        tmp.write_bytes(blob);tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)
    return destination

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-local',type=Path,help='Path to CSV previously downloaded lawfully from Kaggle')
    args=parser.parse_args(argv)
    try:
        if args.from_local:
            blob=args.from_local.read_bytes()
        else:
            print('Retrieving original Kaggle dataset; comply with its terms of use...',flush=True)
            request=urllib.request.Request(API_URL,headers={'User-Agent':'PM25-Research-Dataset-Fingerprint-Verification/1.0'})
            with urllib.request.urlopen(request,timeout=90) as response: blob=read_archive(response.read())
        path=install(blob)
        print(f'PASS: {path} matches original SHA-256 {EXPECTED}')
    except Exception as exc:
        parser.exit(1, f'Unable to install verified Kaggle CSV: {exc}\n'
            f'Download it manually from {SOURCE_URL}\n'
            'Then run: python scripts/download_data.py --from-local PATH_TO_CSV\n')

if __name__=='__main__':main()

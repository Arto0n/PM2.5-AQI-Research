# Dataset is not redistributed here

The original **Global Air Pollution Dataset** belongs to its publisher. The Kaggle page currently presents the license as **“Other (specified in description)”**; permission to redistribute the CSV has not been independently established. This repository intentionally excludes the raw data.

Source: https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset

Expected filename at repository root after an authorized download: `global_air_pollution_dataset.csv`

Expected SHA-256: `e1fc6e682edb8a2102d87e182b844fc8cde49e10763c0ef4293e510bbe240f69`

To fetch for local research (subject to the dataset's terms):

```bash
python scripts/download_data.py
```

Or download directly from Kaggle and register a verified local copy:

```bash
python scripts/download_data.py --from-local "/path/to/global_air_pollution_dataset.csv"
```

The downloader fails closed if the file hash is not an exact match. Dataset is intentionally ignored by Git.

# Uploading this repository to GitHub

The downloadable ZIP contains the **repository contents**, not a single ZIP to commit.

## Recommended: private repository first

1. Extract `PM25_AQI_GitHub_Ready_Repository.zip` onto your computer.
2. Sign in to GitHub and create a **new empty private repository**, for example `PM25-AQI-Research`. Do not initialize it with another README or license (this ZIP has its own README).
3. In a terminal inside the extracted `PM25-AQI-Research` folder, run:

```bash
git init
git branch -M main
git add .
git status
# Inspect changes: no *.csv original data, no .venv or article .docx should be staged.
git commit -m "Publish audited PM2.5 AQI research source and results"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPO.git
git push -u origin main
```

Replace only the `YOUR_ACCOUNT/YOUR_REPO` URL with your actual GitHub repository location; no login details are included in this package. GitHub will prompt for authentication using its supported authentication workflow.

## Browser-only method

Extract the ZIP **first**. In the empty GitHub repository, choose **Add file → Upload files**, then drag/upload the extracted *contents*, including dotfiles and the `.github` folder. GitHub's browser upload handles directories differently across browsers and has file-count limits; local Git is more reliable for a multi-folder project. **Do not upload the ZIP itself as the only repository file.**

## After upload

1. Open `.github/workflows/tests.yml` under the Actions tab to observe CI checks.
2. The first public README page should show the frozen figures and results automatically.
3. Check the dataset CSV is not committed (it is in `.gitignore`).
4. Check the archive receipt and protocol stay committed unchanged.
5. Discuss and choose a code license with all rights holders before making the repository publicly reusable.
6. Confirm `CITATION.cff` author spellings and study title; add a publication DOI only when one exists.

## Security and research boundary

- Do not commit `kaggle.json`, tokens, `.venv`, `.env`, raw dataset or local model binary dumps.
- GitHub Actions must **not** score the already-inspected holdout or run model selection using final scores.
- Public viewers can verify frozen metrics even without an independent legal right to receive the dataset.

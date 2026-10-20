# Public-release decision checklist

The repository is technically uploadable as delivered, but public release is **not** the same as a cleared open-source release.

- [ ] Confirm with the coauthors / supervisor that the research code and summarized study results may be published now.
- [ ] Check target journal preprint and software policies, if applicable.
- [ ] Decide an open-source software license with the copyright holders. **No code LICENSE is assigned automatically.** Without a license others generally do not have an express right to reuse the source.
- [ ] Confirm the third-party Kaggle data licensing conditions. Keep the source CSV outside the public repository unless written permission is established.
- [ ] Verify manuscript author and citation metadata in `CITATION.cff` before publishing.
- [ ] Run `python scripts/verify_release.py` and `python -m pytest -q -m "not slow and not requires_data"`.
- [ ] If the original dataset is available, run `python -m pytest -q -m "not slow"` and compare results.
- [ ] Check for any accidental secrets, user paths, internal drafts, `.venv`, generated large binary models or unlicensed content.
- [ ] When public, add a stable release tag (e.g., `v1.0.0`) and an archive DOI if desired (Zenodo integration is optional).
- [ ] Update `README.md` with the final paper DOI/URL once accepted; do not invent one.

**Recommendation:** create a private GitHub repository now; make it public when permissions, citation metadata and license decisions are confirmed.

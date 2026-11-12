# UCR TwoPatterns raw data

The original UCR TRAIN and TEST .ts files are checked into `TwoPatterns/`.
`TwoPatterns/SOURCE.json` records download URL, sample counts and SHA-256 checksums.

These are real UCR benchmark files, not synthetic placeholder data created by this project. The benchmark itself is a published simulated time-series dataset.

To refresh from the source archive:

```bash
python scripts/fetch_twopatterns.py
```

Do not mix the official TEST split into training.

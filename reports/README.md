# Reference runs

Committed output from real runs, so the repository's claims can be checked without anyone
re-running anything. A fresh local run writes to `audit_outputs/` instead, which is git-ignored.

## `2026-11-04-kaggle-run/`

The run the README quotes. Produced on Kaggle, on `creditcard.csv` with SHA-256 `76274b69…`,
`cmvt` 0.1.0, seed 42.

| File | Contents |
|---|---|
| `model_risk_audit_report.md` | the generated report — verdicts, findings, limitations |
| `run_config.json` | the full parameter set, data fingerprint and drift summary |
| `figures/` | the four figures |
| `tables/` | all 13 evidence tables as CSV |

Every number in the report is interpolated from the tables beside it, so the two cannot disagree.

### Reproducing it

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
make audit
```

Expect the same numbers given the same dataset, seed and library versions — the report's
reproducibility section records the versions used. Differences to look for first: a revised
dataset (the SHA-256 in the report will not match) or a different scikit-learn version
(`HistGradientBoostingClassifier` early-stopping behaviour has changed across releases).

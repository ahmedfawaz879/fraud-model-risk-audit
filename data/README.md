# Data

The dataset is **not** redistributed here (see `.gitignore`). Download it yourself:

**Kaggle — Credit Card Fraud Detection** (`mlg-ulb/creditcardfraud`)
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

Place `creditcard.csv` in this directory:

```
data/creditcard.csv
```

Or with the Kaggle CLI:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
```

## What it contains

284,807 card transactions over two days in September 2013, 492 of them fraudulent (0.172%).
`V1..V28` are anonymised PCA components, `Amount` is the transaction value, `Time` is seconds
since the first transaction, `Class` is 1 for fraud.

## Fingerprint

The audit hashes whatever file it reads and prints the digest in the report, so a result can
always be traced to exact bytes. The committed reference run in
`reports/2026-11-04-kaggle-run/` used:

```
sha256 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
```

If your digest differs, the dataset has been revised and your numbers will not match the
reference run — which is the point of printing it.

## Licence

The dataset is released by the Machine Learning Group at Université Libre de Bruxelles under
[Open Data Commons ODbL v1.0](https://opendatacommons.org/licenses/odbl/1-0/). Cite Dal Pozzolo
et al. as requested on the Kaggle page.

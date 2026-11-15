# Dataset provenance and handling

## Original source

- Dataset: **Heart Failure Prediction Dataset**, compiled by **fedesoriano**.
- [Source page](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- [Public download](https://www.kaggle.com/api/v1/datasets/download/fedesoriano/heart-failure-prediction)
- Original source collections: Cleveland, Hungarian, Switzerland, Long Beach VA, and Statlog.
- Observed Kaggle license label: **Database: Open Database, Contents: Copyright Original Authors**. Retain attribution and consult the linked source terms when redistributing derived data.
- The original heart disease collections are documented by [UCI](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease), with creators Andras Janosi, William Steinbrunn, Matthias Pfisterer, and Robert Detrano.

`raw/heart.csv` is an unchanged copy of the downloaded CSV. `raw/provenance.json` records the source URLs, retrieval timestamp, observed metadata, and SHA-256 hashes. The download script refuses to replace an existing source silently.

## Data quality

The CSV has 918 records, 11 inputs, and one target: `HeartDisease` (1=disease, 0=normal). There are 508 positive and 410 negative labels, no empty cells, and no exact duplicate input vectors. However, cholesterol is zero in 172 records and resting BP is zero in one record.

The model pipeline treats nonpositive cholesterol/BP as missing and learns numerical medians inside each training fold. It retains all records and adds missingness indicators. It does not replace valid zero-valued binary features or zero/negative oldpeak. Missingness may be associated with collection practices, which is a study limitation.

## Processed metadata

- `processed/split_assignments.csv`: zero-based source row, train/test membership, and CV validation fold. Test records use fold `-1`; training records use `0` through `4`.
- `processed/feature_dictionary.csv`: readable feature descriptions.

An imputed full-dataset CSV is deliberately not exported. The fitted preprocessing pipeline applies training-only statistics consistently to new records. This avoids encouraging preprocessing the full dataset before splitting it.

The combined benchmark overlaps the source collections used in the reference paper. It is unsuitable for claims of validation on a new, independent patient population.

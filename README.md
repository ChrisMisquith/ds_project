# Heart Disease Classification and Feature Reduction

A BTech data science project based on the CART study by Ozcan and Peker (2023).
The project implements a transparent classification baseline and asks how much predictive
performance is retained when fewer original clinical variables are used.

## Start here

1. Read [data cleaning and baseline modeling](notebooks/01_data_cleaning_and_baseline.ipynb).
   It includes the data audit, all feature definitions, cleaning, preprocessing,
   model training, cross-validation, test results, and CART interpretation.
2. Read [feature reduction](notebooks/02_feature_reduction.ipynb).
   It includes the forward-selection implementation, comparisons at 3, 5, 7, and 11
   inputs, selection stability, and a compact demonstration model.
3. Open the [baseline PDF report](output/pdf/heart_disease_project_report.pdf) and
   [feature-reduction PDF report](output/pdf/heart_disease_feature_selection.pdf).
4. See [methodology and comparison with the paper](docs/methodology.md) for the
   dataset differences and limitations.

Both notebooks are executed and include their outputs. GitHub can display them without
installing Python. CART is the paper-inspired baseline; logistic regression (LR) and
random forest (RF) are comparison models. XGBoost is not used.

## Dataset and cleaning

The public [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
contains 918 records, 11 inputs, and the binary target `HeartDisease`.
Despite the dataset title, the task is classification of recorded heart disease presence,
rather than prediction of future heart failure.

The inputs are `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`,
`RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, and `ST_Slope`.
`RestingECG` is a summary category (Normal, ST, or LVH), not an ECG waveform.

Nonpositive cholesterol and resting blood pressure are marked missing. Numerical
medians and missingness indicators, categorical imputation and one-hot encoding, and
LR scaling are fitted inside training folds. All records are retained. Valid binary
zeros and zero or negative `Oldpeak` values are preserved.

The unchanged CSV, source attribution, observed license label, and SHA-256 are recorded
in [data/](data/README.md). Third-party data remains subject to its source terms.

## Main findings

On the fixed 184-record baseline test set, tuned CART achieved **79.35% accuracy**,
LR **90.22%**, and RF **90.76%**. LR was selected using development cross-validation
before the test results were inspected. The RF test score does not establish superiority
over LR or over the paper.

For feature reduction, the original test set is excluded. Nested evaluation on the
734 development records gave the three-input LR selection procedure **84.06% mean
accuracy**, compared with **86.51%** for all 11 inputs. Smaller models have a measured
performance trade-off. ST slope and cholesterol appeared in all five three-input
outer-training subsets; the third variable varied.

The final demonstration model uses `ST_Slope`, `Cholesterol`, and `RestingECG`.
Its exact final subset has no fresh independent performance test. The nested scores
evaluate a selection procedure whose selected variables can differ by fold.
Do not compare the feature-reduction mean directly with the baseline holdout accuracy.

## Repository layout

```text
data/          Unchanged dataset, provenance, feature dictionary, split membership
docs/          Methodology, paper comparison, and interpretation
notebooks/     Two complete, executed notebooks
src/           Shared cleaning, modeling, plotting, and selection code
scripts/       Rebuild notebooks, generate reports, and validate results
tests/         Behavioral tests for cleaning, metrics, and feature selection
examples/      Example inputs for the compact demonstration model
output/        PDF reports, figures, metrics, predictions, and verification summaries
```

Generated model binaries, environments, caches, and temporary files are excluded from
Git. Running the project recreates the fitted pipelines locally.

## Reproduce the project

Use Python **3.12**. From the repository root on Windows:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python scripts/run_all.py
```

On macOS or Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python scripts/run_all.py
```

The existing raw CSV is verified and reused. No Kaggle account or API key is required.
The complete run rebuilds and executes both notebooks, recreates the fitted models,
generates both PDFs, and runs the checks. Runtime depends on the computer; searches
use one CPU worker. `requirements-lock.txt` records the verified environment;
`requirements.txt` gives broader compatible dependency ranges.

Optional commands, using the same environment:

```bash
python scripts/run_all.py --study baseline
python scripts/run_all.py --study features
python -m unittest discover -s tests -v
python scripts/predict_compact.py --input examples/compact_inputs.csv
```

Run the baseline at least once before running features alone: it writes the fixed
split assignments. The example prediction command requires the generated compact model.
Example rows come from development data and are demonstrations, not test evidence.

## Checks and limitations

Checks reconstruct metrics from saved predictions, verify disjoint training and
validation membership, confirm training-only medians, reload the fitted pipelines,
check forward-selection decisions, and check notebook execution and PDF integrity.
Verification summaries are saved under `output/`.

This is a small retrospective benchmark with overlapping source collections relative
to the paper. Cleaning, preprocessing, tuning, and evaluation differ from the original
study, so a higher number is not proof of a better method. Selection is model-dependent,
greedy search is not exhaustive, and missingness may encode collection practices.
No clinical deployment, external validation, or new feature-selection algorithm is claimed.

## References

- Ozcan, M., & Peker, S. (2023). *A classification and regression tree algorithm for
  heart disease modeling and prediction*. Healthcare Analytics, 3, 100130.
  [Paper DOI](https://doi.org/10.1016/j.health.2022.100130).
- fedesoriano. [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
- UCI Machine Learning Repository. [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease).
- scikit-learn. [Pipelines and common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
  and [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html).

AI assisted code and documentation preparation. Numerical findings are generated by
execution; the student team should understand the methods before presenting the project.

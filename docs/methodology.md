# Methodology and comparison with the original paper

## What the project implements

The project has two studies: a paper-inspired CART baseline with classifier comparisons,
and an extension that selects smaller groups of original clinical variables. The shared
cleaning and evaluation code is in `src/`. The notebooks show the complete experiment
and its generated results in reading order.

## Original study and our dataset

Ozcan and Peker (2023), [DOI 10.1016/j.health.2022.100130](https://doi.org/10.1016/j.health.2022.100130),
used a combined heart-disease dataset. They describe 1,190 original records, removal of
272 duplicates, and removal of records with missing or invalid measurements to obtain
a final 745-record sample. They report 596 training and 149 test records, using an
80:20 split and CART in R (`rpart`), with 87.25% reported accuracy.

Our public CSV contains 918 records from the same family of source collections and
the same 11 kinds of input variables. It already has no exact duplicate input vectors.
We retain all 918 records by marking invalid cholesterol and resting blood pressure
as missing and imputing inside training folds. The fixed stratified split contains
734 development and 184 test records, with seed 42. This is a Python adaptation,
not an exact reproduction of the paper's cleaned sample or fitted tree.

The input variables overlap, but the patient counts, cleaning, categorical encoding,
split, tuning procedure, and programming implementation differ. Accordingly, the
reported accuracies are not a controlled head-to-head comparison. Shared source
collections also prevent describing our dataset as independent external validation.

## Cleaning and representation

- `RestingBP <= 0` and `Cholesterol <= 0` become missing measurements. There is one
  zero BP record and 172 zero cholesterol records in the preserved CSV.
- Numerical features use training-fitted median imputation with missingness indicators.
- Categories use training-fitted most-frequent imputation and one-hot encoding.
- LR additionally standardizes numerical values using training-fitted statistics.
- No rows are removed. Binary zeros and zero or negative `Oldpeak` remain unchanged.
- `RestingECG` records Normal, ST, or LVH. It is not a sampled ECG signal; this project
  does not process raw waveforms or perform signal filtering.

The baseline notebook shows the before/after missing-value counts, the pipeline
implementation, and a comparison between fitted medians and cleaned training medians.
The raw file is never overwritten, and its recorded SHA-256 is checked on every run.

## Baseline evaluation

Five predefined candidates use the same development cross-validation folds: a majority
classifier, LR, unrestricted CART, tuned CART, and RF. Parameter searches optimize mean
five-fold F1 on development records. CART remains the main interpretation model.
The overall selected candidate is fixed before viewing test metrics. Test predictions
use default decision thresholds; test scores are not used to retune the models.

The positive class is `HeartDisease=1`. Confusion matrices have actual classes as rows
and predictions as columns in order `[0, 1]`. Disease recall is TP/(TP+FN); specificity
is TN/(TN+FP). The implementation uses these definitions consistently, because the
paper's reported confusion-matrix discussion and metric labels are not fully consistent.
Intervals for accuracy and recall are Wilson intervals conditional on this test set.
They do not account for dataset shift or all model-selection uncertainty.

## Feature-reduction evaluation

Only the original 734 development records enter feature selection; the original test
records are excluded. Five stratified outer folds evaluate the entire procedure. In
each outer training partition, three-fold inner CV scores greedy forward selection
using a fixed LR model (`C=1`) and F1. The algorithm adds the best remaining original
variable at each step, keeping all encoded columns and missingness indicators belonging
to that variable together. The search is greedy, rather than exhaustive.

Budgets of 3, 5, 7, and all 11 inputs are evaluated on the same outer validation patients.
A fixed RF comparison uses the LR-selected subsets; it is not independently optimized
for RF. Every fit learns its own preprocessing from its own training partition.

The adaptive compact rule chooses the smallest budget whose inner F1 is within 0.01
absolute F1 of the all-11 inner score. If no reduced budget qualifies, all 11 are retained.
This is a design tolerance, not a statistical equivalence or non-inferiority test.
Outer scores do not decide the subset or budget. Selection frequency and mean pairwise
Jaccard similarity describe consistency across the five overlapping training partitions.

After evaluation, the same rule is applied to all development records to fit a compact
demonstration pipeline. Its exact selected subset has no new independent test score.
Outer fold scores describe the selection workflow, whose selected variables may vary.
Fold standard deviations are descriptive variation, not independent confidence intervals.

## Contribution, limitations, and sensible next steps

The extension goes beyond an importance ranking: it retrains reduced models, measures
the loss from removing variables, compares disease recall and specificity, and checks
how stable the selections are. It applies established methods to a semester project;
it does not introduce a new algorithm or establish clinical usefulness.

The three-input LR workflow lost mean accuracy relative to all 11 inputs. ST slope and
cholesterol were consistently selected, but the third variable changed across folds.
Cholesterol's missingness flag may partly explain its usefulness. Earlier exploratory
project results informed the extension, so its outer scores remain development evidence.
The original holdout is not a newly untouched test for a future redesign.

Useful further work would repeat nested evaluation with other seeds, compare alternative
selection strategies using a predeclared protocol, test whether missingness drives
selection, and measure input acquisition costs. Fewer variables do not automatically
mean cheaper data collection: ECG and exercise-test summaries may still be required.
An independent compatible cohort would provide stronger evidence of transfer, provided
it does not reuse the original source collections.

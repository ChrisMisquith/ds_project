"""Create the full, readable feature-reduction notebook from versioned source."""

import ast
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(
        nbf.v4.new_markdown_cell(dedent(text).strip(), id=f"feature-{len(cells):02d}")
    )


def code(text):
    cells.append(
        nbf.v4.new_code_cell(dedent(text).strip(), id=f"feature-{len(cells):02d}")
    )


source = (ROOT / "src/feature_selection.py").read_text(encoding="utf-8")
functions = {
    node.name: ast.get_source_segment(source, node)
    for node in ast.parse(source).body
    if isinstance(node, ast.FunctionDef)
}

md("""
# Feature Reduction for Heart Disease Classification

**BTech data science project | Reproducible implementation**

Can 3, 5, or 7 original clinical inputs preserve useful performance relative to all 11?
This notebook shows the selection algorithm, trains reduced models, evaluates the
performance trade-off, and checks whether the same variables are selected consistently.
The companion notebook contains data cleaning and the paper-inspired CART baseline.

**Contents:** setup; evaluation protocol; complete selection implementation;
outer evaluation; performance comparisons; selection stability; compact demonstration;
findings, limitations, and references.
""")
md("""
## 1. Setup and input data

Run from top to bottom after the baseline notebook, or use `scripts/run_all.py`.
The verified raw CSV and the baseline split assignments are reused. All original test
records remain outside feature selection.
""")
code("""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import joblib
from IPython.display import display, Markdown, Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'src/feature_selection.py').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.project import (FEATURES, TARGET, make_model_pipeline, load_dataset,
                         evaluate_pipeline, write_json, InvalidMeasurementCleaner)
from src.feature_selection import create_figures

BUDGETS = [3, 5, 7, 11]
TOLERANCE = 0.01
OUT = ROOT / 'output/feature_selection'
data, provenance = load_dataset(ROOT)
split = pd.read_csv(ROOT / 'data/processed/split_assignments.csv')
development_ids = split.loc[split['split'].eq('train'), 'source_row']
holdout_ids = split.loc[split['split'].eq('test'), 'source_row']
assert set(development_ids).isdisjoint(holdout_ids)
print('Development records:', len(development_ids), '| Excluded test records:', len(holdout_ids))
display(pd.read_csv(ROOT / 'data/processed/feature_dictionary.csv'))
""")
md("""
### Cleaning is unchanged from the baseline

Nonpositive cholesterol and resting blood pressure become missing. Numerical medians
and missingness indicators, categorical imputation and encoding, and LR scaling are
fitted inside every candidate's training fold. All encoded columns belonging to one
original variable stay together. Zero or negative Oldpeak and valid binary zeros remain.
""")
code("""
development = data.loc[development_ids, FEATURES]
preview = InvalidMeasurementCleaner().fit_transform(development)
display(pd.DataFrame({'Raw empty cells': development.isna().sum(),
                      'After deterministic cleaning': preview.isna().sum()}))
assert len(preview) == len(development)
""")
md("""
## 2. Evaluation protocol

1. Use only the original 734 development records; exclude the original 184 test records.
2. Use five stratified outer folds to evaluate the full selection procedure.
3. Inside each outer training partition, use three-fold CV and LR F1 to select variables.
4. Compare 3, 5, 7, and all 11 inputs on the same outer validation patients.
5. Use fixed LR settings (`C=1`) and a fixed RF comparison. RF uses the LR-selected subsets.
6. Choose the smallest budget within 0.01 absolute inner F1 of the all-11 inner F1.
   If none qualifies, retain all 11. Outer results do not choose the budget.

The tolerance is a design choice, not a significance or equivalence test. Inner search
scores can be optimistic; outer results evaluate the whole selection procedure.
Because earlier project exploration informed the design, these are development estimates.
""")
md("""
## 3. Complete selection implementation

The following functions are generated from `src/feature_selection.py` so the notebook
and reusable implementation stay consistent. The shared pipeline is defined in full
in the baseline notebook and `src/project.py`. Charts use the shared plotting helper.

### Fixed models and the training-only budget rule
""")
code(functions["model_for"])
code(functions["choose_budget"])
md("""
### Greedy forward selection

Start with no variables. At each step, score each remaining original variable when
added to the current subset and keep the best candidate. Fixed feature order breaks ties.
Record every candidate score. Greedy search is not exhaustive and may miss interactions.
""")
code(functions["forward_path"])
md("""
### Outer evaluation and final demonstration fit

For each outer fold, selection and all preprocessing use only its training partition.
Each patient is evaluated once per model family and feature budget. After the outer
evaluation, the same selection rule fits a demonstration model on all development data.
That last fit does not replace the outer results or create a new independent test score.
""")
code(functions["run_study"])
code("""
result = run_study(ROOT)
create_figures(ROOT)
""")
md("""
## 4. Full versus reduced models

Scores are means across five outer validation folds. Within a budget, selected variables
may differ by fold. These are scores for a selection procedure, not one fixed subset.
The adaptive row uses each fold's training-selected budget. Fold SD describes variation;
the overlapping training folds do not provide independent confidence intervals.
""")
code("""
summary = pd.read_csv(OUT / 'tables/summary.csv')
adaptive = pd.read_csv(OUT / 'tables/adaptive_compact_summary.csv')
display(summary.round(4))
display(adaptive.round(4))
display(Image(filename=str(OUT / 'figures/01_performance.png'), width=1000))
lr = summary.loc[summary.family.eq('LR')].set_index('budget')
display(Markdown(
    f"Three-input LR accuracy: **{lr.loc[3, 'accuracy_mean']:.2%}**; "
    f"all-11 LR accuracy: **{lr.loc[11, 'accuracy_mean']:.2%}**. "
    "Removing inputs has a measured trade-off; report recall and specificity too."))
""")
md("""
## 5. Which variables are selected consistently?

Frequency counts how many of five outer training partitions select a variable.
Mean pairwise Jaccard is intersection divided by union across selected subsets; 1 means
identical subsets. It is a descriptive stability measure, not causal importance or proof
that an omitted variable is medically unnecessary.
""")
code("""
subsets = pd.read_csv(OUT / 'tables/selected_subsets.csv')
display(subsets)
display(pd.read_csv(OUT / 'tables/stability.csv'))
display(Image(filename=str(OUT / 'figures/02_stability.png'), width=850))
print('Compact choices made inside outer training partitions:')
display(subsets.loc[subsets.chosen_compact, ['fold', 'budget', 'features', 'inner_f1']])
""")
md("""
## 6. Final compact model for demonstration

The final subset comes from applying the same procedure to all development records.
It has no fresh independent performance test. Its inner score is not a final accuracy
estimate. The outer scores above assess the workflow rather than this exact final subset.
""")
code("""
bundle = joblib.load(OUT / 'models/compact_model.joblib')
display(Markdown(f"Final demonstration subset: **{bundle['budget']} inputs**: "
                 + ', '.join(bundle['features'])))
examples = data.loc[bundle['train_ids'][:3], bundle['features']]
display(examples)
print('Illustrative development-row predictions:', bundle['model'].predict(examples))
print('These examples are not test evidence.')
""")
md("""
## 7. Findings, limitations, and improvements

- The three-input LR procedure loses mean accuracy relative to all 11 inputs. Smaller
  models are not automatically better; disease recall and specificity must also be checked.
- ST slope and cholesterol appear in all five three-input outer-training subsets. The
  third input varies. RestingECG appears in the final development subset, so the exact final
  three inputs are not a universally confirmed ranking.
- Cholesterol selection also retains its missingness indicator. Collection practices
  may partly explain its usefulness. Test this with a predeclared missingness ablation.
- Fewer variables do not necessarily mean cheaper collection: ECG and exercise-test
  summaries can still require diagnostic measurements. No acquisition costs were measured.
- Greedy selection is an established method and can miss interactions. A future study
  could compare other selectors and repeat nested evaluation with additional seeds.
- Source collections overlap the paper's dataset. Independent compatible data would be
  needed for external validation. These models are academic demonstrations.

The contribution is a measured feature-reduction study with stability analysis, beyond
changing the classifier or reading a single feature-importance ranking.

Candidate scores, subset choices, fold membership, predictions, complete metrics, and
figures are saved in `output/feature_selection/`. Fitted models are regenerated locally.
The original holdout remains excluded from this extension.
""")
md("""
## References

- [Original CART paper](https://doi.org/10.1016/j.health.2022.100130).
- [Dataset source](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
- [scikit-learn feature selection](https://scikit-learn.org/stable/modules/feature_selection.html).
- [Nested evaluation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

AI assisted preparation; numerical results are generated by execution.
""")

nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3 (Heart Disease Project)",
            "language": "python",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "title": "Feature Reduction - BTech Heart Disease Project",
    },
)
nbf.validate(nb)
target = ROOT / "notebooks/02_feature_reduction.ipynb"
nbf.write(nb, target)
print(f"Created {target.name}: {len(cells)} cells.")

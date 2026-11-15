from pathlib import Path
import sys, json, hashlib, unittest
import numpy as np
import pandas as pd
import nbformat, joblib
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.project import FEATURES, TARGET, load_dataset, calculate_metrics, write_json
from src.feature_selection import choose_budget


def main():
    out = ROOT / "output/feature_selection"
    checks = []

    def check(name, ok):
        if not ok:
            raise AssertionError(name)
        checks.append(name)

    tests = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    )
    check("All behavior tests pass", tests.wasSuccessful())
    raw, _ = load_dataset(ROOT)
    original = pd.read_csv(ROOT / "data/processed/split_assignments.csv")
    dev = set(original.loc[original.split.eq("train"), "source_row"])
    roles = pd.read_csv(out / "tables/fold_membership.csv")
    selection = pd.read_csv(out / "tables/selected_subsets.csv")
    pred = pd.read_csv(out / "tables/outer_predictions.csv")
    metrics = pd.read_csv(out / "tables/fold_metrics.csv")
    candidates = pd.read_csv(out / "tables/candidate_search.csv")
    check(
        "Only development patients used",
        set(roles.source_row) == dev and len(dev) == 734,
    )
    for fold in range(5):
        d = roles.loc[roles.fold.eq(fold)]
        train = d.loc[d.role.eq("train"), "source_row"].to_numpy()
        valid = d.loc[d.role.eq("validation"), "source_row"].to_numpy()
        check(
            f"Fold {fold}: train and validation disjoint",
            set(train).isdisjoint(valid) and set(train) | set(valid) == dev,
        )
        sel = selection.loc[selection.fold.eq(fold)]
        scores = dict(zip(sel.budget, sel.inner_f1))
        check(
            f"Fold {fold}: compact rule uses inner scores",
            sel.loc[sel.chosen_compact, "budget"].tolist() == [choose_budget(scores)],
        )
        log = candidates.loc[candidates.fold.eq(fold)]
        path = []
        for step in range(1, 8):
            rows = log.loc[log.step.eq(step)]
            check(
                f"Fold {fold} step {step}: complete forward candidate set",
                set(rows.added_feature) == set(FEATURES) - set(path),
            )
            winner = rows.loc[rows.inner_f1.idxmax()]
            path.append(winner.added_feature)
            check(
                f"Fold {fold} step {step}: candidate paths keep clinical groups",
                all(
                    json.loads(s) == path[:-1] + [f]
                    for s, f in zip(rows.subset, rows.added_feature)
                ),
            )
            if step in [3, 5, 7]:
                check(
                    f"Fold {fold}: budget {step} follows winning path",
                    json.loads(sel.loc[sel.budget.eq(step), "features"].iloc[0])
                    == path,
                )
        for _, row in sel.iterrows():
            features = json.loads(row.features)
            k = int(row.budget)
            check(
                f"Fold {fold} budget {k}: original-variable count",
                len(features) == k
                and len(set(features)) == k
                and set(features) <= set(FEATURES),
            )
            for family in ["LR", "RF"]:
                bundle = joblib.load(out / f"models/fold_{fold}_{family}_{k}.joblib")
                g = pred.loc[
                    pred.fold.eq(fold) & pred.budget.eq(k) & pred.family.eq(family)
                ]
                check(
                    f"Fold {fold}/{family}/{k}: fitted roles and predictions match",
                    set(bundle["train_ids"]) == set(train)
                    and set(g.source_row) == set(valid)
                    and np.array_equal(g.actual, raw.loc[g.source_row, TARGET])
                    and np.array_equal(
                        bundle["model"].predict(raw.loc[g.source_row, features]),
                        g.predicted,
                    )
                    and np.allclose(
                        bundle["model"].predict_proba(raw.loc[g.source_row, features])[
                            :, 1
                        ],
                        g.probability,
                    ),
                )
                expected = calculate_metrics(g.actual, g.predicted, g.probability)
                stored = metrics.loc[
                    metrics.fold.eq(fold)
                    & metrics.family.eq(family)
                    & metrics.budget.eq(k)
                ].iloc[0]
                check(
                    f"Fold {fold}/{family}/{k}: all metrics reconstruct",
                    all(np.isclose(v, stored[key]) for key, v in expected.items()),
                )
    check(
        "Every budget and family scores each patient once",
        len(pred) == 734 * 8
        and pred.groupby(["budget", "family", "source_row"]).size().eq(1).all(),
    )
    freq = pd.read_csv(out / "tables/selection_frequency.csv")
    check(
        "Frequency counts reconstruct",
        all(
            row.folds_selected
            == sum(
                row.feature in json.loads(s)
                for s in selection.loc[selection.budget.eq(row.budget), "features"]
            )
            for _, row in freq.iterrows()
        ),
    )
    summary = pd.read_csv(out / "tables/summary.csv")
    check(
        "Summary means and SDs reconstruct from outer fold metrics",
        all(
            np.isclose(
                row[f"{metric}_{stat}"],
                getattr(
                    metrics.loc[
                        metrics.family.eq(row.family) & metrics.budget.eq(row.budget),
                        metric,
                    ],
                    stat,
                )(),
            )
            for _, row in summary.iterrows()
            for metric in ["accuracy", "f1", "recall", "specificity", "roc_auc"]
            for stat in ["mean", "std"]
        ),
    )
    adaptive = pd.read_csv(out / "tables/adaptive_compact_summary.csv")
    check(
        "Adaptive compact summaries use training-selected budgets",
        all(
            np.isclose(
                row[f"{metric}_mean"],
                metrics.loc[
                    metrics.family.eq(row.family) & metrics.chosen_compact, metric
                ].mean(),
            )
            for _, row in adaptive.iterrows()
            for metric in ["accuracy", "f1", "recall", "specificity"]
        ),
    )
    nb = nbformat.read(ROOT / "notebooks/02_feature_reduction.ipynb", as_version=4)
    nbformat.validate(nb)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    check(
        "All notebook code cells executed without error",
        [c.execution_count for c in cells] == list(range(1, len(cells) + 1))
        and not any(
            o.output_type == "error" for c in cells for o in c.get("outputs", [])
        ),
    )
    pdfpath = ROOT / "output/pdf/heart_disease_feature_selection.pdf"
    pdf = PdfReader(pdfpath)
    check(
        "Four intentional report pages",
        len(pdf.pages) == 4 and all(len(p.extract_text()) > 500 for p in pdf.pages),
    )
    final = joblib.load(out / "models/compact_model.joblib")
    check(
        "Final compact demo fits development only",
        set(final["train_ids"]) == dev and len(final["features"]) == final["budget"],
    )
    result = json.loads((out / "results.json").read_text())
    check(
        "Final demonstration budget follows declared tolerance",
        final["budget"]
        == choose_budget({int(k): v for k, v in result["final_inner_scores"].items()}),
    )
    write_json(
        out / "verification_summary.json",
        {
            "status": "passed",
            "checks_passed": len(checks),
            "checks": checks,
            "unit_tests": tests.testsRun,
            "pdf_pages": len(pdf.pages),
            "notebook_code_cells": len(cells),
            "pdf_sha256": hashlib.sha256(pdfpath.read_bytes()).hexdigest(),
            "visual_review": "Final rendered report must also be inspected.",
        },
    )
    print(
        f"PASS: {len(checks)} checks, {tests.testsRun} tests, {len(cells)} executed cells."
    )


if __name__ == "__main__":
    main()

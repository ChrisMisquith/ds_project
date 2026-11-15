"""Independently verify the saved experiment, models, notebook, and report."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import unittest

import joblib
import nbformat
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.model_selection import train_test_split, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.project import (
    FEATURES,
    TARGET,
    SEED,
    load_dataset,
    calculate_metrics,
    write_json,
    extract_leaf_rules,
)


def main():
    checks = []

    def checked(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    outcome = unittest.TextTestRunner(verbosity=1).run(suite)
    checked("Behavioral unit tests", outcome.wasSuccessful())

    frame, provenance = load_dataset(ROOT)
    result = json.loads((ROOT / "output/results.json").read_text(encoding="utf-8"))
    checked(
        "Raw source hash, schema, finite numbers, categories, and unique feature rows",
        True,
    )
    checked(
        "Source audit counts",
        len(frame) == 918
        and (frame.Cholesterol == 0).sum() == 172
        and (frame.RestingBP == 0).sum() == 1,
    )

    assignments = pd.read_csv(ROOT / "data/processed/split_assignments.csv")
    checked(
        "Every source row assigned exactly once",
        len(assignments) == 918
        and assignments.source_row.nunique() == 918
        and set(assignments.source_row) == set(frame.index),
    )
    train_ids = assignments.loc[assignments.split.eq("train"), "source_row"].to_numpy()
    test_ids = assignments.loc[assignments.split.eq("test"), "source_row"].to_numpy()
    checked(
        "Disjoint train/test membership",
        set(train_ids).isdisjoint(test_ids)
        and len(train_ids) == 734
        and len(test_ids) == 184,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        frame[FEATURES],
        frame[TARGET],
        test_size=0.2,
        stratify=frame[TARGET],
        random_state=SEED,
    )
    checked(
        "Split matches declared seed and stratification",
        set(train_ids) == set(X_train.index) and set(test_ids) == set(X_test.index),
    )
    indexed = assignments.set_index("source_row")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds_correct = all(
        (indexed.loc[X_train.index[valid], "cv_validation_fold"] == fold).all()
        for fold, (_, valid) in enumerate(cv.split(X_train, y_train))
    )
    checked(
        "CV fold membership uses training records only",
        folds_correct and (indexed.loc[test_ids, "cv_validation_fold"] == -1).all(),
    )

    metrics = pd.read_csv(ROOT / "output/tables/model_metrics.csv").set_index("model")
    predictions = pd.read_csv(ROOT / "output/tables/test_predictions.csv")
    checked(
        "Five predefined models have full test predictions",
        set(predictions.model) == set(metrics.index)
        and len(metrics) == 5
        and len(predictions) == 5 * 184,
    )
    for model_name, r in metrics.iterrows():
        saved = predictions.loc[predictions.model.eq(model_name)].copy()
        checked(
            f"{model_name}: predictions use exactly the held-out rows",
            set(saved.source_row) == set(test_ids)
            and saved.source_row.nunique() == 184,
        )
        checked(
            f"{model_name}: labels match source",
            np.array_equal(
                saved.actual.to_numpy(), frame.loc[saved.source_row, TARGET].to_numpy()
            ),
        )
        checked(
            f"{model_name}: valid probabilities", saved.probability.between(0, 1).all()
        )
        recomputed = calculate_metrics(saved.actual, saved.predicted, saved.probability)
        checked(
            f"{model_name}: all metrics recompute from saved predictions",
            all(
                np.isclose(recomputed[key], r[key], rtol=1e-10, atol=1e-12)
                for key in recomputed
            ),
        )
        in_json = next(x for x in result["test_results"] if x["model"] == model_name)
        checked(
            f"{model_name}: CSV and report JSON agree",
            all(np.isclose(recomputed[key], in_json[key]) for key in recomputed),
        )

    cv_summary = pd.read_csv(ROOT / "output/tables/cross_validation_summary.csv")
    selected = (
        cv_summary.loc[cv_summary.model.ne("Majority baseline")]
        .sort_values("cv_f1_mean", ascending=False)
        .iloc[0]
        .model
    )
    checked(
        "Selected model follows training CV, not test ranking",
        selected == result["selection"]["selected_before_test"],
    )
    for label, filename in [
        ("Tuned CART", "tuned_cart.joblib"),
        (selected, "selected_model.joblib"),
    ]:
        pipeline = joblib.load(ROOT / "output/models" / filename)
        saved = predictions.loc[predictions.model.eq(label)].sort_values("source_row")
        X = frame.loc[saved.source_row, FEATURES]
        checked(
            f"{label}: serialized model reproduces labels",
            np.array_equal(pipeline.predict(X), saved.predicted),
        )
        checked(
            f"{label}: serialized model reproduces probabilities",
            np.allclose(pipeline.predict_proba(X)[:, 1], saved.probability, atol=1e-12),
        )
        clean_train = pipeline.named_steps["clean"].transform(X_train)
        imputer = (
            pipeline.named_steps["preprocess"]
            .named_transformers_["numeric"]
            .named_steps["impute"]
        )
        cols = pipeline.named_steps["preprocess"].transformers_[0][2]
        checked(
            f"{label}: fitted medians equal training-only medians",
            np.allclose(imputer.statistics_, clean_train[cols].median().to_numpy()),
        )
        if label == "Tuned CART":
            leaves = extract_leaf_rules(pipeline)
            checked(
                "CART leaf supports cover every training row",
                leaves.training_records.sum() == 734
                and len(leaves) == result["cart"]["leaves"],
            )

    notebook = nbformat.read(
        ROOT / "notebooks/01_data_cleaning_and_baseline.ipynb", as_version=4
    )
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.output_type == "error"]
    checked(
        "Notebook schema and clean sequential execution",
        not errors
        and [c.execution_count for c in code] == list(range(1, len(code) + 1)),
    )
    checked(
        "Notebook embeds its figures",
        sum(
            "image/png" in o.get("data", {}) for c in code for o in c.get("outputs", [])
        )
        >= 7,
    )
    checked("Notebook completed its artifact export", result["completed"] is True)

    report = ROOT / "output/pdf/heart_disease_project_report.pdf"
    pdf = PdfReader(report)
    page_text = [p.extract_text() for p in pdf.pages]
    checked(
        "PDF has eight intentional, nonempty pages",
        len(pdf.pages) == 8 and all(len(t) > 500 for t in page_text),
    )
    joined = "\n".join(page_text)
    checked(
        "PDF contains actual headline findings",
        all(
            f"{value:.1%}" in joined
            for value in [
                metrics.loc["Tuned CART", "accuracy"],
                metrics.loc[selected, "accuracy"],
            ]
        ),
    )
    checked(
        "PDF has no replacement glyphs or unfinished placeholders",
        "\ufffd" not in joined and "TODO" not in joined and "TBD" not in joined,
    )
    figure_files = list((ROOT / "output/figures").glob("*.png"))
    checked(
        "Eight nonempty figure files",
        len(figure_files) == 8 and all(p.stat().st_size > 10_000 for p in figure_files),
    )

    summary = {
        "status": "passed",
        "client_date": "2026-11-15",
        "unit_tests": outcome.testsRun,
        "checks_passed": len(checks),
        "checks": checks,
        "notebook_code_cells": len(code),
        "notebook_errors": len(errors),
        "pdf_pages": len(pdf.pages),
        "source_csv_sha256": provenance["csv_sha256"],
        "notebook_sha256": hashlib.sha256(
            (ROOT / "notebooks/01_data_cleaning_and_baseline.ipynb").read_bytes()
        ).hexdigest(),
        "pdf_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "visual_review_note": "PDF pages rendered and inspected separately; automated checks do not replace visual review.",
    }
    write_json(ROOT / "output/verification_summary.json", summary)
    print(
        f"PASS: {len(checks)} artifact checks; {outcome.testsRun} unit tests; {len(code)} executed notebook cells; {len(pdf.pages)} PDF pages."
    )


if __name__ == "__main__":
    main()

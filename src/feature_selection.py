"""Group-wise greedy forward selection with nested development evaluation.

Groups are ORIGINAL clinical variables, not individual encoded dummy columns.
Selection uses LR inner-CV F1; RF is a predefined secondary comparison using
those same LR-selected subsets. No outer result is used to choose a subset.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.project import (
    FEATURES,
    TARGET,
    make_model_pipeline,
    load_dataset,
    evaluate_pipeline,
    write_json,
)

BUDGETS = [3, 5, 7, 11]
TOLERANCE = 0.01


def model_for(features, family="LR"):
    estimator = (
        LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        if family == "LR"
        else RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1
        )
    )
    return make_model_pipeline(
        estimator, scale_numeric=family == "LR", features=list(features)
    )


def choose_budget(scores, tolerance=TOLERANCE):
    """Smallest predefined subset within 0.01 absolute inner F1 of all 11."""
    return min(k for k in sorted(scores) if scores[k] >= scores[11] - tolerance - 1e-12)


def forward_path(X, y, cv, max_features=7):
    selected = []
    candidates = []
    subsets = {}
    scores = {}
    for step in range(1, max_features + 1):
        step_scores = []
        for feature in FEATURES:
            if feature in selected:
                continue
            subset = selected + [feature]
            values = cross_val_score(
                model_for(subset),
                X[subset],
                y,
                cv=cv,
                scoring="f1",
                n_jobs=1,
                error_score="raise",
            )
            score = float(values.mean())
            step_scores.append((feature, score))
            candidates.append(
                {
                    "step": step,
                    "added_feature": feature,
                    "subset": json.dumps(subset),
                    "inner_f1": score,
                    "inner_f1_sd": float(values.std()),
                }
            )
        winner = max(step_scores, key=lambda item: item[1])[
            0
        ]  # fixed FEATURES order resolves ties
        selected.append(winner)
        if step in BUDGETS:
            subsets[step] = selected.copy()
            scores[step] = dict(step_scores)[winner]
    subsets[11] = FEATURES.copy()
    scores[11] = float(
        cross_val_score(
            model_for(FEATURES),
            X,
            y,
            cv=cv,
            scoring="f1",
            n_jobs=1,
            error_score="raise",
        ).mean()
    )
    return subsets, scores, candidates


def run_study(root):
    out = root / "output/feature_selection"
    for folder in ["tables", "models", "figures"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    protocol = {
        "date": "2026-11-15",
        "outer_folds": 5,
        "inner_folds": 3,
        "seed": 42,
        "budgets": BUDGETS,
        "selection_model": "LR C=1",
        "score": "F1",
        "compact_rule": "smallest budget within 0.01 absolute inner F1 of all 11",
        "scope": "734 original development records; original holdout excluded",
        "secondary_model": "fixed RF on LR-selected subsets, not separately RF-optimized",
    }
    write_json(out / "protocol.json", protocol)
    frame, prov = load_dataset(root)
    split = pd.read_csv(root / "data/processed/split_assignments.csv")
    ids = split.loc[split.split.eq("train"), "source_row"].to_numpy()
    X, y = frame.loc[ids, FEATURES], frame.loc[ids, TARGET]
    rows = []
    predictions = []
    choices = []
    candidates = []
    lineage = []
    for fold, (train, valid) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)
    ):
        train_ids = X.index[train]
        valid_ids = X.index[valid]
        inner = StratifiedKFold(3, shuffle=True, random_state=500 + fold)
        subsets, scores, logs = forward_path(X.loc[train_ids], y.loc[train_ids], inner)
        compact = choose_budget(scores)
        candidates.extend({"fold": fold, **r} for r in logs)
        for role, source_ids in [("train", train_ids), ("validation", valid_ids)]:
            lineage.extend(
                {"fold": fold, "source_row": int(i), "role": role} for i in source_ids
            )
        for k, subset in subsets.items():
            choices.append(
                {
                    "fold": fold,
                    "budget": k,
                    "features": json.dumps(subset),
                    "inner_f1": scores[k],
                    "chosen_compact": k == compact,
                }
            )
            for family in ["LR", "RF"]:
                pipeline = model_for(subset, family).fit(
                    X.loc[train_ids, subset], y.loc[train_ids]
                )
                metrics, pred, p = evaluate_pipeline(
                    pipeline, X.loc[valid_ids, subset], y.loc[valid_ids]
                )
                rows.append(
                    {
                        "fold": fold,
                        "budget": k,
                        "family": family,
                        "chosen_compact": k == compact,
                        **metrics,
                    }
                )
                predictions.extend(
                    {
                        "fold": fold,
                        "budget": k,
                        "family": family,
                        "source_row": int(i),
                        "actual": int(actual),
                        "predicted": int(predicted),
                        "probability": float(probability),
                    }
                    for i, actual, predicted, probability in zip(
                        valid_ids, y.loc[valid_ids], pred, p
                    )
                )
                joblib.dump(
                    {
                        "model": pipeline,
                        "features": subset,
                        "train_ids": list(map(int, train_ids)),
                    },
                    out / f"models/fold_{fold}_{family}_{k}.joblib",
                    compress=3,
                )
        print(
            f"Outer fold {fold+1}/5: inner-selected compact budget {compact}; subset {subsets[compact]}",
            flush=True,
        )
    metrics = pd.DataFrame(rows)
    selection = pd.DataFrame(choices)
    summary = metrics.groupby(["family", "budget"])[
        ["accuracy", "f1", "recall", "specificity", "roc_auc"]
    ].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    adaptive = (
        metrics.loc[metrics.chosen_compact]
        .groupby("family")[["accuracy", "f1", "recall", "specificity"]]
        .agg(["mean", "std"])
    )
    adaptive.columns = ["_".join(c) for c in adaptive.columns]
    adaptive = adaptive.reset_index()
    frequencies = []
    stability = []
    for k in [3, 5, 7]:
        sets = [
            set(json.loads(s))
            for s in selection.loc[selection.budget.eq(k), "features"]
        ]
        for feature in FEATURES:
            frequencies.append(
                {
                    "budget": k,
                    "feature": feature,
                    "folds_selected": sum(feature in s for s in sets),
                    "selection_frequency": np.mean([feature in s for s in sets]),
                }
            )
        pairs = [
            len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1 :]
        ]
        stability.append(
            {
                "budget": k,
                "mean_pairwise_jaccard": float(np.mean(pairs)),
                "folds": len(sets),
            }
        )
    tables = {
        "fold_metrics": metrics,
        "outer_predictions": pd.DataFrame(predictions),
        "selected_subsets": selection,
        "candidate_search": pd.DataFrame(candidates),
        "fold_membership": pd.DataFrame(lineage),
        "summary": summary,
        "adaptive_compact_summary": adaptive,
        "selection_frequency": pd.DataFrame(frequencies),
        "stability": pd.DataFrame(stability),
    }
    for name, t in tables.items():
        t.to_csv(out / f"tables/{name}.csv", index=False)
    # Full-development selection is for the final demo; it does not replace outer scoring.
    subsets, scores, logs = forward_path(
        X, y, StratifiedKFold(3, shuffle=True, random_state=900)
    )
    k = choose_budget(scores)
    features = subsets[k]
    model = model_for(features).fit(X[features], y)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "budget": k,
            "train_ids": list(map(int, ids)),
        },
        out / "models/compact_model.joblib",
        compress=3,
    )
    (root / "examples").mkdir(exist_ok=True)
    X.loc[ids[:3], features].to_csv(root / "examples/compact_inputs.csv", index=False)
    pd.DataFrame(logs).to_csv(out / "tables/final_development_search.csv", index=False)
    result = {
        "completed": True,
        "protocol": protocol,
        "patients": len(ids),
        "source_sha256": prov["csv_sha256"],
        "final_compact_budget": k,
        "final_compact_features": features,
        "final_inner_scores": scores,
        "summary": json.loads(summary.to_json(orient="records")),
        "adaptive_summary": json.loads(adaptive.to_json(orient="records")),
        "stability": stability,
        "warning": "Development evaluation; no external validation or claim of universally important clinical variables.",
    }
    write_json(out / "results.json", result)
    return result


def create_figures(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.charts import configure_style, save, NAVY, TEAL

    configure_style()
    out = root / "output/feature_selection"
    summary = pd.read_csv(out / "tables/summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for family, color in [("LR", TEAL), ("RF", NAVY)]:
        s = summary.loc[summary.family.eq(family)].sort_values("budget")
        for ax, metric in zip(axes, ["f1", "recall"]):
            ax.errorbar(
                s.budget,
                s[f"{metric}_mean"],
                yerr=s[f"{metric}_std"],
                fmt="o-",
                color=color,
                label=family,
                capsize=3,
            )
    for ax, label in zip(axes, ["F1 score", "Disease recall"]):
        ax.set(
            title=f"{label} versus feature budget",
            xlabel="Original clinical inputs retained",
            ylabel=label,
            xticks=BUDGETS,
            ylim=(0.5, 1.02),
        )
        ax.legend(frameon=False)
        ax.grid(alpha=0.15)
    save(fig, out / "figures", "01_performance.png")
    freq = pd.read_csv(out / "tables/selection_frequency.csv")
    pivot = freq.pivot(
        index="feature", columns="budget", values="folds_selected"
    ).reindex(FEATURES)
    fig, ax = plt.subplots(figsize=(7.5, 4.7), layout="constrained")
    ax.imshow(pivot.to_numpy(), cmap="Blues", vmin=0, vmax=5, aspect="auto")
    for i in range(11):
        for j in range(3):
            ax.text(
                j,
                i,
                str(int(pivot.iloc[i, j])),
                ha="center",
                va="center",
                color="white" if pivot.iloc[i, j] >= 3 else NAVY,
            )
    ax.set(
        yticks=range(11),
        yticklabels=FEATURES,
        xticks=range(3),
        xticklabels=["3 inputs", "5 inputs", "7 inputs"],
        xlabel="Feature budget",
        ylabel="Original clinical variable",
        title="Selection stability | number of outer training folds (out of 5)",
    )
    save(fig, out / "figures", "02_stability.png")

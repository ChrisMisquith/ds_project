"""Consistent, publication-style plots for the notebook and PDF report."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, precision_recall_curve
from sklearn.tree import plot_tree

NAVY = "#17324d"
TEAL = "#087f8c"
GOLD = "#d39036"
GRAY = "#66758a"
COLORS = [GRAY, "#627ab7", GOLD, TEAL, "#935a91"]


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlecolor": NAVY,
            "axes.labelcolor": NAVY,
            "text.color": NAVY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 110,
        }
    )


def save(fig, folder: Path, name: str):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return path


def dataset_audit(frame, folder):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    counts = frame.HeartDisease.value_counts().sort_index()
    bars = axes[0].bar(
        ["Normal (0)", "Heart disease (1)"],
        counts.values,
        color=[NAVY, TEAL],
        width=0.55,
    )
    axes[0].bar_label(bars, padding=4)
    axes[0].set(
        title="Target distribution | all 918 records", ylabel="Records", ylim=(0, 600)
    )
    bars = axes[1].bar(
        ["Zero cholesterol", "Zero resting BP"],
        [(frame.Cholesterol <= 0).sum(), (frame.RestingBP <= 0).sum()],
        color=[GOLD, GRAY],
        width=0.55,
    )
    axes[1].bar_label(bars, padding=4)
    axes[1].set(
        title="Measurements requiring treatment", ylabel="Records", ylim=(0, 205)
    )
    return save(fig, folder, "01_dataset_audit.png")


def training_eda(X, y, folder):
    data = X.assign(HeartDisease=y)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    bins = np.arange(25, 81, 5)
    for label, color, name in [(0, NAVY, "Normal"), (1, TEAL, "Heart disease")]:
        axes[0].hist(
            data.loc[data.HeartDisease == label, "Age"],
            bins=bins,
            alpha=0.65,
            label=name,
            color=color,
            edgecolor="white",
        )
    axes[0].set(
        title="Age distribution | training data only",
        xlabel="Age (years)",
        ylabel="Records",
    )
    axes[0].legend(frameon=False, fontsize=9)
    rates = (
        data.groupby("ST_Slope")["HeartDisease"]
        .agg(["mean", "size"])
        .reindex(["Up", "Flat", "Down"])
    )
    bars = axes[1].bar(
        rates.index, rates["mean"] * 100, color=[NAVY, TEAL, GOLD], width=0.6
    )
    axes[1].bar_label(
        bars,
        labels=[f"{v:.0%}\nn={n}" for v, n in zip(rates["mean"], rates["size"])],
        padding=4,
        fontsize=9,
    )
    axes[1].set(
        title="Disease proportion by ST slope",
        xlabel="ST slope category",
        ylabel="Training records with disease (%)",
        ylim=(0, 110),
    )
    return save(fig, folder, "02_training_exploration.png")


def cv_comparison(cv_table, folder):
    ordered = cv_table.sort_values("cv_f1_mean")
    fig, ax = plt.subplots(figsize=(9.5, 3.8), layout="constrained")
    colors = [TEAL if x == "Tuned CART" else NAVY for x in ordered.model]
    ax.barh(
        ordered.model,
        ordered.cv_f1_mean,
        xerr=ordered.cv_f1_std,
        color=colors,
        alpha=0.9,
        height=0.55,
        capsize=4,
    )
    for i, (_, row) in enumerate(ordered.iterrows()):
        ax.text(
            row.cv_f1_mean + 0.013, i, f"{row.cv_f1_mean:.3f}", va="center", fontsize=9
        )
    ax.set(
        xlim=(0, 1.02),
        xlabel="Mean F1 score (error bars: fold standard deviation)",
        title="Model selection | five-fold training cross-validation",
    )
    ax.grid(axis="x", alpha=0.15)
    return save(fig, folder, "03_cross_validation.png")


def confusion_panels(y, predictions, folder):
    names = ["Logistic regression", "Tuned CART", "Random forest"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), layout="constrained")
    for ax, name in zip(axes, names):
        ConfusionMatrixDisplay.from_predictions(
            y,
            predictions[name],
            display_labels=["Normal", "Disease"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
            values_format="d",
        )
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
    return save(fig, folder, "04_test_confusion_matrices.png")


def discrimination_curves(y, probabilities, folder):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    for name, color in [
        ("Logistic regression", NAVY),
        ("Tuned CART", TEAL),
        ("Random forest", GOLD),
    ]:
        fpr, tpr, _ = roc_curve(y, probabilities[name])
        precision, recall, _ = precision_recall_curve(y, probabilities[name])
        axes[0].plot(fpr, tpr, label=name, color=color, lw=2)
        axes[1].plot(recall, precision, label=name, color=color, lw=2)
    axes[0].plot([0, 1], [0, 1], "--", color=GRAY, lw=1, label="Chance reference")
    axes[1].axhline(
        np.mean(y), ls="--", color=GRAY, lw=1, label="Positive-class prevalence"
    )
    axes[0].set(
        title="ROC curves | final test set",
        xlabel="False positive rate",
        ylabel="True positive rate / recall",
    )
    axes[1].set(
        title="Precision-recall curves | final test set",
        xlabel="Recall",
        ylabel="Precision",
    )
    for ax in axes:
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.04))
        ax.legend(loc="lower left", fontsize=8, frameon=False)
        ax.grid(alpha=0.13)
    return save(fig, folder, "05_test_discrimination.png")


def feature_importance(importance, folder):
    ordered = importance.sort_values("f1_decrease_mean")
    fig, ax = plt.subplots(figsize=(9.3, 4.7), layout="constrained")
    ax.barh(
        ordered.feature,
        ordered.f1_decrease_mean,
        xerr=ordered.f1_decrease_std,
        color=TEAL,
        capsize=3,
        height=0.6,
    )
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set(
        title="Tuned CART | permutation importance on the test set",
        xlabel="Decrease in F1 when one original input is shuffled",
    )
    ax.grid(axis="x", alpha=0.13)
    return save(fig, folder, "06_cart_feature_importance.png")


def decision_tree_view(pipeline, folder):
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(17, 7.8), layout="constrained")
    plot_tree(
        pipeline.named_steps["model"],
        feature_names=feature_names,
        class_names=["Normal", "Disease"],
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=8,
        max_depth=3,
        ax=ax,
    )
    ax.set_title(
        "Tuned CART: first three split levels | deeper branches shown as (...)", pad=18
    )
    return save(fig, folder, "07_cart_tree_preview.png")

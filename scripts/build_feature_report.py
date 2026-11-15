"""Four-page report generated from the executed feature-selection experiment."""

from pathlib import Path
import json
from xml.sax.saxutils import escape
import pandas as pd
import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Table,
    TableStyle,
    Image,
    Spacer,
    PageBreak,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/feature_selection"
r = json.loads((OUT / "results.json").read_text())
s = pd.read_csv(OUT / "tables/summary.csv")
a = pd.read_csv(OUT / "tables/adaptive_compact_summary.csv")
choices = pd.read_csv(OUT / "tables/selected_subsets.csv")
stability = pd.read_csv(OUT / "tables/stability.csv")
for name, file in [("FB", "DejaVuSans.ttf"), ("FBB", "DejaVuSans-Bold.ttf")]:
    pdfmetrics.registerFont(
        TTFont(name, str(Path(matplotlib.get_data_path()) / "fonts/ttf" / file))
    )
pdfmetrics.registerFontFamily("FB", normal="FB", bold="FBB")
navy = colors.HexColor("#17324d")
teal = colors.HexColor("#087f8c")
gray = colors.HexColor("#59697b")
pale = colors.HexColor("#eef4f7")
W = A4[0] - 96
styles = {
    "body": ParagraphStyle(
        "fsbody", fontName="FB", fontSize=9.3, leading=14, textColor=navy, spaceAfter=9
    ),
    "small": ParagraphStyle(
        "fssmall", fontName="FB", fontSize=8, leading=12, textColor=gray, spaceAfter=8
    ),
    "title": ParagraphStyle(
        "fstitle",
        fontName="FBB",
        fontSize=23,
        leading=29,
        textColor=navy,
        spaceAfter=13,
    ),
    "h2": ParagraphStyle(
        "fsh2",
        fontName="FBB",
        fontSize=12,
        leading=16,
        textColor=teal,
        spaceBefore=10,
        spaceAfter=8,
        keepWithNext=True,
    ),
    "cell": ParagraphStyle(
        "fscell", fontName="FB", fontSize=7.8, leading=11, textColor=navy
    ),
    "head": ParagraphStyle(
        "fshead", fontName="FBB", fontSize=7.8, leading=11, textColor=colors.white
    ),
}
story = []


def add(t, style="body"):
    story.append(Paragraph(t, styles[style]))


def table(headers, rows, widths):
    data = [[Paragraph(escape(str(v)), styles["head"]) for v in headers]] + [
        [Paragraph(escape(str(v)), styles["cell"]) for v in row] for row in rows
    ]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), navy),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([t, Spacer(1, 8)])


def figure(name, width=W):
    im = Image(str(OUT / "figures" / name))
    im.drawHeight *= width / im.drawWidth
    im.drawWidth = width
    im.hAlign = "CENTER"
    story.append(im)


def page():
    story.append(PageBreak())


def furniture(c, d):
    c.saveState()
    c.setTitle("Compact Feature Selection for Heart Disease Prediction")
    c.setFont("FB", 7)
    c.setFillColor(gray)
    c.drawString(48, 28, "BTech data science | 15 November 2026 | Development study")
    c.drawRightString(A4[0] - 48, 28, str(d.page))
    c.restoreState()


add("Compact Feature Selection<br/>for Heart Disease Prediction", "title")
add("Can fewer clinical inputs preserve useful predictive performance?", "h2")
add(
    "This extension compares 3, 5, 7, and all 11 original clinical variables. Feature selection is "
    "performed inside training partitions and evaluated on matched outer validation patients. "
    "The baseline model comparison is documented in the companion report."
)
table(
    ["Model", "Inputs", "Accuracy", "F1", "Recall", "Specificity"],
    [
        [
            row.family,
            int(row.budget),
            f"{row.accuracy_mean:.1%}",
            f"{row.f1_mean:.3f}",
            f"{row.recall_mean:.1%}",
            f"{row.specificity_mean:.1%}",
        ]
        for _, row in s.iterrows()
    ],
    [60, 45, 95, 70, 95, W - 365],
)
add(
    "Means over five outer folds. These evaluate fold-specific selected subsets, not one fixed global subset. RF uses the LR-selected inputs.",
    "small",
)
lr = s.loc[s.family.eq("LR")].set_index("budget")
add(
    f'The three-input LR workflow achieves {lr.loc[3,"accuracy_mean"]:.2%} accuracy versus '
    f'{lr.loc[11,"accuracy_mean"]:.2%} with all 11. Recall is slightly higher, but specificity '
    "is lower. The smaller workflow therefore has a measured trade-off; the inner tolerance "
    "does not guarantee equivalent performance on outer validation records.",
    "small",
)
figure("01_performance.png")
add(
    "Figure 1. Performance versus original-variable count; error bars show descriptive fold SD. Fewer variables can reduce recall even if accuracy changes little.",
    "small",
)
page()

add("How the selection is evaluated", "title")
add(
    "Only the original 734 development records are used. The original 184 test records are "
    "excluded. Five stratified outer folds evaluate the full feature-selection procedure. "
    "Inside each outer training set, three-fold CV scores candidate subsets using F1."
)
add("Group-wise greedy forward selection", "h2")
add(
    "Start with no inputs. At each step, test each remaining original variable and add the "
    "one with the highest inner-CV F1. Ties follow the fixed feature order. Save the paths "
    "at 3, 5, and 7 variables and compare against all 11. Imputation, encoding, and scaling "
    "are fitted again within each candidate training fold. Encoded categories belonging to "
    "one clinical variable are kept together."
)
add(
    "The selection model is fixed logistic regression (C=1). The secondary random forest "
    "has 150 trees, depth 8, and minimum leaf size 2. These settings are predefined; no "
    "outer score is used to choose a setting. RF subsets are not independently optimized."
)
add("Training-only compact-model rule", "h2")
add(
    "For each outer fold, choose the smallest tested budget whose inner F1 is no more than "
    "0.01 below its all-11 inner F1. The tolerance is an explicit design choice, not a "
    "statistical non-inferiority margin. If no reduced model qualifies, retain all 11."
)
selected = choices.loc[choices.chosen_compact]
table(
    ["Outer fold", "Chosen inputs", "Training-selected subset"],
    [
        [int(row.fold) + 1, int(row.budget), ", ".join(json.loads(row.features))]
        for _, row in selected.iterrows()
    ],
    [65, 80, W - 145],
)
table(
    ["Adaptive compact workflow", "Accuracy", "F1", "Recall"],
    [
        [
            row.family,
            f"{row.accuracy_mean:.1%}",
            f"{row.f1_mean:.3f}",
            f"{row.recall_mean:.1%}",
        ]
        for _, row in a.iterrows()
    ],
    [210, 95, 95, W - 400],
)
add(
    "Adaptive results use each fold's training-only budget choice. Do not select a different budget by inspecting the outer table and describe that choice as independently validated.",
    "small",
)
page()

add("Which variables are stable?", "title")
figure("02_stability.png", width=W - 95)
add(
    "Figure 2. Number of outer training folds selecting each variable, out of five. These folds overlap; the frequencies describe selection consistency within this development dataset.",
    "small",
)
table(
    ["Feature budget", "Mean pairwise Jaccard similarity"],
    [
        [int(row.budget), f"{row.mean_pairwise_jaccard:.3f}"]
        for _, row in stability.iterrows()
    ],
    [160, W - 160],
)
add("A final compact model for demonstration", "h2")
add(
    f"The same selection procedure applied to all 734 development records chose "
    f'<b>{r["final_compact_budget"]} inputs</b>: '
    + escape(", ".join(r["final_compact_features"]))
    + "."
)
add(
    "The saved compact LR pipeline includes imputation and category handling. This exact final "
    "subset has no new independent performance test. Its inner score must not be reported as "
    "final test accuracy; outer results assess the selection workflow.",
    "small",
)
add(
    "ST slope and cholesterol appear in all five three-input subsets; the third variable varies. "
    "RestingECG appears in the final full-development subset but not the outer three-input "
    "subsets. Selecting cholesterol also retains its imputation/missingness indicator, so "
    "selection is not proof that measured cholesterol alone explains its usefulness.",
    "small",
)
page()

add("Interpretation and project contribution", "title")
add("What is added beyond the original paper?", "h2")
add(
    "The reference paper ranks feature importance. This project actually retrains reduced models, "
    "compares them on matched validation patients, measures disease recall and false negatives, "
    "and quantifies subset consistency. It tests whether a compact input set is useful instead "
    "of assuming that importance ranking alone proves which inputs can be removed."
)
add("What the experiment does not establish", "h2")
add(
    "Greedy selection may miss feature interactions. Inner scores are optimistically selected "
    "from many candidates; outer evaluation is therefore necessary. Importance and selection "
    "depend on the model and cohort, not universal medical value. The data is small and "
    "retrospective, and earlier project findings informed this design. No external validation "
    "or causal interpretation is claimed. Fold standard deviations are not independent "
    "confidence intervals."
)
add(
    "Fewer variables also do not automatically mean lower testing cost. ST slope, oldpeak, "
    "and maximum heart rate may require exercise or ECG measurements. Acquisition costs "
    "were not measured. The selected clinical variables should not be interpreted as "
    "recommendations to omit tests in real patients."
)
add("Reproduce and inspect", "h2")
add(
    "Run <b>python scripts/run_all.py --study features</b> from the project root using the "
    "project virtual environment. The executed notebook is "
    "notebooks/02_feature_reduction.ipynb. Candidate search scores, selected subsets, "
    "selection frequencies, fold membership, prediction-level results, complete metrics, "
    "figures, and models are in output/feature_selection/."
)
add(
    "Automated checks reconstruct selection "
    "decisions, metrics, and model predictions, and confirm that the original holdout is "
    "excluded. The report is also rendered and visually checked.",
    "small",
)
add("References", "h2")
for label, url in [
    ("Original CART paper", "https://doi.org/10.1016/j.health.2022.100130"),
    ("Dataset", "https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction"),
    (
        "Feature selection documentation",
        "https://scikit-learn.org/stable/modules/feature_selection.html",
    ),
]:
    add(f'<link href="{url}" color="#087f8c">{label}</link>', "small")
add(
    "Established methods, applied as a semester-project extension. AI assisted preparation; results are generated by execution.",
    "small",
)
target = ROOT / "output/pdf/heart_disease_feature_selection.pdf"
SimpleDocTemplate(
    str(target),
    pagesize=A4,
    leftMargin=48,
    rightMargin=48,
    topMargin=48,
    bottomMargin=48,
).build(story, onFirstPage=furniture, onLaterPages=furniture)
print("Created:", target)

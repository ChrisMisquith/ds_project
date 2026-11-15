"""Build the eight-page report entirely from executed experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    Flowable,
    Preformatted,
    KeepTogether,
)
from sklearn.tree import plot_tree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import charts

OUT = ROOT / "output"
RESULTS = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
if not RESULTS.get("completed"):
    raise RuntimeError(
        "Execute the notebook successfully before generating the report."
    )
metrics = pd.read_csv(OUT / "tables/model_metrics.csv").set_index("model")
cv = pd.read_csv(OUT / "tables/cross_validation_summary.csv")
groups = pd.read_csv(OUT / "tables/selected_model_subgroups.csv")
rules = pd.read_csv(OUT / "tables/cart_leaf_rules.csv")
importance = pd.read_csv(OUT / "tables/cart_permutation_importance.csv")
cart = metrics.loc["Tuned CART"]
lr = metrics.loc["Logistic regression"]
forest = metrics.loc["Random forest"]
unrestricted = metrics.loc["Unrestricted CART"]
selected = RESULTS["selection"]["selected_before_test"]
chosen = metrics.loc[selected]
provenance = json.loads((ROOT / "data/raw/provenance.json").read_text(encoding="utf-8"))

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts/ttf"
for name, filename in [
    ("Body", "DejaVuSans.ttf"),
    ("BodyBold", "DejaVuSans-Bold.ttf"),
    ("BodyItalic", "DejaVuSans-Oblique.ttf"),
    ("Mono", "DejaVuSansMono.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
pdfmetrics.registerFontFamily(
    "Body", normal="Body", bold="BodyBold", italic="BodyItalic", boldItalic="BodyBold"
)
NAVY = colors.HexColor("#17324d")
TEAL = colors.HexColor("#087f8c")
GOLD = colors.HexColor("#c2832f")
GRAY = colors.HexColor("#59697b")
PALE = colors.HexColor("#eef4f7")
LINE = colors.HexColor("#d9e3e9")
WIDTH = A4[0] - 96

styles = {
    "eyebrow": ParagraphStyle(
        "eyebrow",
        fontName="BodyBold",
        fontSize=8.5,
        leading=13,
        textColor=TEAL,
        spaceAfter=12,
    ),
    "cover": ParagraphStyle(
        "cover",
        fontName="BodyBold",
        fontSize=28,
        leading=34,
        textColor=NAVY,
        spaceAfter=12,
    ),
    "title": ParagraphStyle(
        "title",
        fontName="BodyBold",
        fontSize=21,
        leading=27,
        textColor=NAVY,
        spaceAfter=12,
    ),
    "h2": ParagraphStyle(
        "h2",
        fontName="BodyBold",
        fontSize=11.5,
        leading=16,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    ),
    "body": ParagraphStyle(
        "body", fontName="Body", fontSize=9.3, leading=14, textColor=NAVY, spaceAfter=8
    ),
    "small": ParagraphStyle(
        "small",
        fontName="Body",
        fontSize=8.1,
        leading=11.7,
        textColor=GRAY,
        spaceAfter=7,
    ),
    "caption": ParagraphStyle(
        "caption",
        fontName="BodyItalic",
        fontSize=7.8,
        leading=11,
        textColor=GRAY,
        spaceAfter=8,
    ),
    "table": ParagraphStyle(
        "table", fontName="Body", fontSize=8.0, leading=11, textColor=NAVY
    ),
    "tablehead": ParagraphStyle(
        "tablehead",
        fontName="BodyBold",
        fontSize=7.9,
        leading=10.5,
        textColor=colors.white,
    ),
    "code": ParagraphStyle(
        "code",
        fontName="Mono",
        fontSize=7.8,
        leading=11,
        textColor=NAVY,
        backColor=PALE,
        borderPadding=10,
        spaceBefore=6,
        spaceAfter=12,
    ),
}
story = []


def p(text, style="body"):
    return Paragraph(text, styles[style])


def add(text, style="body"):
    story.append(p(text, style))


def heading(number, title):
    add(f"IMPLEMENTATION REPORT / {number:02d}", "eyebrow")
    add(title, "title")


def table(headers, rows, widths, *, font="table"):
    data = [[p(escape(str(x)), "tablehead") for x in headers]]
    data += [[p(escape(str(x)), font) for x in row] for row in rows]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, LINE),
            ]
        )
    )
    return t


def figure(name, width=WIDTH):
    image = Image(str(OUT / "figures" / name))
    image.drawHeight *= width / image.drawWidth
    image.drawWidth = width
    image.hAlign = "CENTER"
    story.append(image)


def bullet(label, text):
    add(f"<b>{label}</b> {text}")


def page():
    story.append(PageBreak())


class Cards(Flowable):
    def __init__(self, items):
        super().__init__()
        self.items = items
        self.width = WIDTH
        self.height = 91

    def draw(self):
        gap = 10
        width = (WIDTH - 2 * gap) / 3
        for i, (value, label, detail) in enumerate(self.items):
            x = i * (width + gap)
            self.canv.setFillColor(PALE)
            self.canv.roundRect(x, 8, width, 80, 6, fill=1, stroke=0)
            self.canv.setFillColor(TEAL)
            self.canv.setFont("BodyBold", 24)
            self.canv.drawString(x + 12, 57, value)
            self.canv.setFillColor(NAVY)
            self.canv.setFont("BodyBold", 8)
            self.canv.drawString(x + 12, 39, label)
            self.canv.setFillColor(GRAY)
            self.canv.setFont("Body", 7.3)
            self.canv.drawString(x + 12, 23, detail)


def furniture(canvas, doc):
    canvas.saveState()
    canvas.setTitle("Heart Disease Classification - BTech Implementation and Findings")
    canvas.setAuthor("BTech Student Project")
    canvas.setSubject(
        "Executed CART implementation, model comparisons, findings and future work"
    )
    if doc.page > 1:
        canvas.setFillColor(GRAY)
        canvas.setFont("BodyBold", 7.3)
        canvas.drawString(
            48, A4[1] - 29, "HEART DISEASE CLASSIFICATION  /  BTECH PROJECT"
        )
    canvas.setStrokeColor(LINE)
    canvas.line(48, 41, A4[0] - 48, 41)
    canvas.setFont("Body", 7.1)
    canvas.setFillColor(GRAY)
    canvas.drawString(48, 27, "Academic implementation | 15 November 2026")
    canvas.drawRightString(A4[0] - 48, 27, f"{doc.page:02d}")
    canvas.restoreState()


# Produce a readable, deliberately abridged tree for the printed page.
pipeline = joblib.load(OUT / "models/tuned_cart.joblib")
charts.configure_style()
fig, ax = plt.subplots(figsize=(10.4, 5.2), layout="constrained")
plot_tree(
    pipeline.named_steps["model"],
    feature_names=pipeline.named_steps["preprocess"].get_feature_names_out(),
    class_names=["Normal", "Disease"],
    filled=True,
    rounded=True,
    impurity=False,
    max_depth=2,
    fontsize=10,
    ax=ax,
)
ax.set_title(
    "First two split levels; (...) indicates a deeper branch", fontsize=11, pad=14
)
charts.save(fig, OUT / "figures", "08_cart_report_overview.png")

# PAGE 1: an accessible brief with actual findings.
add("BTECH / DATA SCIENCE LAB / EXECUTED PROJECT", "eyebrow")
add("Heart Disease<br/>Classification", "cover")
add("Implementation, findings, limitations, and next steps", "h2")
add(
    "An interpretable machine learning study inspired by Ozcan and Peker (2023). "
    "The project uses a public tabular benchmark, a Python implementation of CART, "
    "and predefined model comparisons. All numerical findings below come from the executed notebook."
)
story.append(Spacer(1, 7))
story.append(
    Cards(
        [
            (
                f"{cart.accuracy:.1%}",
                "Tuned CART accuracy",
                "184 held-out test records",
            ),
            (f"{chosen.accuracy:.1%}", "CV-selected model accuracy", selected),
            (
                str(RESULTS["dataset"]["rows"]),
                "Original records retained",
                "11 original input features",
            ),
        ]
    )
)
add("The main finding", "h2")
add(
    f"Logistic regression was selected by the training cross-validation protocol "
    f'(mean F1 {cv.set_index("model").loc["Logistic regression","cv_f1_mean"]:.3f}). '
    f"Its held-out F1 was {lr.f1:.3f}. Random forest reached {forest.accuracy:.1%} test accuracy "
    f"and F1 {forest.f1:.3f}, but its slightly higher test score did not change the selection. "
    "The CV scores for those two models were nearly tied."
)
add(
    f'Tuned CART remained easy to inspect, with {RESULTS["cart"]["leaves"]} leaves, '
    f"but its {cart.accuracy:.1%} test accuracy was lower than the paper's reported approximately 87%. "
    "Different cleaning, categorical handling, tuning, and test partitions prevent a direct superiority claim."
)
add("Project scope", "h2")
story.append(
    table(
        ["Question", "Implemented answer"],
        [
            (
                "What is predicted?",
                "Existing heart disease label: 1=disease, 0=normal.",
            ),
            (
                "How is it evaluated?",
                "734 training records; 184 reserved test records; five-fold training CV.",
            ),
            (
                "What is delivered?",
                "Executed notebook, preserved source data, models, predictions, figures, and this report.",
            ),
        ],
        [118, WIDTH - 118],
    )
)
story.append(Spacer(1, 10))
add(
    "This is an academic benchmark study. It does not predict future disease onset or establish clinical suitability.",
    "small",
)
page()

# PAGE 2: data and cleaning.
heading(2, "Dataset and data quality")
add(
    "The selected Heart Failure Prediction Dataset contains 918 records, 11 inputs, and one binary "
    "target. Despite the title, the target is heart disease presence. The source combines older "
    "collections also used by the reference paper; it is not an independent cohort."
)
figure("01_dataset_audit.png")
add(
    "Figure 1. Basic source audit. There are 508 disease labels and 410 normal labels.",
    "caption",
)
add("A conservative, explicit cleaning policy", "h2")
bullet(
    "172 zero cholesterol values and one zero resting BP:",
    "treat these as missing measurements and learn median replacements inside each training fold. "
    "Missingness indicators preserve whether a measurement was unavailable.",
)
bullet(
    "Valid zeros are retained:",
    "binary zeros and zero/negative oldpeak are not removed. No exact duplicate rows or repeated "
    "input vectors were found. All 918 records are retained; this differs from the paper's deletion policy.",
)
add("Inputs used by the model", "h2")
story.append(
    table(
        ["Group", "Inputs"],
        [
            (
                "Numerical / binary",
                "Age, resting BP, cholesterol, fasting blood sugar, maximum heart rate, oldpeak",
            ),
            (
                "Nominal categories",
                "Sex, chest pain type, resting ECG, exercise angina, ST slope",
            ),
            ("Target", "HeartDisease: 1 for heart disease; 0 for normal"),
        ],
        [130, WIDTH - 130],
    )
)
story.append(Spacer(1, 10))
add(
    "The original CSV is unchanged. Its source, retrieval time, and SHA-256 checksum are recorded in "
    "data/raw/provenance.json. See data/processed/feature_dictionary.csv for category definitions.",
    "small",
)
page()

# PAGE 3: implementation, with code and settings.
heading(3, "Implementation and evaluation")
add(
    "An 80:20 split is stratified by the target, with random seed 42. The training set contains "
    "406 disease and 328 normal labels; the test set contains 102 disease and 82 normal labels. "
    "Exploratory association plots use training data only."
)
add("The preprocessing pipeline", "h2")
add(
    "Raw inputs pass through a deterministic invalid-value cleaner, a column transformer "
    "(training-fold median imputation plus category encoding), and the classifier. "
    "Only logistic regression also scales numerical inputs. This arrangement keeps learned "
    "statistics out of the validation and test records."
)
story.append(
    Preformatted(
        "pipeline = make_model_pipeline(classifier)\n"
        "search = GridSearchCV(\n"
        "    pipeline, parameter_grid, cv=stratified_cv,\n"
        '    scoring={"f1": "f1", "recall": "recall",\n'
        '             "roc_auc": "roc_auc"},\n'
        '    refit="f1", error_score="raise"\n'
        ")\n"
        "search.fit(X_train, y_train)  # training only",
        styles["code"],
    )
)
add("Predefined comparison models", "h2")
story.append(
    table(
        ["Model", "Settings searched", "Purpose"],
        [
            (
                "Majority baseline",
                "1",
                "Checks whether learning improves on always predicting the majority class",
            ),
            (
                "Logistic regression",
                "6",
                "Linear comparison: regularization C and optional class balancing",
            ),
            (
                "Unrestricted CART",
                "1",
                "Shows the behavior of an unconstrained single tree",
            ),
            (
                "Tuned CART",
                "90",
                "Depth 2-6; minimum leaf size 5/10/20; pruning 0/.005/.01; optional balancing",
            ),
            (
                "Random forest",
                "8",
                "200 trees; depth, minimum leaf size, and feature subsampling varied",
            ),
        ],
        [118, 65, WIDTH - 183],
    )
)
story.append(Spacer(1, 10))
add(
    "Selection rule: highest mean five-fold training-CV F1. Each candidate is then refitted on "
    "all training records. The reserved test set is used for final comparison, not for retuning "
    "or choosing a new winner. CV standard deviations and search logs are saved.",
    "small",
)
page()

# PAGE 4: numerical results.
heading(4, "Results on the reserved test set")
add(
    "Models below remain in training-CV order. Class 1 is the positive class. "
    "Recall measures detected disease cases; specificity measures correctly classified normal cases. "
    "F1 balances precision and recall, while ROC AUC evaluates score discrimination."
)
rows = []
for name in cv.model:
    r = metrics.loc[name]
    rows.append(
        [
            name,
            f'{cv.set_index("model").loc[name,"cv_f1_mean"]:.3f}',
            f"{r.accuracy:.1%}",
            f"{r.precision:.1%}",
            f"{r.recall:.1%}",
            f"{r.specificity:.1%}",
            f"{r.f1:.3f}",
            f"{r.roc_auc:.3f}",
        ]
    )
story.append(
    table(
        [
            "Model",
            "CV F1",
            "Test acc.",
            "Precision",
            "Recall",
            "Spec.",
            "Test F1",
            "AUC",
        ],
        rows,
        [112, 48, 56, 61, 55, 53, 57, WIDTH - 442],
    )
)
story.append(Spacer(1, 12))
figure("04_test_confusion_matrices.png")
add(
    "Figure 2. Rows are actual labels; columns are predicted labels. All three matrices use the same 184 test records.",
    "caption",
)
add("Errors and uncertainty matter", "h2")
add(
    f"Tuned CART: {int(cart.tp)} true positives, {int(cart.tn)} true negatives, "
    f"{int(cart.fp)} false positives, and {int(cart.fn)} false negatives. "
    f"Its 95% Wilson accuracy interval is {cart.accuracy_ci_low:.1%}-{cart.accuracy_ci_high:.1%}."
)
add(
    f"Logistic regression: {int(lr.fn)} missed disease cases out of 102; "
    f"recall {lr.recall:.1%} with a 95% Wilson interval of "
    f"{lr.recall_ci_low:.1%}-{lr.recall_ci_high:.1%}. "
    f"Random forest misses {int(forest.fn)} cases on this split. "
    "These differences do not establish statistical or clinical superiority."
)
add(
    f"The unrestricted tree fits the training set perfectly ({unrestricted.train_accuracy:.0%} accuracy) "
    f"but reaches only {unrestricted.accuracy:.1%} on the test set, illustrating the danger of judging a model by training accuracy.",
    "small",
)
page()

# PAGE 5: CART interpretability.
heading(5, "What the CART model learned")
add(
    f'The selected CART has depth {RESULTS["cart"]["depth"]}, '
    f'{RESULTS["cart"]["leaves"]} leaves, and a minimum leaf size of 10. '
    "The selected pruning parameter is 0, so depth and leaf-size constraints provide the "
    "selected regularization; additional cost-complexity pruning was not selected."
)
figure("08_cart_report_overview.png")
add(
    "Figure 3. An abridged view of the fitted tree. The complete paths are available in cart_rules.txt and cart_leaf_rules.csv.",
    "caption",
)
add("Examples from the largest leaves", "h2")
for _, rule in rules.head(3).iterrows():
    label = "disease" if rule.predicted_class == 1 else "normal"
    add(
        f"<b>{int(rule.training_records)} training records; predicted {label}.</b><br/>"
        f"{escape(rule.rule)}.<br/>"
        f'<font color="#59697b">Fitted disease fraction: {rule.fitted_disease_fraction:.1%}.</font>',
        "small",
    )
add(
    "These are fitted decision paths, not medical recommendations. Fractions describe tree "
    "composition and are not calibrated patient risk estimates. Numerical thresholds are applied "
    "after training-data imputation. ASY means asymptomatic chest pain category.",
    "small",
)
page()

# PAGE 6: interpretation and subgroups.
heading(6, "Feature effects and subgroup audit")
figure("06_cart_feature_importance.png", width=WIDTH - 10)
add(
    "Figure 4. Test-set F1 decrease across 20 fixed-seed shuffles per original input. Bars are mean effects; error bars are shuffle standard deviations.",
    "caption",
)
top = ", ".join(importance.head(3).feature)
add(
    f"The largest permutation effects are for <b>{top}</b>. "
    "These describe reliance by this fitted tree. They do not establish causal risk factors; "
    "correlated features and collection practices can affect the ranking."
)
add(f"Subgroup observations for {selected}", "h2")
rows = []
for _, r in groups.iterrows():
    rows.append(
        [
            r["group"],
            str(int(r.n)),
            str(int(r.positive_cases)),
            f"{r.recall:.1%}",
            f"{r.recall_ci_low:.1%}-{r.recall_ci_high:.1%}",
        ]
    )
story.append(
    table(
        ["Group", "Records", "Disease cases", "Recall", "95% recall interval"],
        rows,
        [128, 62, 86, 65, WIDTH - 341],
    )
)
story.append(Spacer(1, 10))
add(
    "There are only six female disease cases in the test set. The resulting recall interval is "
    "very wide, so the table cannot establish fair performance across groups. Age groups were "
    "fixed at below 55 versus 55 or above for a descriptive check, not threshold tuning.",
    "small",
)
page()

# PAGE 7: honest extensions and limits.
heading(7, "Improvements and future work")
add("What was implemented in this project", "h2")
bullet(
    "A more inspectable workflow:",
    "source hashes, saved split assignments, full parameter-search logs, an executed notebook, "
    "prediction-level exports, and metrics recomputed from those predictions.",
)
bullet(
    "Data retention and controlled preprocessing:",
    "fold-specific imputation retains every source record. A deterministic cleaner avoids "
    "discarding valid zeros, and pipelines prevent learned preprocessing from using test records.",
)
bullet(
    "Comparisons and uncertainty:",
    f"logistic regression and a random forest were implemented alongside CART. The forest's "
    f"test F1 difference from tuned CART is {forest.f1-cart.f1:+.3f} on this split. "
    "Intervals and subgroup denominators make the limits of point estimates visible.",
)
add("Additions worth investigating next", "h2")
story.append(
    table(
        ["Priority", "Proposed addition", "Why it matters"],
        [
            (
                "1",
                "Independent external data",
                "Tests transfer beyond the overlapping source collections; requires compatible measurements and labels.",
            ),
            (
                "2",
                "Repeated / nested validation",
                "Measures sensitivity to the split and separates tuning from unbiased CV assessment.",
            ),
            (
                "3",
                "Missingness sensitivity study",
                "Checks whether zero-valued measurements and their indicators encode source-specific artifacts.",
            ),
            (
                "4",
                "Calibration and threshold study",
                "Uses separate validation data to study probability quality and recall/precision trade-offs.",
            ),
            (
                "5",
                "Richer data or a demo",
                "Add smoking/family history only if supported by a new dataset; a sample-record interface can demonstrate rule paths.",
            ),
        ],
        [52, 150, WIDTH - 202],
    )
)
add("What this study cannot establish", "h2")
add(
    "The benchmark is small, retrospective, and not representative of every patient population. "
    "One holdout and non-nested CV do not establish robust generalization. The data contains "
    "exercise-test and ECG information; it is not a lifestyle-only screening model. "
    "Source identifiers are not supplied, limiting reliable site-based validation. "
    "The reported paper metrics also contain a confusion-matrix labeling inconsistency, "
    "documented in the methodology notes.",
    "small",
)
page()

# PAGE 8: handoff and sources.
heading(8, "Reproducing and reviewing the work")
add("Project layout", "h2")
story.append(
    Preformatted(
        "data/raw/        Original CSV and provenance\n"
        "data/processed/  Split membership and feature dictionary\n"
        "docs/            Methodology and source-paper comparison\n"
        "notebooks/       Two executed study notebooks\n"
        "src/             Shared preprocessing and evaluation code\n"
        "scripts/         Download, execution, report, and checks\n"
        "tests/           Behavioral checks for evaluation errors\n"
        "output/          Figures, tables, models, JSON, and PDF",
        styles["code"],
    )
)
add("Run from the project root with Python 3.12", "h2")
story.append(
    Preformatted(
        "python -m venv .venv\n"
        ".venv\\Scripts\\python -m pip install -r requirements-lock.txt\n"
        ".venv\\Scripts\\python scripts/run_all.py",
        styles["code"],
    )
)
add(
    "The notebook already contains its outputs. Raw data is reused after hash verification, "
    "so rerunning the experiment does not require downloading it again. The verified run used "
    f'Python {RESULTS["python_version"]}, scikit-learn {RESULTS["versions"]["scikit-learn"]}, '
    f'and pandas {RESULTS["versions"]["pandas"]}. Exact environment versions are locked.',
    "small",
)
add("Verification procedure", "h2")
add(
    "Checks cover raw-data integrity, disjoint split and fold membership, metric orientation, "
    "training-only imputation, preserved legitimate zeros, prediction/metric consistency, "
    "saved-model reloads, notebook execution, and PDF text/page integrity. "
    "The final PDF was also rendered for visual inspection. Detailed results are saved in "
    "output/verification_summary.json.",
    "small",
)
add("References and attribution", "h2")
references = [
    (
        "Ozcan, M., &amp; Peker, S. (2023). A classification and regression tree algorithm for heart disease modeling and prediction. Healthcare Analytics, 3, 100130.",
        "https://doi.org/10.1016/j.health.2022.100130",
        "Paper DOI",
    ),
    (
        "fedesoriano. Heart Failure Prediction Dataset. Source retrieved 23 October 2026. Kaggle lists Database: Open Database, Contents: Copyright Original Authors.",
        provenance["source_page"],
        "Dataset source",
    ),
    (
        "UCI Machine Learning Repository. Heart Disease. Background for original collections and target definitions.",
        "https://archive.ics.uci.edu/dataset/45/heart%2Bdisease",
        "UCI documentation",
    ),
    (
        "scikit-learn. Common pitfalls and recommended practices; GridSearchCV; permutation importance.",
        "https://scikit-learn.org/stable/common_pitfalls.html",
        "Official documentation",
    ),
]
for i, (description, url, label) in enumerate(references, 1):
    add(
        f'{i}. {description} <link href="{escape(url)}" color="#087f8c">{label}</link>.',
        "small",
    )
add(
    "AI assisted code and report preparation. Findings were generated by execution; the student "
    "team should understand and review the implementation before presenting it.",
    "small",
)

folder = OUT / "pdf"
folder.mkdir(parents=True, exist_ok=True)
target = folder / "heart_disease_project_report.pdf"
document = SimpleDocTemplate(
    str(target),
    pagesize=A4,
    rightMargin=48,
    leftMargin=48,
    topMargin=51,
    bottomMargin=54,
    pageCompression=1,
)
document.build(story, onFirstPage=furniture, onLaterPages=furniture)
print("Created:", target)

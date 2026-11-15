"""Check the publication scope, readable links, and generated deliverables."""

from pathlib import Path
import hashlib
import json
import re
import sys

import nbformat
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.project import write_json


def main():
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    check(
        "Exactly the two requested study notebooks",
        [p.name for p in notebooks]
        == ["01_data_cleaning_and_baseline.ipynb", "02_feature_reduction.ipynb"],
    )
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        cells = [c for c in notebook.cells if c.cell_type == "code"]
        check(
            f"{path.name}: all cells executed in order",
            [c.execution_count for c in cells] == list(range(1, len(cells) + 1)),
        )
        check(
            f"{path.name}: no execution errors",
            not any(
                o.output_type == "error" for c in cells for o in c.get("outputs", [])
            ),
        )
    baseline = nbformat.read(notebooks[0], as_version=4)
    reduced = nbformat.read(notebooks[1], as_version=4)
    check(
        "Baseline notebook includes complete preprocessing implementation",
        any("def make_model_pipeline(" in c.source for c in baseline.cells),
    )
    check(
        "Feature notebook includes the full search and study implementation",
        all(
            any(f"def {name}(" in c.source for c in reduced.cells)
            for name in ["model_for", "choose_budget", "forward_path", "run_study"]
        ),
    )
    for path in [
        ROOT / "README.md",
        ROOT / "docs/methodology.md",
        ROOT / "data/README.md",
    ]:
        text = path.read_text(encoding="utf-8")
        targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
        local = [t.split("#")[0] for t in targets if not re.match(r"https?://", t)]
        check(
            f"{path.name}: all local document links exist",
            all((path.parent / t).exists() for t in local if t),
        )
    relevant = [
        p
        for folder in ["src", "scripts", "tests", "docs", "notebooks"]
        for p in (ROOT / folder).rglob("*")
        if p.suffix in {".py", ".md", ".ipynb"}
    ]
    check(
        "No uncertainty-study files included",
        not any("reliability" in p.name or "uncertainty" in p.name for p in relevant),
    )
    check(
        "No old notebook names or local user paths in project text",
        all(
            not any(
                term in p.read_text(encoding="utf-8")
                for term in [
                    "03_compact_feature_selection",
                    "02_missing_measurements",
                    "C:\\Users\\misqu",
                    "C:/Users/misqu",
                ]
            )
            for p in relevant if p.resolve() != Path(__file__).resolve()
        ),
    )
    reports = {
        "heart_disease_project_report.pdf": 8,
        "heart_disease_feature_selection.pdf": 4,
    }
    check(
        "Exactly the two scoped PDF reports",
        {p.name for p in (ROOT / "output/pdf").glob("*.pdf")} == set(reports),
    )
    for name, count in reports.items():
        reader = PdfReader(ROOT / "output/pdf" / name)
        text = "\n".join(p.extract_text() for p in reader.pages)
        check(
            f"{name}: complete PDF text and page count",
            len(reader.pages) == count and "\ufffd" not in text,
        )
        check(
            f"{name}: no excluded-study references",
            "uncertainty experiments" not in text and "hash-preserved" not in text,
        )
    original = pd.read_csv(ROOT / "output/tables/model_metrics.csv").set_index("model")
    feature = pd.read_csv(ROOT / "output/feature_selection/tables/summary.csv")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expected = [
        original.loc[name, "accuracy"]
        for name in ["Tuned CART", "Logistic regression", "Random forest"]
    ]
    expected += [
        feature.loc[
            feature.family.eq("LR") & feature.budget.eq(k), "accuracy_mean"
        ].iloc[0]
        for k in [3, 11]
    ]
    check(
        "README headline numbers match freshly generated metrics",
        all(f"{v:.2%}" in readme for v in expected),
    )
    result = json.loads((ROOT / "output/feature_selection/results.json").read_text())
    check(
        "README demonstration inputs match fitted subset",
        all(f"`{name}`" in readme for name in result["final_compact_features"]),
    )
    check(
        "Generated model files are excluded from Git",
        "*.joblib" in (ROOT / ".gitignore").read_text(),
    )
    summary = {
        "status": "passed",
        "checks_passed": len(checks),
        "checks": checks,
        "notebook_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in notebooks
        },
    }
    write_json(ROOT / "output/repository_verification.json", summary)
    print(f"PASS: {len(checks)} publication checks.")


if __name__ == "__main__":
    main()

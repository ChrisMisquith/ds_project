"""Rebuild the two scoped studies with fresh notebook kernels and verification."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(script, *arguments):
    print(f"\nRunning {script}", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", choices=["all", "baseline", "features"], default="all"
    )
    args = parser.parse_args()
    run("download_data.py")
    if args.study in {"all", "baseline"}:
        run("build_baseline_notebook.py")
        run("execute_notebook.py", "notebooks/01_data_cleaning_and_baseline.ipynb")
        run("build_baseline_report.py")
        run("validate_baseline.py")
    if args.study in {"all", "features"}:
        if not (ROOT / "data/processed/split_assignments.csv").exists():
            raise RuntimeError(
                "Run the baseline first to create the fixed development split."
            )
        run("build_feature_notebook.py")
        run("execute_notebook.py", "notebooks/02_feature_reduction.ipynb")
        run("build_feature_report.py")
        run("validate_feature_selection.py")
    if args.study == "all":
        run("validate_repository.py")
    print("\nRequested studies rebuilt and all automated checks passed.")


if __name__ == "__main__":
    main()

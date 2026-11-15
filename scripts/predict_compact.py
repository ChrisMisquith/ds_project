"""Predict with the saved compact model using only its selected clinical columns."""

from pathlib import Path
import argparse, sys
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/feature_selection/tables/user_predictions.csv",
    )
    args = parser.parse_args()
    bundle = joblib.load(ROOT / "output/feature_selection/models/compact_model.joblib")
    frame = pd.read_csv(args.input)
    missing = set(bundle["features"]) - set(frame)
    if missing:
        raise ValueError(
            f'Required columns: {bundle["features"]}; missing: {sorted(missing)}'
        )
    X = frame[bundle["features"]]
    output = pd.DataFrame(
        {
            "predicted_class": bundle["model"].predict(X),
            "disease_score": bundle["model"].predict_proba(X)[:, 1],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print("Inputs used:", ", ".join(bundle["features"]))
    print(output.to_string(index=False))
    print("Saved:", args.output.resolve())
    print("Academic predictions; not validated clinical risk estimates.")


if __name__ == "__main__":
    main()

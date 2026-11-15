"""Download the public source once, validate it, and record its provenance.

Run from the project root: python scripts/download_data.py
Existing data is verified against its recorded hash and never silently replaced.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/fedesoriano/heart-failure-prediction"
PAGE = "https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction"
EXPECTED = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
    "HeartDisease",
]


def main():
    folder = ROOT / "data" / "raw"
    folder.mkdir(parents=True, exist_ok=True)
    target, metadata = folder / "heart.csv", folder / "provenance.json"
    if target.exists():
        if not metadata.exists():
            raise RuntimeError(
                "Existing heart.csv has no provenance.json; inspect it before replacing it."
            )
        recorded = json.loads(metadata.read_text(encoding="utf-8"))
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != recorded["csv_sha256"]:
            raise ValueError(
                "Raw data hash differs from provenance. Original data may have changed."
            )
        print("Existing source file verified:", actual)
        return

    request = Request(URL, headers={"User-Agent": "BTech-Heart-Disease-Project/1.0"})
    with urlopen(request, timeout=60) as response:
        archive = response.read(2_000_001)
    if len(archive) > 2_000_000:
        raise ValueError(
            "Unexpectedly large download; inspect the source before proceeding."
        )
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        if "heart.csv" not in zipped.namelist():
            raise ValueError("The source archive no longer contains heart.csv.")
        content = zipped.read("heart.csv")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    rows = list(reader)
    if reader.fieldnames != EXPECTED or len(rows) != 918:
        raise ValueError(
            "Unexpected source schema or row count; inspect the new dataset version."
        )
    if {r["HeartDisease"] for r in rows} != {"0", "1"}:
        raise ValueError("Unexpected target labels.")

    record = {
        "dataset": "Heart Failure Prediction Dataset",
        "uploader": "fedesoriano",
        "source_page": PAGE,
        "download_url": URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "access_checked_client_date": "2026-10-23",
        "license_label_observed": "Database: Open Database, Contents: Copyright Original Authors",
        "license_metadata_source": "https://www.kaggle.com/api/v1/datasets/list?search=heart-failure-prediction",
        "license_note": "Retain source attribution; consult the source terms when redistributing derived data.",
        "zip_sha256": hashlib.sha256(archive).hexdigest(),
        "csv_sha256": hashlib.sha256(content).hexdigest(),
        "rows": len(rows),
        "input_features": 11,
        "target": "HeartDisease (1=heart disease, 0=normal)",
        "source_version_last_updated": "2021-09-10T18:11:57.517Z",
    }
    # All validation occurs before writing the original source.
    target.write_bytes(content)
    metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("Downloaded and validated 918 records; SHA-256:", record["csv_sha256"])


if __name__ == "__main__":
    main()

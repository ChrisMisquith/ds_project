"""Execute in a fresh project-local kernel and retain outputs on failure."""

import json
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / "tmp"
TEMP.mkdir(exist_ok=True)
# Use only project-local Jupyter state; no user-level kernel registration required.
for variable, folder in [
    ("JUPYTER_RUNTIME_DIR", "jupyter-runtime"),
    ("IPYTHONDIR", "ipython"),
    ("MPLCONFIGDIR", "matplotlib"),
]:
    path = TEMP / folder
    path.mkdir(exist_ok=True)
    os.environ[variable] = str(path)
kernel = TEMP / "jupyter-data/kernels/heart-project"
kernel.mkdir(parents=True, exist_ok=True)
(kernel / "kernel.json").write_text(
    json.dumps(
        {
            "argv": [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ],
            "display_name": "Heart Disease Project",
            "language": "python",
        }
    ),
    encoding="utf-8",
)
os.environ["JUPYTER_PATH"] = str(TEMP / "jupyter-data")

path = (
    (ROOT / sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else ROOT / "notebooks/01_data_cleaning_and_baseline.ipynb"
)
if not path.is_relative_to(ROOT) or path.suffix != ".ipynb":
    raise ValueError("Notebook must be a project-local .ipynb file.")
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="heart-project")
client = NotebookClient(
    notebook,
    timeout=900,
    kernel_name="heart-project",
    km=manager,
    resources={"metadata": {"path": str(ROOT)}},
    allow_errors=False,
    on_cell_start=lambda cell, cell_index: (
        print(
            f"Executing notebook cell {cell_index+1}/{len(notebook.cells)}", flush=True
        )
        if cell.cell_type == "code"
        else None
    ),
)
print("Starting fresh notebook kernel:", path.name, flush=True)
try:
    client.execute()
finally:
    # Stable line endings keep notebook hashes consistent in Git on every OS.
    path.write_text(nbformat.writes(notebook), encoding="utf-8", newline="\n")
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
errors = [
    output
    for cell in notebook.cells
    if cell.cell_type == "code"
    for output in cell.get("outputs", [])
    if output.output_type == "error"
]
if errors:
    raise RuntimeError("Notebook contains execution errors.")
print(
    "Fresh-kernel execution passed:",
    sum(c.cell_type == "code" for c in notebook.cells),
    "code cells.",
)

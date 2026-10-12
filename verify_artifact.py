"""Read-only, dependency-free check of the packaged source, data and figures."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    items = manifest["files"]
    for item in items:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert sha(path) == item["release_sha256"], item["path"]
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=item["path"])
    main_values = list(csv.DictReader((ROOT / "data" / "plotted_values.csv").open(encoding="utf-8")))
    supplement_values = list(csv.DictReader((ROOT / "data" / "supplement_plotted_values.csv").open(encoding="utf-8")))
    assert main_values and supplement_values
    assert {"figure", "panel", "value", "source"} <= set(main_values[0])
    assert {"figure", "role", "opponent", "value", "status"} <= set(supplement_values[0])
    figures = sorted((ROOT / "figures").glob("*.pdf"))
    assert len(figures) == 9
    print(f"Verified {len(items)} packaged hashes and Python syntax; "
          f"{len(main_values)} main and {len(supplement_values)} supplementary plotted rows; "
          f"{len(figures)} final figure PDFs.")
    print("This is an artifact-integrity check, not a model or scientific rerun.")


if __name__ == "__main__":
    main()

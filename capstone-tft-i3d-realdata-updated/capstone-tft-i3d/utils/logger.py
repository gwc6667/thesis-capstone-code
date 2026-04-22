from __future__ import annotations
from pathlib import Path
import csv
import json


def append_result_row(csv_path, row: dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    existing_fields = []
    if csv_path.exists():
        with csv_path.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            rows = list(reader)

    fieldnames = []
    for key in [*existing_fields, *row.keys()]:
        if key not in fieldnames:
            fieldnames.append(key)

    rows.append(row)

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in rows:
            writer.writerow({field: existing_row.get(field, '') for field in fieldnames})


def write_json(path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

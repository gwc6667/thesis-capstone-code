from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

metrics = pd.read_csv(DATA / "final_metrics.csv")
losses = json.loads((DATA / "train_losses.json").read_text(encoding="utf-8"))
config = json.loads((DATA / "reconstruction_config.json").read_text(encoding="utf-8"))

print("=== Final Metrics (Table 4.1) ===")
print(metrics.to_string(index=False))
print()
print("=== Train Losses (Figure 4.8) ===")
for model, vals in losses.items():
    print(f"{model}: {vals}")
print()
print("=== Reconstruction Config (Figures 4.5-4.7) ===")
print(json.dumps(config, indent=2))

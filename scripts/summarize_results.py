#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Summarize one or more fold_metrics.csv files.")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--out", default="summary_all.csv")
    args = parser.parse_args()
    rows = []
    for f in args.files:
        path = Path(f)
        df = pd.read_csv(path)
        row = {"file": str(path)}
        for col in df.columns:
            if col == "fold":
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals):
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

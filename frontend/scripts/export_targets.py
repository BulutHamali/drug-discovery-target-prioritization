#!/usr/bin/env python3
"""Export the ML out-of-sample ranking into a frontend-safe JSON artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"symbol", "score", "label", "rank", "fold_idx"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("../ml/cache/oos_predictions.parquet"))
    parser.add_argument("--output", type=Path, default=Path("public/data/targets.json"))
    parser.add_argument("--feature-set", default="biology_only")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Prediction file not found: {args.input}\n"
            "Run: python3 ml/train_eval.py --feature-set biology_only"
        )

    frame = pd.read_parquet(args.input)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise SystemExit(f"Prediction file is missing required columns: {', '.join(sorted(missing))}")

    frame = frame.sort_values("rank").copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["score"] = frame["score"].astype(float).round(8)
    frame["label"] = frame["label"].astype(int)
    frame["rank"] = frame["rank"].astype(int)
    frame["fold_idx"] = frame["fold_idx"].astype(int)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "ml/cache/oos_predictions.parquet",
        "feature_set": args.feature_set,
        "rows": len(frame),
        "targets": frame[["symbol", "score", "label", "rank", "fold_idx"]].to_dict("records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output} ({len(frame):,} targets, feature set: {args.feature_set})")


if __name__ == "__main__":
    main()

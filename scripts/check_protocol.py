#!/usr/bin/env python
from __future__ import annotations

import argparse, json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check split files for unintended overlap/leakage.")
    parser.add_argument("result_dir")
    parser.add_argument(
        "--strict-cdan",
        action="store_true",
        help="Also forbid CDAN target/test overlap. Do not use this for the original GitHub-style transductive CDAN protocol.",
    )
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    errors = []
    transductive = []
    for split_file in result_dir.glob("fold_*/split_indices.json"):
        obj = json.loads(split_file.read_text(encoding="utf-8"))
        test = set(obj.get("test_indices", []))
        for key in ["source_train_indices", "validation_indices"]:
            overlap = test.intersection(obj.get(key, []))
            if overlap:
                errors.append((str(split_file), key, sorted(overlap)[:10]))
        cdan_overlap = test.intersection(obj.get("cdan_target_indices", []))
        cdan_source = obj.get("cdan_target_source", "unknown")
        if cdan_overlap:
            if args.strict_cdan or cdan_source != "test":
                errors.append((str(split_file), "cdan_target_indices", sorted(cdan_overlap)[:10]))
            else:
                transductive.append(str(split_file))
    if errors:
        print("Found unintended split overlap:")
        for e in errors:
            print(e)
        raise SystemExit(1)
    if transductive:
        print(f"OK: source/validation do not overlap test. CDAN target uses test fold in {len(transductive)} folds (original GitHub-style transductive protocol).")
    else:
        print(f"OK: no test overlap found in {result_dir}")


if __name__ == "__main__":
    main()

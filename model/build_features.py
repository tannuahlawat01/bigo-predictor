"""
build_features.py
Runs the AST feature extractor over every example in data/training_data.csv
and produces a feature matrix (one row per code sample, one column per
feature + the label) ready for model training.
"""
import csv
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # allow importing from features/
from features.ast_extractor import extract_features
INPUT_CSV = Path(__file__).parent.parent / "data" / "training_data.csv"
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "feature_matrix.csv"

FEATURE_COLUMNS = [
    "max_loop_depth",
    "num_loops",
    "has_recursion",
    "recursion_branch_factor",
    "has_sort_call",
    "has_hash_structure",
    "has_input_dependent_loop_bound",
    "has_halving_pattern",
]

def main():
    rows_out = []
    skipped = 0
    total = 0

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            features = extract_features(row["code"])

            if features is None:
                skipped += 1
                continue

            out_row = {col: features[col] for col in FEATURE_COLUMNS}
            out_row["complexity"] = row["complexity"]
            rows_out.append(out_row)

    print(f"Processed {total} examples")
    print(f"Skipped (failed to parse): {skipped}")
    print(f"Successfully extracted: {len(rows_out)}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS + ["complexity"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nSaved feature matrix to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
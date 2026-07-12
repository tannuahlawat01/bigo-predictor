"""
collect_data.py
Extracts Python-only, labeled (code, complexity) pairs from the CodeComplex
dataset (train + test JSONL files) and saves them to a clean CSV for
feature extraction and model training.
Source dataset: https://github.com/sybaik1/CodeComplex
"""
import json
import re
import csv
from pathlib import Path
# ---- Paths ----
CODECOMPLEX_DIR = Path(__file__).parent / "CodeComplex" / "LLM-qlora" / "codecomplex-simple"
OUTPUT_CSV = Path(__file__).parent / "training_data.csv"

# ---- Java detection keywords (used to EXCLUDE non-Python examples) ----
JAVA_MARKERS = ["import java", "public class", "public static void main", "System.out"]

# ---- Label normalization: dataset's raw labels -> our 6 target classes ----
LABEL_MAP = {
    "constant": "O(1)",
    "logn": "O(log n)",
    "linear": "O(n)",
    "nlogn": "O(n log n)",
    "quadratic": "O(n^2)",
    "cubic": "O(n^2)",       # merged into quadratic bucket (rare class, keeps scope to 6 classes)
    "np": "O(2^n)",          # NP-hard / exponential bucket
    "exponential": "O(2^n)",
}

def extract_code(user_content: str) -> str:
    """
    The user message wraps the code between '----...----' divider lines,
    followed by trailing instructions. Extract just the code block.
    """
    parts = user_content.split("----------------------------------------")
    # parts[0] is empty (message starts with the divider), parts[1] is the code
    if len(parts) < 2:
        return ""
    return parts[1].strip()

def extract_label(assistant_content: str) -> str:
    """
    Assistant message looks like: 'complexity: logn'
    """
    match = re.search(r"complexity:\s*(\w+)", assistant_content.strip())
    if not match:
        return ""
    raw_label = match.group(1).lower()
    return LABEL_MAP.get(raw_label, "")

def is_python_code(code: str) -> bool:
    if any(marker in code for marker in JAVA_MARKERS):
        return False
    # crude but effective: Python code has 'def ' and no semicolon-heavy Java style
    return "def " in code or "print(" in code

def process_file(filepath: Path, rows: list) -> None:
    if not filepath.exists():
        print(f"  (skipping, not found: {filepath.name})")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        messages = obj.get("messages", [])
        if len(messages) < 3:
            continue

        user_content = messages[1]["content"]
        assistant_content = messages[-1]["content"]

        code = extract_code(user_content)
        if not code or not is_python_code(code):
            continue

        label = extract_label(assistant_content)
        if not label:
            continue

        rows.append({"code": code, "complexity": label})

def main():
    rows = []
    print("Processing CodeComplex dataset files...")
    process_file(CODECOMPLEX_DIR / "train_dataset.json", rows)
    process_file(CODECOMPLEX_DIR / "test_dataset.json", rows)

    print(f"\nTotal Python examples extracted: {len(rows)}")

    # Class distribution check
    from collections import Counter
    dist = Counter(r["complexity"] for r in rows)
    print("\nClass distribution:")
    for label, count in sorted(dist.items()):
        print(f"  {label:12s}: {count}")

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "complexity"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
"""
predict.py

Loads the trained model and label encoder, and provides a function to
predict the time complexity of a given Python code snippet.

This is the module the FastAPI endpoint (api/main.py) will import.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from features.ast_extractor import extract_features

MODEL_DIR = Path(__file__).parent
FEATURE_COLUMNS = [
    "max_loop_depth",
    "num_loops",
    "has_recursion",
    "recursion_branch_factor",
    "has_sort_call",
    "has_hash_structure",
    "has_input_dependent_loop_bound",
]

def generate_explanation(features: dict, complexity: str) -> str:
    """
    Produces a short, human-readable reason for the prediction based on
    which extracted features are most relevant to the predicted class.
    Rule-based, not learned — this mirrors the model's own feature
    importances (loop depth and count matter most, then sort/recursion).
    """
    reasons = []

    depth = features["max_loop_depth"]
    num_loops = features["num_loops"]
    has_recursion = features["has_recursion"]
    branch_factor = features["recursion_branch_factor"]
    has_sort = features["has_sort_call"]
    has_hash = features["has_hash_structure"]
    input_dependent = features["has_input_dependent_loop_bound"]

    if has_recursion and branch_factor >= 2:
        reasons.append(f"recursive calls with a branching factor of {branch_factor} per invocation")
    elif has_recursion:
        reasons.append("a single recursive call chain")

    if depth >= 2:
        reasons.append(f"{depth} levels of nested loops")
    elif depth == 1 and input_dependent:
        reasons.append("a single loop scaling with input size")
    elif depth == 1:
        reasons.append("a single loop with a fixed bound")

    if has_sort:
        reasons.append("a sort operation")

    if has_hash and depth <= 1:
        reasons.append("hash-based lookups (dict/set), which can keep operations closer to linear")

    if not reasons:
        reasons.append("no loops or recursion detected")

    reason_text = ", ".join(reasons)

    # A note for the model's known weak spot, so the explanation stays honest
    caveat = ""
    if complexity in ("O(2^n)", "O(n^2)"):
        caveat = (
            " Note: O(2ⁿ) and O(n²) can look structurally similar to this model "
            "(both often involve nested loop-like patterns), so this is worth a manual check."
        )

    return f"Predicted {complexity} primarily due to: {reason_text}.{caveat}"

# Load once at import time, not on every prediction call (avoids reloading
# the model from disk on every single request — matters once this is
# wrapped in an API that might get many requests)
_model = None
_label_encoder = None


def _load_model():
    global _model, _label_encoder
    if _model is None:
        _model = joblib.load(MODEL_DIR / "complexity_model.pkl")
        _label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    return _model, _label_encoder


def predict_complexity(code: str) -> dict:
    """
    Given Python source code as a string, returns a dict with:
      - complexity: predicted class (e.g. "O(n^2)")
      - confidence: probability of the predicted class
      - all_probabilities: dict of every class -> probability
      - features: the extracted feature values (for explainability)
      - error: present only if code failed to parse
    """
    features = extract_features(code)

    if features is None:
        return {
            "error": "Code could not be parsed. Please check for syntax errors.",
            "complexity": None,
            "confidence": None,
        }

    model, label_encoder = _load_model()

    # Build a single-row DataFrame matching training feature order
    row = {col: float(features[col]) for col in FEATURE_COLUMNS}
    X = pd.DataFrame([row])

    probabilities = model.predict_proba(X)[0]
    predicted_idx = probabilities.argmax()
    predicted_class = label_encoder.classes_[predicted_idx]

    all_probs = {
        label_encoder.classes_[i]: round(float(p), 3)
        for i, p in enumerate(probabilities)
    }

    explanation = generate_explanation(features, predicted_class)
    return {
        "complexity": predicted_class,
        "confidence": round(float(probabilities[predicted_idx]), 3),
        "all_probabilities": all_probs,
        "features": features,
        "explanation": explanation,
        "error": None,
    }


if __name__ == "__main__":
    # Quick manual test
    test_code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""
    result = predict_complexity(test_code)
    print("Prediction result:")
    for key, val in result.items():
        print(f"  {key}: {val}")
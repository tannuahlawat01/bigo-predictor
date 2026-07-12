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

    return {
        "complexity": predicted_class,
        "confidence": round(float(probabilities[predicted_idx]), 3),
        "all_probabilities": all_probs,
        "features": features,
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
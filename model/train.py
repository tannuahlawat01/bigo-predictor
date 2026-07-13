"""
train.py
Trains and compares two classifiers (Random Forest, XGBoost) on the
AST-derived feature matrix, then saves the better-performing model.
"""
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from xgboost import XGBClassifier
DATA_PATH = Path(__file__).parent.parent / "data" / "feature_matrix.csv"
MODEL_DIR = Path(__file__).parent
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

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Convert boolean-string columns to actual 0/1 ints
    for col in ["has_recursion", "has_sort_call", "has_hash_structure", "has_input_dependent_loop_bound"]:
        df[col] = df[col].astype(str).map({"True": 1, "False": 0}).fillna(df[col])

    X = df[FEATURE_COLUMNS].astype(float)
    y = df["complexity"]
    return X, y

def evaluate(name, model, X_test, y_test, label_encoder):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{'=' * 50}")
    print(f"{name} — Accuracy: {acc:.3f}")
    print(f"{'=' * 50}")
    print(classification_report(
        y_test, preds,
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    return acc

def main():
    print("Loading feature matrix...")
    X, y = load_data()
    print(f"Total examples: {len(X)}")
    print(f"Features: {FEATURE_COLUMNS}")

    # Encode string labels (e.g. "O(n^2)") to integers for XGBoost
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Classes: {list(label_encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # --- Model 1: Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = evaluate("Random Forest", rf, X_test, y_test, label_encoder)

    # --- Model 2: XGBoost ---
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="mlogloss"
    )
    xgb.fit(X_train, y_train)
    xgb_acc = evaluate("XGBoost", xgb, X_test, y_test, label_encoder)

    # --- Confusion matrix for the better model ---
    best_name, best_model, best_acc = (
        ("Random Forest", rf, rf_acc) if rf_acc >= xgb_acc else ("XGBoost", xgb, xgb_acc)
    )
    print(f"\nBest model: {best_name} ({best_acc:.3f} accuracy)")

    preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    print(f"\nConfusion matrix ({best_name}):")
    print(f"{'':12s}" + "".join(f"{c:>10s}" for c in label_encoder.classes_))
    for i, row in enumerate(cm):
        print(f"{label_encoder.classes_[i]:12s}" + "".join(f"{v:>10d}" for v in row))

    # --- Feature importance (interview gold: "which features mattered most") ---
    if hasattr(best_model, "feature_importances_"):
        print(f"\nFeature importances ({best_name}):")
        importances = sorted(
            zip(FEATURE_COLUMNS, best_model.feature_importances_),
            key=lambda x: -x[1]
        )
        for feat, imp in importances:
            print(f"  {feat:35s} {imp:.3f}")

    # --- Save the best model + label encoder ---
    joblib.dump(best_model, MODEL_DIR / "complexity_model.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    print(f"\nSaved best model to {MODEL_DIR / 'complexity_model.pkl'}")

if __name__ == "__main__":
    main()
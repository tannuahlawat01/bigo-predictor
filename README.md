# Big-O — Time Complexity Predictor

Predicts the asymptotic time complexity of a Python function — O(1), O(log n), O(n), O(n log n), O(n²), or O(2ⁿ) — directly from source code structure, without executing it. Uses AST-based static analysis and a Random Forest classifier.

**Live demo:** [bigo-predictor.streamlit.app](https://bigo-predictor-tscxphn7nxcd3uw7c6nkq7.streamlit.app/)

**API:** [bigo-predictor.onrender.com](https://bigo-predictor.onrender.com) (`/health`, `/predict`)

## Why this project

Predicting time complexity is normally either manually estimated by the developer or checked by brute-force running code against large inputs. Static prediction from source structure alone is a genuinely underexplored problem — most existing work is academic rather than production tooling. This project treats it as a real, testable ML problem: extract structural features from code, train a classifier, and honestly report where it works and where it doesn't.

## How it works

```
Python code → AST parser → 8 structural features → Random Forest → predicted complexity + confidence + explanation
```

**Features extracted** (via Python's built-in `ast` module):
- Max loop nesting depth, total loop count
- Recursion presence and branching factor (calls per invocation)
- Sort call presence, hash structure (dict/set) presence
- Input-dependent loop bounds (`range(len(x))`, direct iteration, or `while` loops with a `len()` check)
- Halving pattern detection (`mid = (lo + hi) // 2`, binary-search-style)

## Dataset

Sourced from the **CodeComplex** academic benchmark (Jeon et al.) — competitive programming submissions expert-annotated for time complexity. From the full dataset (~9,800 Java + Python examples), this project extracts **4,844 Python-only labeled examples** across 6 complexity classes, reasonably balanced (521–1,234 examples per class).

Source: https://github.com/sybaik1/CodeComplex

To regenerate `data/feature_matrix.csv` from scratch:
```bash
cd data
git clone https://github.com/sybaik1/CodeComplex.git
python collect_data.py
cd ..
python model/build_features.py
```

## Model performance

**Random Forest, 59.8% accuracy** on a 6-class problem (random baseline: 16.7%), using only 8 structural features — no code embeddings or language models.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| O(1) | 0.64 | 0.61 | 0.62 |
| O(2ⁿ) | 0.66 | 0.18 | 0.29 |
| O(log n) | 0.68 | 0.43 | 0.53 |
| O(n log n) | 0.70 | 0.72 | 0.71 |
| O(n) | 0.41 | 0.48 | 0.44 |
| O(n²) | 0.57 | 0.80 | 0.67 |

**Feature importances:** loop depth and count dominate (0.25 each), followed by sort-call presence (0.20) and input-dependent loop bounds (0.12). Recursion presence alone barely matters (0.03) — branching factor and loop structure carry the real signal.

## Known limitations

- **O(2ⁿ) vs O(n²) confusion** — the model's primary error source (recall: 0.18 for O(2ⁿ)). Both classes exhibit similar loop-nesting depth in real competitive programming code (e.g., bitmask DP or backtracking often combines loops with recursion), which these 8 features can't fully disambiguate. Confirmed via the confusion matrix, not just assumed.
- **Cross-function calls aren't resolved** — the extractor only analyzes the function it's given. A `merge_sort` calling an undefined `merge()` function was tested and misclassified as O(log n) with 95% confidence, because the extractor has no visibility into what the called function does. This is a genuine scope boundary of single-function static analysis, not a bug.
- **`while`-loop bound detection is intentionally conservative** — only explicit `len()` calls in the condition are trusted. An earlier, broader heuristic (flagging any two-variable comparison) was tested and found to measurably *regress* accuracy (59.2% → 55.9%) by diluting the feature's signal; it was reverted in favor of this narrower, higher-precision rule, which improved O(log n) recall from 0.33 to 0.57.
- **Python-only** — AST parsing is language-specific; Java or C++ support would require separate extractors (see Future Work).

## Future work

- **Language support**: extend to Java via `javalang` (pure-Python, no system dependencies, AST maps closely to the existing feature set). C++ would require `clang`/`libclang` bindings or `tree-sitter` — meaningfully more infrastructure.
- **Cross-function resolution**: recursively extract and merge features from user-defined functions called within the analyzed snippet, addressing the merge_sort limitation above.
- **Loop bound reasoning**: a dedicated feature for detecting "shrinking search space" patterns beyond the current halving-pattern check, to further improve O(log n) vs O(n) disambiguation.
- **Code embeddings**: compare this feature-engineered approach against a pretrained code embedding model (e.g., CodeBERT), mirroring the baseline-vs-advanced-model comparison methodology used in the NLP Resume Screener project (TF-IDF vs. SBERT).

## Project structure

```
bigo-predictor/
  data/
    collect_data.py       # extracts + cleans Python examples from CodeComplex
    training_data.csv      # 4,846 labeled (code, complexity) pairs
    feature_matrix.csv     # extracted features + labels, ready for training
  features/
    ast_extractor.py       # AST-based feature engineering (the core logic)
  model/
    build_features.py      # runs the extractor over the full dataset
    train.py                # trains Random Forest / XGBoost, saves the best model
    predict.py               # loads trained model, runs inference
    complexity_model.pkl     # trained model (committed for deployment)
    label_encoder.pkl        # label encoder (committed for deployment)
  api/
    main.py                  # FastAPI inference endpoint
  app/
    streamlit_app.py         # demo UI
  tests/
    test_extractor.py        # unit tests for feature extraction
  Dockerfile
  requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt

# Train the model (only needed if model/*.pkl doesn't exist)
python model/train.py

# Terminal 1 — backend
python api/main.py

# Terminal 2 — frontend
streamlit run app/streamlit_app.py
```

## Deployment

- **Backend**: FastAPI on Render (free tier). Model files are committed to the repo (not gitignored) since Render's free tier uses an ephemeral filesystem — training on every cold start proved too slow and unreliable, so the model is trained locally and shipped as an artifact instead.
- **Frontend**: Streamlit Community Cloud, calling the Render backend over HTTPS.
- Render's free tier sleeps after 15 minutes of inactivity; the first request after idle time takes ~30-50 seconds to wake the backend.

## Tech stack

Python · `ast` (stdlib) · scikit-learn · XGBoost · pandas · FastAPI · Streamlit · pytest

## Testing

```bash
pytest tests/test_extractor.py -v
```
# Big-O — Time Complexity Predictor

Predicts the asymptotic time complexity (O(1), O(log n), O(n), O(n log n),
O(n²), O(2ⁿ)) of a Python function directly from source code — without
executing it — using AST-based feature extraction and a tree-based classifier.

## Why this project

Predicting time complexity is normally either manually estimated or checked
by brute-force running code against large inputs. Static prediction from
source code structure is a genuinely underexplored problem — most existing
work is academic (see Dataset section) rather than production tooling.

## Dataset

Training data is sourced from the **CodeComplex** academic benchmark
(Jeon et al.) — competitive programming submissions expert-annotated for
time complexity. From the full dataset (Java + Python, ~9,800 examples),
this project extracts and cleans **4,846 Python-only labeled examples**
across 6 complexity classes.

Source: https://github.com/sybaik1/CodeComplex

To regenerate `data/training_data.csv` from scratch:
```bash
cd data
git clone https://github.com/sybaik1/CodeComplex.git
python collect_data.py
```

## Project structure

```
bigo-predictor/
  data/
    collect_data.py       # extracts + cleans Python examples from CodeComplex
    training_data.csv      # 4,846 labeled (code, complexity) pairs
  features/
    ast_extractor.py       # AST-based feature engineering (the core logic)
  model/
    train.py               # trains Random Forest / XGBoost classifiers
    predict.py              # loads trained model, runs inference
  api/
    main.py                 # FastAPI inference endpoint
  app/
    streamlit_app.py        # demo UI
  tests/
    test_extractor.py       # unit tests for feature extraction
```

## Status

- [x] Dataset collection and cleaning
- [ ] AST feature extractor
- [ ] Model training
- [ ] FastAPI inference endpoint
- [ ] Streamlit demo UI
"""
streamlit_app.py

Demo UI for the Big-O predictor. Paste Python code, get a predicted
time complexity with confidence scores and feature breakdown.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Big-O Predictor", page_icon="⏱️", layout="wide")

# ---- Sidebar: project info ----
with st.sidebar:
    st.header("About")
    st.markdown(
        "Predicts Python code's time complexity from its **structure** "
        "(loops, recursion, sorts) — without executing it."
    )
    st.divider()
    st.subheader("Model")
    st.markdown(
        "- **Random Forest**, 7 AST features\n"
        "- Trained on **4,844** Python examples\n"
        "- Source: CodeComplex academic benchmark\n"
        "- **59.2%** test accuracy (6 classes, 16.7% baseline)"
    )
    st.divider()
    st.subheader("Known limitation")
    st.caption(
        "O(2ⁿ) and O(n²) are sometimes confused — both show similar "
        "loop-nesting patterns in real competitive programming code."
    )

# ---- Main content ----
st.title("⏱️ Big-O Time Complexity Predictor")
st.markdown(
    "Paste a Python function below and get its predicted time complexity, "
    "explained through the structural features that drove the prediction."
)

EXAMPLES = {
    "Bubble Sort — O(n²)": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr""",
    "Binary Search — O(log n)": """def binary_search(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return -1""",
    "Naive Fibonacci — O(2ⁿ)": """def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)""",
    "Linear Search — O(n)": """def linear_search(arr, target):
    for x in arr:
        if x == target:
            return True
    return False""",
}

col_select, col_blank = st.columns([2, 3])
with col_select:
    example_choice = st.selectbox("Try an example", ["— Write your own —"] + list(EXAMPLES.keys()))

default_code = EXAMPLES.get(example_choice, "")

code = st.text_area("Python code", value=default_code, height=250, placeholder="def my_function(arr):\n    ...")

predict_clicked = st.button("Predict Complexity", type="primary", use_container_width=False)

COMPLEXITY_COLORS = {
    "O(1)": "🟢",
    "O(log n)": "🟢",
    "O(n)": "🟡",
    "O(n log n)": "🟡",
    "O(n^2)": "🟠",
    "O(2^n)": "🔴",
}

if predict_clicked:
    if not code.strip():
        st.warning("Please paste some code first.")
    else:
        with st.spinner("Analyzing code structure..."):
            try:
                response = requests.post(API_URL, json={"code": code}, timeout=10)

                if response.status_code == 400:
                    st.error(f"⚠️ {response.json().get('detail', 'Could not parse code.')}")
                elif response.status_code != 200:
                    st.error("Server error. Is the FastAPI server running on port 8000?")
                else:
                    result = response.json()
                    complexity = result["complexity"]
                    confidence = result["confidence"]
                    icon = COMPLEXITY_COLORS.get(complexity, "⚪")

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"### {icon} Predicted complexity: `{complexity}`")
                        st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")

                    with col2:
                        if confidence < 0.4:
                            st.caption("⚠️ Low confidence — this code sits near a class boundary.")

                    st.subheader("Class probabilities")
                    probs = result["all_probabilities"]
                    sorted_probs = dict(sorted(probs.items(), key=lambda x: -x[1]))
                    st.bar_chart(sorted_probs)

                    st.subheader("What the model saw")
                    features = result["features"]
                    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
                    fcol1.metric("Max loop depth", features["max_loop_depth"])
                    fcol2.metric("Total loops", features["num_loops"])
                    fcol3.metric("Recursion?", "Yes" if features["has_recursion"] else "No")
                    fcol4.metric("Branch factor", features["recursion_branch_factor"])

                    with st.expander("Raw feature values"):
                        st.json(features)

            except requests.exceptions.ConnectionError:
                st.error(
                    "🔌 Could not connect to the API. Make sure it's running: "
                    "`python api/main.py`"
                )

st.divider()
st.caption(
    "Built with AST-based static analysis — no LLMs, no code execution. "
    "See the README for methodology, dataset, and known limitations."
)
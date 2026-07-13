"""
Big-O Complexity Predictor
Modern Streamlit UI
"""

import requests
import streamlit as st
from streamlit_ace import st_ace

API_URL = "https://bigo-predictor.onrender.com/health"

st.set_page_config(
    page_title="Big-O Complexity Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:1rem;
    max-width:1300px;
}

[data-testid="stSidebar"]{
    background:#121826;
}

.hero{
    padding:22px;
    border-radius:16px;
    background:linear-gradient(135deg,#1f2937,#111827);
    border:1px solid #30363d;
}

.hero h1{
    color:white;
    margin-bottom:8px;
}

.hero p{
    color:#d1d5db;
    font-size:17px;
}

.metric-card{
    background:#171f2c;
    border-radius:15px;
    padding:18px;
    text-align:center;
    border:1px solid #2b3442;
}

.metric-title{
    color:#9ca3af;
    font-size:14px;
}

.metric-value{
    color:white;
    font-size:28px;
    font-weight:700;
}

.result-card{
    background:#161b22;
    border-radius:18px;
    padding:22px;
    border:1px solid #30363d;
}

.feature-card{
    background:#1b2230;
    border-radius:14px;
    padding:14px;
    text-align:center;
    border:1px solid #2e3748;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
}

.stButton>button{
    border-radius:10px;
    height:45px;
    font-weight:600;
}

.stFileUploader{
    border-radius:12px;
}

[data-testid="stMetric"]{
    background:#171f2c;
    padding:12px;
    border-radius:12px;
    border:1px solid #2b3442;
}

</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------

with st.sidebar:

    st.title("🚀 Big-O Predictor")

    st.write(
        "Predict the **time complexity** of Python programs using "
        "AST-based Machine Learning."
    )

    st.divider()

    st.subheader("Model")

    st.success("Random Forest")

    st.write("• 4,844 Python programs")
    st.write("• 7 AST features")
    st.write("• 6 complexity classes")
    st.write("• Accuracy: **59.8%**")

    st.divider()

    st.subheader("Supported Classes")

    st.code("""
O(1)
O(log n)
O(n)
O(n log n)
O(n²)
O(2ⁿ)
""")

    st.divider()

    st.info(
        "The model predicts complexity without executing the code. "
        "Predictions are based only on structural AST features."
    )

# ---------- Hero Section ----------

st.markdown("""
<div class="hero">
<h1>🚀 Big-O Time Complexity Predictor</h1>

<p>
Predict the asymptotic time complexity of Python code using
Machine Learning and Abstract Syntax Tree (AST) analysis.
No execution. No LLMs. Pure static analysis.
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# ---------- Top Statistics ----------

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown("""
<div class="metric-card">
<div class="metric-title">Dataset</div>
<div class="metric-value">4,844</div>
</div>
""", unsafe_allow_html=True)

with c2:
    st.markdown("""
<div class="metric-card">
<div class="metric-title">Accuracy</div>
<div class="metric-value">59.2%</div>
</div>
""", unsafe_allow_html=True)

with c3:
    st.markdown("""
<div class="metric-card">
<div class="metric-title">Model</div>
<div class="metric-value">Random Forest</div>
</div>
""", unsafe_allow_html=True)

with c4:
    st.markdown("""
<div class="metric-card">
<div class="metric-title">Features</div>
<div class="metric-value">7 AST</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------- Example Programs ----------

EXAMPLES = {
    "Write Your Own":"",
    "Linear Search":"""def linear_search(arr,target):
    for x in arr:
        if x==target:
            return True
    return False""",

    "Binary Search":"""def binary_search(arr,target):
    lo,hi=0,len(arr)-1
    while lo<=hi:
        mid=(lo+hi)//2
        if arr[mid]==target:
            return mid
        elif arr[mid]<target:
            lo=mid+1
        else:
            hi=mid-1
    return -1""",

    "Bubble Sort":"""def bubble_sort(arr):
    n=len(arr)
    for i in range(n):
        for j in range(n-i-1):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1]=arr[j+1],arr[j]
    return arr""",

    "Merge Sort":"""def merge_sort(arr):
    if len(arr)<=1:
        return arr
    mid=len(arr)//2
    left=merge_sort(arr[:mid])
    right=merge_sort(arr[mid:])
    return merge(left,right)""",

    "Naive Fibonacci":"""def fib(n):
    if n<=1:
        return n
    return fib(n-1)+fib(n-2)"""
}

st.subheader("📝 Python Code")

col1,col2=st.columns([3,1])

with col2:
    sample=st.selectbox(
        "Load Sample",
        list(EXAMPLES.keys())
    )

default_code=EXAMPLES[sample]
# ---------- File Upload ----------

uploaded_file = st.file_uploader(
    "Upload a Python file",
    type=["py"],
    help="Upload a .py file or use the editor below."
)

if uploaded_file is not None:
    default_code = uploaded_file.read().decode("utf-8")

# ---------- Code Editor ----------
editor_key = f"ace_editor_{sample}_{uploaded_file.name if uploaded_file else 'default'}"
editor_col, info_col = st.columns([3,1], gap="large")

with editor_col:

    st.write("### Code Editor")

    code = st_ace(
        value=default_code,
        language="python",
        theme="monokai",
        key=editor_key,
        height=420,
        font_size=15,
        tab_size=4,
        show_gutter=True,
        show_print_margin=False,
        wrap=True,
        auto_update=True
    )

with info_col:

    st.write("### Quick Stats")

    total_lines = len(code.splitlines()) if code else 0
    total_chars = len(code)

    st.metric("Lines", total_lines)
    st.metric("Characters", total_chars)

    st.write("")

    st.info(
        "The model extracts structural AST features such as loop depth, "
        "recursion and branching before making a prediction."
    )

    st.write("")

    st.success(
        """
✔ No execution

✔ Static analysis

✔ Machine Learning

✔ Fast prediction
"""
    )

st.write("")

# ---------- Action Buttons ----------

btn1, btn2, btn3 = st.columns([1,1,5])

with btn1:
    predict_clicked = st.button(
        "🚀 Predict",
        use_container_width=True,
        type="primary"
    )

with btn2:
    clear_clicked = st.button(
        "🗑 Clear",
        use_container_width=True
    )

if clear_clicked:
    st.rerun()

st.divider()

# ---------- Model Pipeline ----------

st.subheader("How Prediction Works")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
<div class="feature-card">
<h3>📄</h3>
<b>Input Code</b><br>
Python Function
</div>
""", unsafe_allow_html=True)

with c2:
    st.markdown("""
<div class="feature-card">
<h3>🌳</h3>
<b>AST Parser</b><br>
Feature Extraction
</div>
""", unsafe_allow_html=True)

with c3:
    st.markdown("""
<div class="feature-card">
<h3>🤖</h3>
<b>Random Forest</b><br>
Classification
</div>
""", unsafe_allow_html=True)

with c4:
    st.markdown("""
<div class="feature-card">
<h3>📈</h3>
<b>Prediction</b><br>
Complexity Class
</div>
""", unsafe_allow_html=True)

st.write("")
# ---------- Prediction ----------

COMPLEXITY_COLORS = {
    "O(1)": "🟢",
    "O(log n)": "🟢",
    "O(n)": "🟡",
    "O(n log n)": "🟠",
    "O(n²)": "🔴",
    "O(n^2)": "🔴",
    "O(2ⁿ)": "🔴",
    "O(2^n)": "🔴"
}

if predict_clicked:

    if not code.strip():
        st.warning("Please enter some Python code.")
        st.stop()

    with st.spinner("Analyzing code... (first request may take up to a minute if the server was idle)"):

        try:

            response = requests.post(API_URL, json={"code": code}, timeout=60)

            if response.status_code == 400:
                st.error(response.json()["detail"])
                st.stop()

            if response.status_code != 200:
                st.error("Unable to connect to the API.")
                st.stop()

            result = response.json()

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI server.")
            st.info("Run:\n\npython api/main.py")
            st.stop()

    complexity = result["complexity"]
    confidence = result["confidence"]
    probabilities = result["all_probabilities"]
    features = result["features"]

    icon = COMPLEXITY_COLORS.get(complexity, "⚪")

    st.markdown("---")
    st.subheader("Prediction Result")

    left, right = st.columns([2, 1], gap="large")

    with left:

        st.markdown(f"""
        <div class="result-card">
            <h2>{icon} {complexity}</h2>
            <p style="font-size:18px;">
                Predicted Time Complexity
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.progress(confidence)
        st.write(f"**Confidence:** {confidence*100:.2f}%")
        st.write("")
        st.markdown(f"💡 **Why this prediction:** {result['explanation']}")
        if confidence >= 0.80:
            st.success("Very High Confidence")
        elif confidence >= 0.60:
            st.info("High Confidence")
        elif confidence >= 0.40:
            st.warning("Moderate Confidence")
        else:
            st.error("Low Confidence")
    with right:
        st.metric("Prediction", complexity)
        st.metric("Confidence", f"{confidence*100:.1f}%")
        st.metric("Model", "Random Forest")
        st.metric("Classes", "6")
    st.write("")
    tab1, tab2 = st.tabs(["📊 Probabilities", "🌳 AST Features"])
    with tab1:
        st.write("### Class Probabilities")
        ordered_probs = dict(
            sorted(
                probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )
        st.bar_chart(ordered_probs)
        st.write("")
        for label, prob in ordered_probs.items():
            st.write(f"**{label}**")
            st.progress(prob)
            st.caption(f"{prob*100:.2f}%")
    with tab2:
        st.write("### Extracted AST Features")
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Maximum Loop Depth",
                features["max_loop_depth"]
            )
            st.metric(
                "Total Loops",
                features["num_loops"]
            )
        with c2:
            st.metric(
                "Recursion",
                "Yes" if features["has_recursion"] else "No"
            )
            st.metric(
                "Branch Factor",
                features["recursion_branch_factor"]
            )
        st.write("")
        with st.expander("Raw Feature Vector"):
            st.json(features)
    st.success("Prediction completed successfully.")
# ---------- Project Information ----------
st.divider()
st.header("📖 About the Project")
tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Model",
    "Limitations"
])
with tab1:

    st.markdown("""
### Big-O Time Complexity Predictor

This application predicts the **time complexity** of Python programs
using **Machine Learning** and **Abstract Syntax Tree (AST)** analysis.

Instead of executing code, the program parses the AST, extracts
structural features, and predicts the most likely complexity class.

The complete prediction pipeline is:

Python Code → AST Parser → Feature Extraction → Random Forest → Complexity Prediction
""")

    c1, c2, c3 = st.columns(3)

    c1.metric("Dataset Size", "4,844")
    c2.metric("Complexity Classes", "6")
    c3.metric("Model Accuracy", "59.2%")

with tab2:

    st.markdown("""
### Machine Learning Model

**Algorithm**
- Random Forest Classifier

**Training Dataset**
- CodeComplex Benchmark

**Features Used**
- Maximum loop depth
- Total loops
- Recursive calls
- Recursion branch factor
- AST structural properties

The model predicts complexity without executing the code, making inference very fast.
""")

with tab3:

    st.warning("""
### Known Limitations

• Static analysis cannot perfectly estimate every algorithm.

• O(n²) and O(2ⁿ) occasionally appear structurally similar.

• Runtime depends on input size and hidden constants, which cannot always be inferred from syntax.

• The model currently supports Python source code only.
""")

# ---------- Frequently Asked Questions ----------

st.divider()

st.header("❓ Frequently Asked Questions")

with st.expander("How does the prediction work?"):
    st.write(
        "The Python code is converted into an Abstract Syntax Tree (AST). "
        "Several structural features are extracted and passed to a trained "
        "Random Forest classifier, which predicts the most probable complexity class."
    )

with st.expander("Does the application execute my code?"):
    st.success(
        "No. The model performs only static analysis. Your code is never executed."
    )

with st.expander("Why is the confidence sometimes low?"):
    st.write(
        "Some algorithms share similar structural patterns. "
        "For example, nested loops and recursive branching can produce similar AST features."
    )

with st.expander("Can this replace manual complexity analysis?"):
    st.write(
        "No. This tool is designed to assist developers and students. "
        "Human analysis is still the most reliable approach for complex algorithms."
    )

# ---------- Technology Stack ----------

st.divider()

st.header("🛠 Technology Stack")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    st.info("""
**Frontend**

• Streamlit

• streamlit-ace
""")

with tech2:
    st.info("""
**Backend**

• FastAPI

• REST API
""")

with tech3:
    st.info("""
**Machine Learning**

• Scikit-learn

• Random Forest
""")

with tech4:
    st.info("""
**Analysis**

• Python AST

• Static Analysis
""")

# ---------- Footer ----------

st.divider()

st.markdown(
"""
<div class="footer">

<h3>🚀 Big-O Time Complexity Predictor</h3>

Built using
<strong>Python</strong> •
<strong>Streamlit</strong> •
<strong>FastAPI</strong> •
<strong>Scikit-learn</strong> •
<strong>AST Analysis</strong>

<br><br>

Designed and Developed by <b>Tannu Ahlawat</b>

<br><br>

<i>Static Analysis • Machine Learning • Explainable Prediction</i>

</div>
""",
unsafe_allow_html=True
)
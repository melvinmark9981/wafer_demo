import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Wafer Defect Classifier — FYP2 Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== FILE NAMES ==================
DATA_FILE     = "wafer_demo_v2.npz"
BASELINE_FILE = "baseline_cnn_v2_best.keras"
HYBRID_FILE   = "hybrid_cnn_se_v2_best.keras"

# ================== HEADER ==================
st.title("🔬 Wafer Map Defect Classification")
st.caption("Melvin Mark Michael · 202210051 · BDS32013 Final Year Project II · Raffles University Malaysia")
st.markdown("---")

# ================== CHECK FILES ==================
missing = [f for f in [DATA_FILE, BASELINE_FILE, HYBRID_FILE] if not os.path.exists(f)]
if missing:
    st.error(f"❌ Missing file(s): {', '.join(missing)}")
    st.info("➡️ Put all files in the same folder as streamlit_app_v2.py and rerun.")
    st.stop()

# ================== LOAD DATA ==================
@st.cache_resource
def load_data():
    data     = np.load(DATA_FILE)
    X_test   = data["X_test"]           # (N, 48, 48, 1)
    y_test   = data["y_test"]           # (N,)
    classes  = data["classes"].tolist()
    return X_test, y_test, classes

@st.cache_resource
def load_models():
    baseline = keras.models.load_model(BASELINE_FILE, compile=False)
    hybrid   = keras.models.load_model(HYBRID_FILE,   compile=False)
    return baseline, hybrid

@st.cache_data
def build_class_index(_y_test, _classes):
    """Pre-compute indices for each class for instant lookup."""
    index = {}
    for cls in _classes:
        cls_id = _classes.index(cls)
        index[cls] = np.where(np.array(_y_test) == cls_id)[0].tolist()
    return index

with st.spinner("Loading models and data..."):
    X_test, y_test, classes = load_data()
    baseline_model, hybrid_model = load_models()
    class_index = build_class_index(tuple(y_test), tuple(classes))

model_dict = {
    "Baseline CNN v2  —  95.18% accuracy": baseline_model,
    "Hybrid CNN+SE v2  —  95.74% accuracy": hybrid_model,
}

# ================== SESSION STATE ==================
if "idx" not in st.session_state:
    st.session_state.idx = 0

# ================== SIDEBAR ==================
st.sidebar.header("🎛️ Controls")

# Model selector
model_name = st.sidebar.selectbox("Choose Model", list(model_dict.keys()))

st.sidebar.markdown("---")

# --- FIND BY CLASS (new feature) ---
st.sidebar.subheader("🔍 Find by Class")
st.sidebar.caption("Jump instantly to any defect type")

# Highlight the hard classes
hard_classes = ["Scratch", "Loc", "Near-full", "Donut"]
all_classes_sorted = hard_classes + [c for c in classes if c not in hard_classes and c != "none"] + ["none"]

selected_class = st.sidebar.selectbox(
    "Select defect class",
    all_classes_sorted,
    format_func=lambda c: f"⭐ {c} (hard!)" if c in hard_classes else c
)

col_find1, col_find2 = st.sidebar.columns(2)

with col_find1:
    if st.button("▶ First", use_container_width=True):
        indices = class_index.get(selected_class, [])
        if indices:
            st.session_state.idx = int(indices[0])
            st.session_state.manual_idx = int(indices[0])
            st.rerun()

with col_find2:
    if st.button("🔀 Random", use_container_width=True):
        indices = class_index.get(selected_class, [])
        if indices:
            picked = int(np.random.choice(indices))
            st.session_state.idx = picked
            st.session_state.manual_idx = picked
            st.rerun()

# Show how many samples exist for selected class
count = len(class_index.get(selected_class, []))
st.sidebar.caption(f"📊 {count} samples of **{selected_class}** in test set")

st.sidebar.markdown("---")

# --- MANUAL NAVIGATION ---
st.sidebar.subheader("🔢 Manual Navigation")

# Sync manual_idx with idx
if "manual_idx" not in st.session_state:
    st.session_state.manual_idx = st.session_state.idx

idx_input = st.sidebar.number_input(
    "Sample index",
    min_value=0,
    max_value=int(len(X_test) - 1),
    value=int(st.session_state.manual_idx),
    step=1,
    key="idx_input"
)

# Only update if user manually changed it
if int(idx_input) != st.session_state.manual_idx:
    st.session_state.manual_idx = int(idx_input)
    st.session_state.idx = int(idx_input)
    st.rerun()

if st.sidebar.button("🎲 Pick completely random", use_container_width=True):
    picked = int(np.random.randint(0, len(X_test)))
    st.session_state.idx = picked
    st.session_state.manual_idx = picked
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"**Total test samples:** {len(X_test):,}")

# ================== GET CURRENT SAMPLE ==================
idx     = st.session_state.idx
x       = X_test[idx]
y_true  = int(y_test[idx])
model   = model_dict[model_name]

x_input = np.expand_dims(x, axis=0)

# ================== PREDICT BOTH MODELS ==================
y_prob_baseline = baseline_model.predict(x_input, verbose=0)[0]
y_prob_hybrid   = hybrid_model.predict(x_input, verbose=0)[0]
y_prob          = model.predict(x_input, verbose=0)[0]

y_pred_baseline = int(np.argmax(y_prob_baseline))
y_pred_hybrid   = int(np.argmax(y_prob_hybrid))
y_pred          = int(np.argmax(y_prob))

true_label      = classes[y_true]
pred_label      = classes[y_pred]
pred_baseline   = classes[y_pred_baseline]
pred_hybrid     = classes[y_pred_hybrid]

correct         = y_pred == y_true
base_correct    = y_pred_baseline == y_true
hyb_correct     = y_pred_hybrid   == y_true

# ================== MAIN LAYOUT ==================
col_img, col_pred, col_compare = st.columns([1, 1, 1])

# --- Column 1: Wafer Map Image ---
with col_img:
    st.subheader(f"Sample #{idx}")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(x.squeeze(), cmap="viridis")
    ax.set_title(f"True Label: {true_label}", fontsize=13, fontweight="bold", color="#1a1a2e")
    ax.axis("off")
    st.pyplot(fig)
    plt.close()

# --- Column 2: Current Model Prediction ---
with col_pred:
    st.subheader("Prediction")
    st.markdown(f"**Model:** `{model_name}`")

    if correct:
        st.success(f"✅ CORRECT — Predicted: **{pred_label}**")
    else:
        st.error(f"❌ WRONG — Predicted: **{pred_label}** | True: **{true_label}**")

    st.markdown("**Class Probabilities:**")
    # Sort by probability descending
    sorted_probs = sorted(zip(classes, y_prob), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_probs:
        bar_color = "🟢" if cls == true_label else ("🔴" if cls == pred_label and not correct else "⬜")
        st.markdown(f"{bar_color} `{cls:<12}` **{prob:.4f}** ({prob*100:.1f}%)")

# --- Column 3: Head-to-head comparison ---
with col_compare:
    st.subheader("🥊 Model Comparison")
    st.caption("Both models on this same sample")

    # Baseline result
    if base_correct:
        st.success(f"✅ **Baseline CNN**\nPredicted: `{pred_baseline}`\nConfidence: {y_prob_baseline.max():.3f}")
    else:
        st.error(f"❌ **Baseline CNN**\nPredicted: `{pred_baseline}`\nConfidence: {y_prob_baseline.max():.3f}")

    st.markdown("")

    # Hybrid result
    if hyb_correct:
        st.success(f"✅ **Hybrid CNN+SE**\nPredicted: `{pred_hybrid}`\nConfidence: {y_prob_hybrid.max():.3f}")
    else:
        st.error(f"❌ **Hybrid CNN+SE**\nPredicted: `{pred_hybrid}`\nConfidence: {y_prob_hybrid.max():.3f}")

    # Highlight if one is better
    if base_correct and not hyb_correct:
        st.warning("⚠️ Baseline got it, Hybrid missed!")
    elif hyb_correct and not base_correct:
        st.info("🏆 Hybrid got it right, Baseline missed!")
    elif hyb_correct and base_correct:
        st.success("🎯 Both models correct!")
    else:
        st.error("Both models wrong on this sample")

# ================== STATS BAR ==================
st.markdown("---")
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Baseline Accuracy", "95.18%")
s2.metric("Hybrid Accuracy",   "95.74%", "+0.56%")
s3.metric("Hybrid Macro F1",   "83.88%", "+3.10%")
s4.metric("Scratch Recall",    "+17%",   "Hybrid vs Baseline")
s5.metric("Near-full Recall",  "1.00",   "Perfect (Hybrid)")

# ================== DEMO TIPS ==================
with st.expander("💡 Demo Tips — Click to expand"):
    st.markdown("""
    **Hard classes to showcase (use Find by Class above):**
    - 🔴 **Scratch** — Only 119 test samples. Baseline recall 0.77 vs Hybrid 0.94 (+17%)
    - 🔴 **Loc** — Only 359 test samples. Baseline recall 0.76 vs Hybrid 0.85 (+9%)
    - 🔴 **Near-full** — Only 15 test samples. Hybrid achieves **perfect recall 1.00**
    - 🔴 **Donut** — Only 55 test samples. Minimal improvement but rare class

    **What to say during demo:**
    > *"I will now find a Scratch sample — one of the hardest classes with only 119 test samples.
    > Watch how the Hybrid model correctly identifies it while the Baseline struggles."*

    **Navigation shortcuts:**
    - Use **Find by Class → Scratch → Random** to jump to a Scratch sample instantly
    - Switch models to compare on the same sample
    - The **Model Comparison** panel shows both results side by side automatically
    """)
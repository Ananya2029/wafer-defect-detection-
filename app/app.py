"""
Wafer defect inspection app.

    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wafer.data import LABELS, ROOT_CAUSES, load_splits  # noqa: E402
from wafer.explain import grad_cam, load_model, predict_one  # noqa: E402

st.set_page_config(page_title="Wafer Defect Inspection", page_icon="🔬", layout="wide")
WAFER_CMAP = ListedColormap(["white", "#9ecae1", "#d62728"])  # blank, good die, defective die


@st.cache_resource
def resources():
    model, thr = load_model()
    splits = load_splits()
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    return model, thr, splits["test"], metrics


def draw_map(ax, wafer, title):
    ax.imshow(wafer, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


model, thresholds, (X_test, y_test), metrics = resources()
cnn = metrics["results"]["CNN"]

st.title("🔬 Wafer Defect Pattern Inspection")
st.caption(f"CNN trained on MixedWM38 · test exact-match {cnn['exact_match']:.1%} · "
           f"macro-F1 {cnn['macro_f1']:.3f} · detects 8 patterns and their mixtures")

with st.sidebar:
    st.header("Pick a wafer")
    source = st.radio("From the held-out test set", ["Random wafer", "By pattern"])
    if source == "By pattern":
        pattern = st.selectbox("Contains pattern", LABELS)
        pool = np.nonzero(y_test[:, LABELS.index(pattern)] == 1)[0]
    else:
        pool = np.arange(len(X_test))
    if "idx" not in st.session_state or st.button("🎲 Next wafer"):
        st.session_state.idx = int(np.random.choice(pool))
    idx = st.session_state.idx if st.session_state.idx in pool else int(pool[0])
    st.caption(f"Test wafer #{idx}")

wafer, truth = X_test[idx], y_test[idx]
prob, detected = predict_one(model, thresholds, wafer)
true_labels = [LABELS[j] for j in range(len(LABELS)) if truth[j] == 1]

c1, c2 = st.columns([1, 1.3])
with c1:
    fig, ax = plt.subplots(figsize=(4, 4))
    draw_map(ax, wafer, "Wafer map (red = failed die)")
    st.pyplot(fig)
with c2:
    st.subheader("Detected: " + (", ".join(detected) if detected else "no defect pattern ✅"))
    st.write("Actual: " + (", ".join(true_labels) if true_labels else "normal wafer"))
    if set(detected) == set(true_labels):
        st.success("Prediction matches the label exactly.")
    else:
        st.warning("Prediction differs from the label.")
    st.bar_chart({"probability": {l: float(p) for l, p in zip(LABELS, prob)}}, horizontal=True)

if detected:
    st.subheader("Where the model looked (Grad-CAM) and likely root causes")
    cols = st.columns(len(detected))
    for col, name in zip(cols, detected):
        cam = grad_cam(model, wafer, LABELS.index(name))
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        draw_map(ax, wafer, f"{name}  ({prob[LABELS.index(name)]:.0%})")
        ax.imshow(np.ma.masked_where(cam < 0.5, cam), cmap="inferno", alpha=0.55)
        col.pyplot(fig)
        col.caption(f"**Typical cause:** {ROOT_CAUSES[name]}")

with st.expander("📊 Model performance on the test set"):
    st.image(str(ROOT / "reports" / "model_comparison.png"))
    st.dataframe({l: cnn["per_label"][l] for l in LABELS})

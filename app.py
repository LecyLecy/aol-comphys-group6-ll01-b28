from __future__ import annotations

import streamlit as st
from PIL import Image

from inference.api import load_model, predict_pil_ui, PredictConfig, INSTRUCTIONS

st.set_page_config(page_title="Trash Classifier MVP", page_icon="♻️", layout="centered")

st.title("♻️ Trash Classifier (TrashNet)")
st.caption("Upload foto / pakai kamera → prediksi kelas sampah + instruksi buang.")

# Cache model supaya load sekali saja
@st.cache_resource
def _load():
    model, labels, device = load_model()
    return model, labels, device

model, labels, device = _load()

cfg = PredictConfig(
    confidence_threshold=0.60,
    margin_threshold=0.15,
    topk=2,
)

tab1, tab2 = st.tabs(["Upload Image", "Camera"])

img: Image.Image | None = None

with tab1:
    f = st.file_uploader("Upload gambar (jpg/png)", type=["jpg", "jpeg", "png"])
    if f is not None:
        img = Image.open(f).convert("RGB")

with tab2:
    cam = st.camera_input("Ambil foto dari kamera")
    if cam is not None:
        img = Image.open(cam).convert("RGB")

if img is None:
    st.info("Masukkan gambar dulu ya (upload atau camera).")
    st.stop()

st.image(img, caption="Input image", use_container_width=True)

if st.button("Predict", type="primary"):
    out = predict_pil_ui(img, model, labels, device, cfg)

    label = out["label"]
    conf = out["confidence"]
    top = out["top"]
    needs_review = out["needs_review"]
    instruction = out["instruction"]

    st.subheader("Hasil Prediksi")
    st.write(f"**Label:** `{label}`")
    st.write(f"**Confidence:** `{conf:.3f}`")

    st.subheader("Instruksi Buang")
    st.success(instruction)

    st.subheader("Top-2 Prediksi")
    st.write(f"1) `{top[0]['label']}` — {top[0]['confidence']:.3f}")
    if len(top) > 1:
        st.write(f"2) `{top[1]['label']}` — {top[1]['confidence']:.3f}")

    if needs_review:
        st.warning("Model butuh review (ambiguous / confusable). Foto ulang atau pilih manual di bawah.")

        choices = [t["label"] for t in top]
        manual = st.selectbox("Pilih label manual (override)", choices, index=0)

        if st.button("Apply Override"):
            st.write(f"✅ Override ke: `{manual}`")
            st.success(INSTRUCTIONS.get(manual, "Buang sesuai kategori yang benar."))

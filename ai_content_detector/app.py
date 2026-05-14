"""AI Content Detector — Streamlit application.

Detects whether text or images are likely AI-generated using an ensemble
of internal (bench-aitextdetect, bench-imai-artifact) and external
(Binoculars, Fast-DetectGPT, DivEye, SigLIP-2) detectors.

Usage:
    streamlit run ai_content_detector/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure ml_pipeline is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ML_PIPELINE = _REPO_ROOT / "ml_pipeline"
if str(_ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_ML_PIPELINE))
if str(_REPO_ROOT / "ai_content_detector") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "ai_content_detector"))

from detectors.ensemble import EnsembleAggregator, DetectionResult

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Content Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("AI Content Detector")
st.caption(
    "Detect AI-generated text and images using an ensemble of internal and external detectors."
)


# ---------------------------------------------------------------------------
# Detector loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_text_detectors():
    """Load all available text detectors."""
    from detectors.text_detectors import (
        ModernBERTDetector,
        TFIDFDetector,
        BinocularsDetector,
        FastDetectGPTDetector,
        DivEyeDetector,
        InversionDetector,
        IPADDetector,
    )

    candidates = [
        ModernBERTDetector(),
        TFIDFDetector(),
        BinocularsDetector(),
        FastDetectGPTDetector(),
        DivEyeDetector(),
        InversionDetector(),
        IPADDetector(),
    ]
    available = []
    unavailable = []
    for d in candidates:
        if d.is_available():
            available.append(d)
        else:
            unavailable.append(d.name)
    return available, unavailable


@st.cache_resource
def load_image_detectors():
    """Load all available image detectors."""
    from detectors.image_detectors import (
        EfficientNetDetector,
        CLIPImageDetector,
        DINOv2Detector,
        SigLIPDetector,
        FrequencyDetector,
        PatchBasedClassifier,
    )

    candidates = [
        EfficientNetDetector(),
        CLIPImageDetector(),
        DINOv2Detector(),
        SigLIPDetector(),
        FrequencyDetector(),
    ]
    
    # Try adding patch-based versions of available spatial classifiers
    if EfficientNetDetector().is_available():
        candidates.append(PatchBasedClassifier(EfficientNetDetector()))
    elif CLIPImageDetector().is_available():
        candidates.append(PatchBasedClassifier(CLIPImageDetector()))

    available = []
    unavailable = []
    for d in candidates:
        if d.is_available():
            available.append(d)
        else:
            unavailable.append(d.name)
    return available, unavailable


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    if score >= 0.65:
        return "🔴"
    if score <= 0.35:
        return "🟢"
    return "🟡"


def _label_text(label: str) -> str:
    return {"ai": "Likely AI-Generated", "human": "Likely Human", "uncertain": "Uncertain"}.get(
        label, label
    )


def display_results(results: dict):
    """Render detection results."""
    agg = results["aggregate_score"]
    agg_label = results["aggregate_label"]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            label="Overall Confidence",
            value=f"{agg:.1%}",
            help="0% = certainly human, 100% = certainly AI",
        )
        color = _score_color(agg)
        st.markdown(f"### {color} {_label_text(agg_label)}")

    with col2:
        st.subheader("Per-Detector Scores")
        for r in results["per_detector"]:
            icon = _score_color(r.score)
            label = f"{icon} **{r.detector_name}**: {r.score:.1%}"
            if r.label == "error":
                label += f" (error: {r.details.get('error', 'unknown')})"

            with st.expander(label, expanded=False):
                st.json(r.details)

            # Progress bar visualization
            st.progress(r.score, text=f"{r.detector_name}: {r.score:.1%}")


# ---------------------------------------------------------------------------
# Text detection tab
# ---------------------------------------------------------------------------

def text_tab():
    st.header("Text Detection")
    st.markdown("Paste text or upload a `.txt` file to check if it was AI-generated.")

    detectors, unavailable = load_text_detectors()

    if unavailable:
        st.info(f"Unavailable detectors (missing checkpoints or GPU): {', '.join(unavailable)}")

    if not detectors:
        st.error(
            "No text detectors available. Ensure model checkpoints exist "
            "and/or a GPU is available for zero-shot detectors."
        )
        return

    # Detector selection
    selected = st.multiselect(
        "Select detectors",
        options=[d.name for d in detectors],
        default=[d.name for d in detectors],
    )
    active_detectors = [d for d in detectors if d.name in selected]

    # Input
    input_method = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)

    text = ""
    if input_method == "Paste text":
        text = st.text_area(
            "Enter text to analyze",
            height=300,
            placeholder="Paste your text here...",
        )
    else:
        uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
        if uploaded:
            text = uploaded.read().decode("utf-8", errors="replace")
            st.text_area("Uploaded text", text, height=200, disabled=True)

    if st.button("Analyze Text", type="primary", disabled=not text.strip()):
        if len(text.strip()) < 50:
            st.warning("Please provide at least 50 characters for reliable detection.")
            return

        with st.spinner("Running detectors..."):
            ensemble = EnsembleAggregator(active_detectors)
            results = ensemble.detect(text)

        display_results(results)


# ---------------------------------------------------------------------------
# Image detection tab
# ---------------------------------------------------------------------------

def image_tab():
    st.header("Image Detection")
    st.markdown("Upload an image to check if it was AI-generated.")

    detectors, unavailable = load_image_detectors()

    if unavailable:
        st.info(f"Unavailable detectors (missing checkpoints or deps): {', '.join(unavailable)}")

    if not detectors:
        st.error(
            "No image detectors available. Ensure model checkpoints exist "
            "and required packages are installed."
        )
        return

    selected = st.multiselect(
        "Select detectors",
        options=[d.name for d in detectors],
        default=[d.name for d in detectors],
        key="img_det_select",
    )
    active_detectors = [d for d in detectors if d.name in selected]

    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
    )

    if uploaded:
        from PIL import Image

        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Running detectors..."):
                ensemble = EnsembleAggregator(active_detectors)
                results = ensemble.detect(image)

            display_results(results)


# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------

def about_tab():
    st.header("About")
    st.markdown("""
### Detectors

**Text Detectors:**
- **ModernBERT (QLoRA)** — Fine-tuned on MAGE dataset (ACL 2024) via QLoRA.
  Loaded from `bench-aitextdetect` checkpoint.
- **TF-IDF + LogReg** — Classical baseline from the same benchmark.
- **Binoculars** (ICML 2024) — Zero-shot detector using cross-perplexity ratio
  of two reference LMs. >90% TPR at 0.01% FPR.
- **Fast-DetectGPT** (ICLR 2024) — Zero-shot via conditional probability curvature.
  340x faster than original DetectGPT.
- **DivEye** (TMLR 2026) — Zero-shot using surprisal diversity features.
  Outperforms prior zero-shot detectors by up to 33.2%.

**Image Detectors:**
- **EfficientNet-B4-NS** — Trained on ArtiFact (25 generators, 200x200).
  Best in-domain accuracy from `bench-imai-artifact`.
- **CLIP Classifier** — Best cross-dataset generalizer from the same benchmark.
- **DINOv2 ViT-B** — Self-supervised ViT pretrained on diverse data.
- **SigLIP-2 Detector** — HuggingFace pipeline for deepfake detection
  using Google's SigLIP-2 vision transformer.

### Methodology

Each detector independently scores the input. The **ensemble aggregator**
combines scores via configurable weighted averaging. The displayed confidence
represents the estimated probability that the content is AI-generated.

### References

- Hans et al., *Spotting LLMs With Binoculars* (ICML 2024)
- Bao et al., *Fast-DetectGPT* (ICLR 2024)
- Basani & Chen, *Diversity Boosts AI-Generated Text Detection* (TMLR 2026)
- Li et al., *MAGE* (ACL 2024)
- ArtiFact Dataset (25 generators)
    """)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

tab_text, tab_image, tab_about = st.tabs(["Text Detection", "Image Detection", "About"])

with tab_text:
    text_tab()

with tab_image:
    image_tab()

with tab_about:
    about_tab()

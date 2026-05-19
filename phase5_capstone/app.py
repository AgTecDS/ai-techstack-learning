"""
app.py
------
Streamlit frontend for the image search capstone.

Flow:
  1. User uploads a PNG/JPEG of a handwritten digit
  2. App runs ONNX inference → predicted class + confidence
  3. App embeds the image and queries Qdrant for the 5 most similar indexed images
  4. Results are shown in a grid with similarity scores

Run locally:
    streamlit run app.py

With Docker Compose:
    docker compose up --build
    open http://localhost:8501
"""

import io
import os
import sys

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# Add pipeline to path
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    ONNX_MODEL_PATH,
    OnnxInferenceEngine,
    ensure_collection,
    get_qdrant_client,
    search_similar,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MNIST Image Search",
    page_icon="🔍",
    layout="wide",
)

# ── Session state helpers ──────────────────────────────────────────────────────
# st.session_state persists objects across reruns within the same browser session.
# We use it to cache the model and database client so they are not re-created
# on every interaction (which would be slow and expensive).

@st.cache_resource(show_spinner="Loading ONNX model...")
def load_engine() -> OnnxInferenceEngine:
    """Cached: loaded once and reused across all user sessions."""
    try:
        return OnnxInferenceEngine(ONNX_MODEL_PATH)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


@st.cache_resource(show_spinner="Connecting to Qdrant...")
def load_qdrant():
    """Cached: one Qdrant client reused across sessions."""
    client = get_qdrant_client()
    ensure_collection(client)
    return client


# ── Utility ────────────────────────────────────────────────────────────────────

def create_placeholder_image(label: int, score: float) -> Image.Image:
    """
    Generate a simple placeholder image showing the digit label and score.
    Used when the actual image bytes are not stored in Qdrant payload.
    In a production system you would fetch the image from MinIO.
    """
    img  = Image.new("L", (80, 80), color=240)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), str(label), fill=0)
    draw.text((5, 50), f"{score:.3f}", fill=100)
    return img


def softmax_bar(probs: list):
    """Render class probabilities as a horizontal bar chart."""
    import pandas as pd
    data = {"digit": list(range(10)), "probability": probs}
    st.bar_chart(
        data={"probability": probs},
        use_container_width=True,
        height=160,
    )


# ── Main UI ────────────────────────────────────────────────────────────────────

def main():
    st.title("MNIST Digit Image Search")
    st.markdown(
        "Upload a handwritten digit image. The app will classify it using an ONNX model "
        "and find the most visually similar images from the indexed database using "
        "Qdrant vector search."
    )

    # Load shared resources
    engine = load_engine()
    client = load_qdrant()

    # ── Sidebar controls ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Search settings")
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        use_filter = st.checkbox("Filter by predicted label", value=False)
        label_filter = None
        if use_filter:
            label_filter = st.selectbox("Digit label", list(range(10)), index=0)

        st.divider()
        st.subheader("Database info")
        try:
            from qdrant_client.models import CountResult
            count = client.count(collection_name=COLLECTION_NAME, exact=True)
            st.metric("Indexed vectors", count.count)
        except Exception:
            st.warning("Could not retrieve count.")

        st.markdown("**Seed the database:**")
        seed_n = st.number_input("Images to index", min_value=10, max_value=10000,
                                  value=200, step=50)
        if st.button("Seed from MNIST test set"):
            with st.spinner(f"Indexing {seed_n} images..."):
                try:
                    from pipeline import seed_from_mnist
                    seed_from_mnist(seed_n)
                    st.success(f"Indexed {seed_n} images!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Seeding failed: {e}")

    # ── File upload ────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a digit image (PNG or JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Draw a digit in Paint or use any MNIST sample image.",
    )

    if uploaded is None:
        st.info("Upload an image above to get started.")

        # Show a demo section
        st.subheader("How it works")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**1. ONNX Inference**")
            st.markdown("The uploaded image is preprocessed (grayscale, 28x28, normalized) "
                        "and fed through the ONNX model to get a predicted digit and "
                        "probability distribution.")
        with col2:
            st.markdown("**2. Embedding Extraction**")
            st.markdown("A 128-dimensional feature vector is extracted from the model's "
                        "penultimate layer. This captures the image's visual characteristics "
                        "independent of the final classification.")
        with col3:
            st.markdown("**3. Vector Search**")
            st.markdown("Qdrant's HNSW index finds the K most similar embeddings in "
                        "sub-millisecond time, even with millions of vectors.")
        return

    # ── Process uploaded image ─────────────────────────────────────────────────
    raw_bytes = uploaded.read()
    query_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    col_img, col_pred = st.columns([1, 2])

    with col_img:
        st.subheader("Query image")
        st.image(query_img, width=200)

    with col_pred:
        st.subheader("Prediction")
        with st.spinner("Running ONNX inference..."):
            predicted, confidence, probs = engine.predict(query_img)

        st.metric("Predicted digit", predicted)
        st.metric("Confidence", f"{confidence:.2%}")
        st.markdown("**Class probabilities:**")
        softmax_bar(probs)

    # ── Similarity search ──────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Top {top_k} similar images from database")

    with st.spinner("Searching Qdrant..."):
        results = search_similar(
            client, engine, query_img,
            top_k=top_k,
            label_filter=label_filter,
        )

    if not results:
        st.warning(
            "No results found. The database may be empty. "
            "Use the sidebar to seed it with MNIST images."
        )
        return

    # Display results in a grid
    cols = st.columns(min(top_k, 5))
    for i, hit in enumerate(results):
        col = cols[i % 5]
        with col:
            payload = hit.payload or {}
            label   = payload.get("predicted_label", "?")
            true_l  = payload.get("true_label", "?")

            # In production: fetch actual image bytes from MinIO using the image_path payload
            placeholder = create_placeholder_image(label, hit.score)
            st.image(placeholder, width=80, caption=f"score: {hit.score:.3f}")
            st.caption(f"Predicted: {label} | True: {true_l}")

    # Show raw results table
    with st.expander("Raw search results"):
        import pandas as pd
        rows = []
        for hit in results:
            p = hit.payload or {}
            rows.append({
                "id":              hit.id,
                "score":           round(hit.score, 5),
                "predicted_label": p.get("predicted_label"),
                "true_label":      p.get("true_label"),
                "confidence":      round(p.get("confidence", 0), 4),
                "split":           p.get("split"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    main()

import streamlit as st
import numpy as np
from pathlib import Path

from streamlit_drawable_canvas import st_canvas
from src.predictor import Predictor


st.set_page_config(
    page_title="AI Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="wide"
)

css_file = Path(__file__).parent / "assets" / "style.css"

with open(css_file, "r", encoding="utf-8") as file:
    st.markdown(
        f"<style>{file.read()}</style>",
        unsafe_allow_html=True
    )


with st.sidebar:

    st.title("🤖 AI Digit Recognizer")

    st.info(
        """
### Model

**Framework:** PyTorch

**Architecture:** CNN

**Dataset:** MNIST

**Training Accuracy:** 99.46%

**Classes:** 10
"""
    )

    st.success("✅ Real-time Prediction")
    st.success("✅ Top-3 Results")
    st.success("✅ Confidence Score")


st.title("✍️ AI Handwritten Digit Recognizer")

st.markdown(
"""
Recognize handwritten digits using a **PyTorch Convolutional Neural Network** trained on the **MNIST dataset**.

Draw a digit on the canvas and click **Predict Digit**.
"""
)

st.divider()


@st.cache_resource
def load_model():
    return Predictor()


predictor = load_model()

left_col, right_col = st.columns([1, 1])


with left_col:

    st.subheader("📝 Draw Digit")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=320,
        width=320,
        drawing_mode="freedraw",
        key="canvas",
    )

    predict = st.button(
        "🚀 Predict Digit",
        use_container_width=True
    )


with right_col:

    st.subheader("🤖 Prediction")

    if predict:

        if canvas_result.image_data is not None:

            image = canvas_result.image_data[:, :, :3]
            image = np.mean(image, axis=2)

            results = predictor.predict(image)

            best = results[0]

            st.metric(
                "Predicted Digit",
                best["digit"]
            )

            st.metric(
                "Confidence",
                f"{best['confidence']*100:.2f}%"
            )

            st.markdown("### 🏆 Top 3 Predictions")

            for item in results:

                st.write(f"**Digit {item['digit']}**")

                st.progress(
                    float(item["confidence"])
                )

                st.caption(
                    f"{item['confidence']*100:.2f}%"
                )

        else:

            st.warning(
                "Please draw a digit first."
            )


st.divider()

st.caption(
    "Built using PyTorch • Streamlit • MNIST Dataset"
)
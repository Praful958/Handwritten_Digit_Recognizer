import streamlit as st
import numpy as np

from streamlit_drawable_canvas import st_canvas

from src.predictor import Predictor


# Page Config
st.set_page_config(
    page_title="AI Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="centered"
)


# Title
st.title("✍️ AI Handwritten Digit Recognizer")

st.write(
    "Draw a digit and our PyTorch CNN model will predict it."
)


# Load Model
@st.cache_resource
def load_predictor():

    return Predictor()


predictor = load_predictor()


# Canvas
st.subheader("🖊️ Draw Digit Here")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


# Predict Button
if st.button("🔍 Predict"):

    if canvas_result.image_data is not None:

        image = canvas_result.image_data[:, :, :3]

        image = np.mean(
            image,
            axis=2
        )


        results = predictor.predict(image)


        st.subheader("🏆 Top 3 Predictions")


        rank = 1

        for item in results:

            digit = item["digit"]

            confidence = item["confidence"]


            st.write(
                f"### #{rank} Prediction: {digit}"
            )


            st.progress(
                float(confidence)
            )


            st.write(
                f"Confidence: {confidence*100:.2f}%"
            )


            st.divider()


            rank += 1


    else:

        st.warning(
            "Please draw a digit first."
        )
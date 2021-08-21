import numpy as np
import streamlit as st
import torch
from PIL import Image
from src.utils import transform_inference

DEMO_IMAGE = "/workdir/web_app/sample_from_test/normal.jpeg"


def setup_parameters():
    st.title("XRay classification using ResNet152")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 350px;
            margin-left: -350px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Image parameters")

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert("RGB")

    else:
        demo_image = DEMO_IMAGE
        image = Image.open(demo_image).convert("RGB")

    st.sidebar.text("Original Image")
    st.sidebar.image(image)

    return image


def model_inference(model, image):
    image = transform_inference(image)
    image = image.unsqueeze_(0)
    logits = model(image)
    proba = torch.exp(logits).detach().numpy()[0]

    return proba


def setup_annotation(proba, image):
    st.subheader("Output Image")
    st.image(np.array(image), use_column_width=True)

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown("**COVID-19 probability**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Normal probability**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Viral Pneumonia probability**")
        kpi3_text = st.markdown("0")

    kpi1_text.write(
        "<h1 style='text-align: center; color: red;'>{:.2f}</h1>".format(round(proba[0], 2)), unsafe_allow_html=True
    )
    kpi2_text.write(
        "<h1 style='text-align: center; color: red;'>{:.2f}</h1>".format(round(proba[1], 2)), unsafe_allow_html=True
    )
    kpi3_text.write(
        "<h1 style='text-align: center; color: red;'>{:.2f}</h1>".format(round(proba[2], 2)), unsafe_allow_html=True
    )

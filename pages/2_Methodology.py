import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Methodology", layout="centered")

st.title("Methodology")

# Get the path of the image relative to this script
image_path = os.path.join(os.path.dirname(__file__), "methodology.png")

if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, caption="Methodology Diagram", use_container_width=True)
else:
    st.warning("Methodology image not found. Please ensure 'methodology.png' is in the pages directory.")

# streamlit_app.py
import streamlit as st
from PIL import Image
from inference import run_instantid
import requests
from io import BytesIO

st.set_page_config(page_title="AI Portrait Generator with InstantID")
st.title("AI Portrait Generator with InstantID")
st.write("Upload a clear face photo or provide an Instagram image URL, and describe the style to generate a portrait.")

# Option 1: Upload a face image
uploaded_file = st.file_uploader("Upload a Face Image", type=["jpg", "jpeg", "png"])

# Option 2: Paste an Instagram photo URL
insta_url = st.text_input("Or paste an Instagram image URL")

# Prompt
prompt = st.text_input("Prompt", value="A cinematic portrait in neon lighting")

generate_btn = st.button("Generate Portrait")

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif insta_url:
    try:
        response = requests.get(insta_url, timeout=10)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            st.error("Failed to fetch image from URL.")
    except Exception as e:
        st.error(f"Error fetching image: {e}")

if image and generate_btn:
    with st.spinner("Generating portrait..."):
        result = run_instantid(image, prompt)
        if result:
            st.image(result, caption="Generated Portrait", use_column_width=True)
        else:
            st.error("Something went wrong during image generation.")

# streamlit_app.py
import streamlit as st
from PIL import Image
from inference import run_instantid
import requests
from io import BytesIO
from bs4 import BeautifulSoup

st.set_page_config(page_title="AI Portrait Generator with InstantID")
st.title("AI Portrait Generator with InstantID")
st.write("Upload a clear face photo or provide an Instagram image URL, and describe the style to generate a portrait.")

# Option 1: Upload a face image
uploaded_file = st.file_uploader("Upload a Face Image", type=["jpg", "jpeg", "png"])

# Option 2: Paste an Instagram photo URL
insta_url = st.text_input("Or paste an Instagram post URL")

# Prompt
prompt = st.text_input("Prompt", value="A cinematic portrait in neon lighting")

generate_btn = st.button("Generate Portrait")

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif insta_url:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(insta_url, headers=headers, timeout=10).text
        soup = BeautifulSoup(html, 'html.parser')
        meta_tag = soup.find("meta", property="og:image")
        if meta_tag:
            image_url = meta_tag.get("content")
            response = requests.get(image_url, headers=headers, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            st.error("Could not find image in Instagram page metadata.")
    except Exception as e:
        st.error(f"Error fetching image from Instagram: {e}")

if image and generate_btn:
    with st.spinner("Generating portrait..."):
        result = run_instantid(image, prompt)
        if result:
            st.image(result, caption="Generated Portrait", use_column_width=True)
        else:
            st.error("Something went wrong during image generation. Please make sure the face is clear and frontal.")

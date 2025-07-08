# streamlit_app.py
import streamlit as st
from PIL import Image
from inference import run_instantid

st.set_page_config(page_title="AI Portrait Generator with InstantID")
st.title("AI Portrait Generator with InstantID")
st.write("Upload a clear face photo and describe the style to generate a portrait.")

uploaded_file = st.file_uploader("Upload a Face Image", type=["jpg", "jpeg", "png"])
prompt = st.text_input("Prompt", value="A cinematic portrait in neon lighting")

generate_btn = st.button("Generate Portrait")

if uploaded_file and generate_btn:
    image = Image.open(uploaded_file).convert("RGB")
    with st.spinner("Generating portrait..."):
        result = run_instantid(image, prompt)
        if result:
            st.image(result, caption="Generated Portrait", use_column_width=True)
        else:
            st.error("Something went wrong during image generation.")

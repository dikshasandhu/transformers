import streamlit as st
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
import requests
import tempfile

# Set up the LLaVA model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

# Streamlit app UI
st.title("LLaVA: Image Captioning App")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prompt input
    prompt = st.text_area("Enter your prompt", "USER: <image>\nDescribe this image\nASSISTANT:")

    # Button to generate caption
    if st.button('Generate Caption'):
        with st.spinner('Generating caption...'):
            # Prepare image and prompt
            inputs = processor(prompt, images=[img], padding=True, return_tensors="pt").to("cuda")
            output = model.generate(**inputs, max_new_tokens=50)
            generated_text = processor.batch_decode(output, skip_special_tokens=True)
            for text in generated_text:
                st.write(f"Caption: {text.split('ASSISTANT:')[-1]}")

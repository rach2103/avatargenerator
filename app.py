import streamlit as st
import numpy as np
from PIL import Image
from api_tokens import HUGGING_FACE  # Assuming this is your Hugging Face API token file
import torch
from diffusers import StableDiffusionImg2ImgPipeline

@st.cache_resource
class AvatarGenerator:
    def __init__(self):
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "nitrosocke/Ghibli-Diffusion", torch_dtype=torch.float32, use_safetensors=True
        )

    def generate_avatar(self, text_prompt, image, inference_steps=50, guidance_scale=7.5):
        image = Image.open(image).convert("RGB")
        image = image.resize((512, 512))  # Resize the image to 512x512
        generator = torch.Generator().manual_seed(1024)  # Set the seed for reproducibility
        
        # Generate the avatar using the pipeline with the given parameters
        image = self.pipe(prompt=text_prompt, image=image, strength=0.5, guidance_scale=guidance_scale,
                          generator=generator, negative_prompt="nsfw", num_inference_steps=inference_steps).images[0]
        
        image.save(f"./{text_prompt}.jpg")  # Save the generated image locally
        return image

# Initialize the AvatarGenerator class
app = AvatarGenerator()

# Create a Streamlit app interface
st.title("🧑🏽‍🎨 Avatar Generator")

# Allow users to upload an image
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Get the text prompt from the user
text_prompt = st.text_input("Enter a text prompt:")

# Allow users to adjust inference steps and guidance scale using sliders
inference_steps = st.slider("Inference Steps", min_value=10, max_value=100, value=50, step=1)
guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5, step=0.1)

# Submit button to trigger image generation
submit_button = st.button("Generate Avatar")

# Check if the user has provided both an image and a text prompt
if image_file is None or text_prompt == "":
    st.error("Please provide both an image and a text prompt.")
else:
    # Generate the avatar image when the user clicks the submit button
    if submit_button:
        # Generate the avatar with the user-defined inference steps and guidance scale
        avatar = app.generate_avatar(f"ghibli style, {text_prompt}", image_file, inference_steps, guidance_scale)

        # Display the avatar image
        st.image(avatar, caption="Generated Avatar", use_column_width=True)

import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # Prepare the image for the model
    raw_image = Image.fromarray(input_image.astype('uint8'), 'RGB')

    inputs = processor(images=raw_image, return_tensors="pt")

    # Generate caption
    out = model.generate(**inputs, max_new_tokens=50)
    
    # Decode the generated caption
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="BLIP Image Captioning",
    description="Upload an image to generate a caption using the BLIP model."
)
iface.launch()
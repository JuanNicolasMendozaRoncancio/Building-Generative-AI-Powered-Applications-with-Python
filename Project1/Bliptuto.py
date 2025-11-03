import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_path = "C:/Users/juann/Building-Generative-AI-Powered-Applications-with-Python/Project1/images/UNAL.png"
image = Image.open(image_path).convert("RGB")

inputs = processor(image, return_tensors="pt")

print("Generating caption...")
outputs = model.generate(**inputs, max_new_tokens=50)

# Decodifica el resultado
caption = processor.decode(outputs[0], skip_special_tokens=True)
print("Generated Caption:", caption)

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" 
import torch
from transformers import pipeline

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
  ignore_warning=True,
)


sample = "Generative AI-Powered Meeting Assistant\speech_to_text_probe.mp3"

prediction = pipe(sample, batch_size=8)["text"]
print(prediction)
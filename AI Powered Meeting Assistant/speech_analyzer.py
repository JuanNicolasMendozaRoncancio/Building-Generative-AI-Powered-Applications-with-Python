import os
import torch
import gradio as gr
from huggingface_hub import login
from transformers import pipeline


TOKEN = "Your_Hugging_Face_Token_Here"
login(token=TOKEN)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

llm = pipeline(
    "text-generation",
    model="openai-community/gpt2",
    token=TOKEN,     
    device=-1        
)


def analyze_speech(audio_file):

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )

    text = pipe(audio_file, batch_size=8)["text"]

    prompt = f"Summarize and clean up the following meeting transcript:\n\n{text}\n\nSummary:"
    result = llm(prompt, temperature=0.1)
    summary = result[0]['generated_text']

    return summary


audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox(label="LLM Output")

iface = gr.Interface(
    fn=analyze_speech,
    inputs=audio_input,
    outputs=output_text,
    title="AI Meeting Companion",
    description="Upload an audio file — it will be transcribed and summarized by GPT-2.",
)

iface.launch()

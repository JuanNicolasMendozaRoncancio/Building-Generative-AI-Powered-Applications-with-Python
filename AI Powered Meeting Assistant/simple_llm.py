from huggingface_hub import login
from transformers import pipeline

TOKEN = "Your_Hugging_Face_Token_Here"
login(token=TOKEN)

llm = pipeline(
    "text-generation",
    model="openai-community/gpt2",
    token=TOKEN,     
    device=-1        
)

prompt = "Write a short poem about the sea."

result = llm(prompt, max_new_tokens=50, temperature=0.7)
print(result[0]['generated_text'])

import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging
logging.set_verbosity_error()


model_name = "facebook/blenderbot-400M-distill"


model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []
print("Chat Started. Write 'quit' to exit.")

while True:
    history = "\n".join(conversation_history)

    input_text = input("User: ")
    
    if input_text.lower() == 'quit':
        break

    inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")

    outputs = model.generate(**inputs, max_length=1000)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print("Chatbot:", response)

    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"Chatbot: {response}")


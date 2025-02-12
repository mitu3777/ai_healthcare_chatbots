import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
model_name = "microsoft/BioGPT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], attention_mask=attention_mask, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# for Streamlit 
st.title("AI Healthcare Chatbot")

st.write("Ask me about health-related symptoms or diseases")

# User input
user_input = st.text_input("Your Question:")


if user_input:
    response = generate_response(user_input)
    st.write(f"**AI's Response:** {response}")

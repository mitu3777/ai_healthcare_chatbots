import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load MedBERT model
medbert_model = AutoModelForCausalLM.from_pretrained('dmis-lab/biobert-large-cased-v1.1')
medbert_tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-large-cased-v1.1')

# Load BioGPT model
biogpt_model = AutoModelForCausalLM.from_pretrained('microsoft/biogpt')
biogpt_tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')


def get_response(query, model_type='biogpt'):
    if model_type == 'medbert':
        tokenizer = medbert_tokenizer
        model = medbert_model
    else:
        tokenizer = biogpt_tokenizer
        model = biogpt_model

    inputs = tokenizer.encode(query, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# to check if a question is medical
def is_medical(query):
    medical_keywords = ['health', 'symptom', 'disease', 'medicine', 'treatment', 'doctor', 'hospital']
    if any(keyword in query.lower() for keyword in medical_keywords):
        return True
    return False

 # for chatbot
def chatbot(query):
    if is_medical(query):
        response = get_response(query, model_type='medbert')
    else:
        response = get_response(query, model_type='biogpt')
    return response

# Streamlit 
st.title("Healthcare Chatbot")

user_input = st.text_input("Ask me anything related to health:")

if user_input:
    response = chatbot(user_input)
    st.write("Response: ", response)

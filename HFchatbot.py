from dotenv import load_dotenv
import os
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load env vars
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Streamlit setup
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")
st.title("💬 Generative AI Chatbot")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ Hugging Face model (Inference API)
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens=512,
        temperature=0.2
    )
)


# User input
user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    response = llm.invoke(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            *st.session_state.chat_history
        ]
    )

    assistant_response = response.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

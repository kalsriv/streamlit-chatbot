from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint

# Load env vars
load_dotenv()

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
    endpoint=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-70B-Instruct",   # ✅ choose any HF model
        max_new_tokens=512,
        temperature=0.2,
    )
)

# User input
user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # ✅ Chat-style invoke
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

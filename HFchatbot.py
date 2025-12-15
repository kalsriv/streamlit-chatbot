from dotenv import load_dotenv
import os
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# Load env vars
load_dotenv()
# hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

hf_token = st.secrets["HUGGINGFACE_API_KEY"]


# Streamlit setup
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")
st.title("💬 Generative AI Chatbot")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# # ✅ Hugging Face model (Inference API)
# llm = ChatHuggingFace(
#     llm=HuggingFaceEndpoint(
#         #repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#         repo_id="HuggingFaceH4/zephyr-7b-beta",
#         max_new_tokens=512,
#         temperature=0.2,
#         huggingfacehub_api_token=hf_token

#     )
# )


# # ✅ Hugging Face model (Inference API)
# llm = HuggingFaceEndpoint(
#     repo_id="bigscience/bloom",
#     max_new_tokens=512,
#     temperature=0.2,
#     huggingfacehub_api_token=hf_token
# )


llm = HuggingFaceEndpoint(
   repo_id="Writer/clinical-camel-70b-instruct-v2",
   task="text-generation",
#    max_new_tokens=512,
   do_sample=False,
   repetition_penalty=1.03,
   provider="auto"
)
chat_model = ChatHuggingFace(llm=llm)


# User input
user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Build prompt from chat history
    prompt = "You are a helpful assistant.\n"
    for message in st.session_state.chat_history:
        role = "User" if message["role"] == "user" else "Assistant"
        prompt += f"{role}: {message['content']}\n"
    prompt += "Assistant:"

    # Invoke model
    response = llm.invoke(prompt)
    #response = chat_model.invoke(prompt)
    

    assistant_response = response
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

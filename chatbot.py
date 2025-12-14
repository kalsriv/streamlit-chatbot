from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq


# load the env variables
load_dotenv()

# streamlit page setup
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered",
)
st.title("💬 Generative AI Chatbot")

# initiate chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# show chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# llm initiate
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
)

# input box
user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    response = llm.invoke(
        input = [{"role": "system", "content": "You are a helpful assistant"}, *st.session_state.chat_history]
    )
    assistant_response = response.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

# “I switched my VS Code workspace into a Dev Container because it gives me a clean, Linux‑based development environment that’s consistent with most production systems. It also lets me use a proper bash shell instead of Windows PowerShell, which simplifies tooling, package management, and dependency isolation. VS Code generated a .devcontainer folder for configuration, and I added it to .gitignore so it doesn’t pollute the repository. This keeps my Git history clean while still letting me work inside a reproducible Linux environment.”
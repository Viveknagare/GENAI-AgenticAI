from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

st.title("🤖 AI Q&A Bot")
st.markdown("Q&A Bot with langchain and gemini")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
    
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Please enter your message")
if query:
    st.session_state.messages.append({"role" : "user", "content" : query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    st.chat_message("assistant").markdown(res.content)
    st.session_state.messages.append({"role" : "assistant" , "content" : res.content})
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from serpapi_search_tools import web_search, maps_search, news_search
import streamlit as st

llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True)

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()
    st.session_state.History = []

agent = create_agent(
    model=llm,
    checkpointer=st.session_state.memory,
    tools=[web_search(), maps_search(), news_search()],
    system_prompt="you are a Q&A bot which can answer the user very quickly in few milliseconds and can search on google as well"
)

print(st.session_state.memory)

#Web Interface

st.subheader("⚡️Get Answer at the speed of thought")

for message in st.session_state.History:
    st.chat_message(message['role']).markdown(message['content'])

query = st.chat_input("Enter your Question")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.History.append({'role' : 'user', 'content' : query})
    
    response = agent.invoke(
    {'messages':[{'role' : 'user', 'content' : query}]},
    {'configurable' : {'thread_id' : "4"}},
    stream_mode='messages'
    )
    
    ai_container = st.chat_message('ai')
    with ai_container:
        space = st.empty()
        
        message = ""
        
        for chunk in response:
            message = message + chunk[0].content
            space.write(message)
            
        st.session_state.History.append({'role' : 'ai', 'content' : message})
        
    # ans = response['messages'][-1].content
    # st.chat_message("ai").markdown(ans)

 



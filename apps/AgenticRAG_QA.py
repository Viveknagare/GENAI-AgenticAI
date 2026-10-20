from dotenv import load_dotenv
load_dotenv()
import os

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
     
if "agent" not in st.session_state:
    st.session_state.agent = None
    
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def process_document(path):
    #Load the documents

    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    ##Split into multiple chunks

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunk_data = splitter.split_documents(docs)

    #Embeddings and Vector DB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_DB = InMemoryVectorStore.from_documents(
        documents=chunk_data,
        embedding=embeddings
    )

    ##create Agent - tool, llm, system_prompt
    llm = ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def retrieve_tool(query:str):
        """
        Based on the user query, fetch and return the most relevant chunks from the Knowledge base.
        """
        similar_chunks = vector_DB.similarity_search(query=query, k=4)
        context = ""

        for chunk in similar_chunks:
            context += chunk.page_content + "\n\n"
            
        return context

    System_Prompt = """You are a helpful AI Assistant and answer questions using retrived context. 
    My Knowledge base consists of the details from the uploaded document.
    Always use the `retrieve_tool` tool for query requiring external knowledge
    """

    memory = InMemorySaver()

    Agent = create_agent(
        model=llm,
        tools=[retrieve_tool],
        system_prompt=System_Prompt,
        checkpointer=memory
    )
    
    st.session_state.agent = Agent
    st.session_state.document_uploaded = True

# Upload UI
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Ente a PDF File", type=['pdf'], accept_multiple_files=True)
    if uploaded:
        with st.spinner("processing..."):
            path = "./doc_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())
                   
            process_document(path=path) 
            st.rerun()  #variables would be updated and page will be refreshed according to the new data

# Chat UI

if st.session_state.document_uploaded and st.session_state.agent:
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    query = st.chat_input("Ask anything related to uploaded documents")
    if query:
        st.session_state.messages.append({'role' : 'user', 'content' : query})
        st.chat_message('user').markdown(query)
        response = st.session_state.agent.invoke(
            {'messages' : [{'role' : 'user', 'content' : query}]},
            {'configurable' : {'thread_id' : '1'}}
        )
        result = response['messages'][-1].content
        st.chat_message('assistant').markdown(result)
        st.session_state.messages.append({'role' : 'assistant', 'content' : result})
        


#Basic Agent without UI
# while True:
#     query = input("AI: ")
#     if query.lower() == "exit": 
#         break
#     response = Agent.invoke(
#         {'messages':[{'role':'user', 'content' : query}]},
#         {'configurable':{'thread_id' : '1'}}
#         )
#     print("AI: ", response['messages'][-1].content)



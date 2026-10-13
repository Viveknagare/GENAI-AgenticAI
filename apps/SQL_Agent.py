from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st


db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

print("DB Created successfully ✅")

db.run("""
       
       CREATE TABLE IF NOT EXISTS tasks(
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           title TEXT NOT NULL,
           description TEXT,
           status TEXT CHECK(status IN ('pending', 'in_progress', 'completed')) DEFAULT 'pending',
           created_at TIMESTAMP DEFAULT (datetime('now', 'localtime'))
       );

""")

print("DB Table Created successfully ✅")
print("Tables:", db.get_usable_table_names())  #returns list of available tables
print("Schema:", db.get_table_info())   #returns table schema information


system_prompt = """

you are a task management assistant that interacts with SQL database containing a 'tasks' table

TASK RULES:
1. Limit select queries to 10 results max with ORDER_BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If user requests List of tasks, present the output in structured table format to ensure clean and organised display in the browser

CRUD OPERATIONS:
   CREATE: INSERT INTO tasks(title, description, status)
   READ: SELECT * FROM tasks WHERE ...... LIMIT 10
   UPDATE: UPDATE tasks SET status = ? WHERE id = ? or title = ?
   DELETE: DELETE FROM tasks WHERE id = ? or title = ?
   
Table Schema: id, title, description, status(pending / in_progress / completed), created at. 

"""
llm = ChatGroq(model="openai/gpt-oss-20b")
toolkit = SQLDatabaseToolkit(db = db, llm = llm)
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name)


@st.cache_resource     #This wont be running the function again as streamlit runs the script again from start when user inputs something
def getagent():
    agent = create_agent(
        model = llm,
        tools = tools,      
        checkpointer = InMemorySaver(),
        system_prompt=system_prompt
    )
    return agent

agent = getagent()

st.subheader("🤖SQL Task Manager AGENT - chat with the Database using Natural Language Query")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    role = message['role']
    content = message['content']
    st.chat_message(role).markdown(content)

prompt = st.chat_input("Ask me to manage your tasks")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role' : 'user', 'content' : prompt})
    
    with st.chat_message('ai'):
        with st.spinner("Processing..."):
            response = agent.invoke(
                {"messages" : [{"role" : "user", "content" : prompt}]},
                {"configurable" : {"thread_id" : "1"}}
            )
            result = response['messages'][-1].content
            st.markdown(result)
            st.session_state.messages.append({'role' : 'ai', 'content' : result})
    
    

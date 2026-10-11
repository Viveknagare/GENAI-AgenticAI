from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from serpapi_search_tools import web_search, news_search, maps_search
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI(model = "gpt-4o")

agent = create_agent(
    model = llm,
    tools = [web_search(), maps_search(), news_search()],
    system_prompt = "you are a agent which is having access to google search so for any user query try to give the latest answer",
    checkpointer=InMemorySaver()
)

while True:
    query = input("user: ")
    if query in ["exit", "quit"]:
        break
    
    response = agent.invoke(
        {"messages":[{"role" : "user", "content" : query}]},
        {"configurable" : {"thread_id" : "10"}}
        )
    print(response['messages'][-1].content)
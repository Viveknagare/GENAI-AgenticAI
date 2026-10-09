from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from serpapi_search_tools import web_search, news_search, maps_search

llm = ChatGoogleGenerativeAI(model = "gemini-3.8-flash")

agent = create_agent(
    model = llm,
    tools = [web_search(), maps_search(), news_search()],
    system_prompt = "ou are a agent which is having access to google search so for any user query try to give the latest answer"
)

while True:
    query = input("user: ")
    if query in ["exit", "quit"]:
        break
    
    response = agent.invoke({"messages":[{"role" : "user", "content" : query}]})
    print(response['messages'][-1].content)
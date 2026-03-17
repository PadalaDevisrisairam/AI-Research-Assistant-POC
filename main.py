import os
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool

# Import your custom search function
from tools import web_search

# Load env (override=True ensures .env takes priority over system env vars)
load_dotenv(override=True)
load_dotenv(find_dotenv(), override=True)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# Tool
@tool
def search_tool(query: str):
    """Search the web for information"""
    return web_search(query)

tools = [search_tool]

# Create agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are an AI research assistant. Use tools when necessary."
)

# Loop
while True:

    query = input("\nAsk Research Question (type exit): ")

    if query.lower() == "exit":
        break

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    print("\nAnswer:\n")
    print(result["messages"][-1].content)
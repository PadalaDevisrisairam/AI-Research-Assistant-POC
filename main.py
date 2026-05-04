import os
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from rag import load_rag

# Import your custom search function
from tools import web_search

# Load env (override=True ensures .env takes priority over system env vars)
load_dotenv(override=True)
load_dotenv(find_dotenv(), override=True)
db = load_rag()
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

@tool
def calculator(expression: str):
    """Evaluate mathematical expressions"""
    return eval(expression)

@tool
def rag_search(query: str):
    """Search from local documents"""
    docs = db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

tools = [search_tool,calculator, rag_search]

# Create agent
agent = create_agent(
    model=llm,
    tools=tools,
   system_prompt="""
You are an AI Research Assistant.

You have access to:
1. Web Search (for latest info)
2. RAG Search (for internal knowledge)
3. Calculator

Rules:
- ALWAYS try RAG first for conceptual queries
- If RAG is insufficient → use web search
- Use web search for latest info
- Combine both when needed
- For numbers → use calculator


Always respond with:
1. Summary
2. Key Points
3. Conclusion
""",
debug=True
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

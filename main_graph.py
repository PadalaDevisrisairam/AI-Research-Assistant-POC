import os 
import operator
from typing import Annotated, TypedDict, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_core.tools import tool
from tools import web_search
from rag import load_rag
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv(override=True)
# Check if API Key is loaded
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY not found in .env file!")
else:
    print(f"API Key loaded successfully (starts with: {api_key[:5]}...)")

# 1. Define the State
class AgentState(TypedDict):
    # The 'add_messages' function tells LangGraph to append new messages 
    # to the existing list rather than overwriting them.
    messages: Annotated[list[BaseMessage], add_messages]
# 2. Initialize LLM and Tools
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Use 1.5-flash or 2.0-flash
    google_api_key=api_key, 
    temperature=0
)
# Re-define or import your tools
@tool
def search_tool(query: str):
    """Search the web for latest information."""
    return web_search(query)
@tool
def rag_search(query: str):
    """Search from local documents for concepts and definitions."""
    db = load_rag()
    docs = db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])
# Combine tools and bind them to the LLM
tools = [search_tool, rag_search]
llm_with_tools = llm.bind_tools(tools)
# 3. Define the Assistant Node
def assistant(state: AgentState):
    # This function calls the LLM with the current messages
    # and returns the response to be added to the state.
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
# 4. Create the Tool Node
# This is a pre-built node that automatically runs the tools 
# requested by the LLM.
tool_node = ToolNode(tools)
# 5. Build the Graph
builder = StateGraph(AgentState)
# Add our nodes to the graph
builder.add_node("assistant", assistant)
builder.add_node("tools", tool_node)
# Define the flow
# Start with the assistant
builder.set_entry_point("assistant")
# Add a conditional edge
# If the LLM says "I need a tool", it goes to 'tools'.
# If the LLM says "I'm done", it goes to 'END'.
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
# After the tools finish, they MUST go back to the assistant 
# to explain the results (This is the "Cycle").
builder.add_edge("tools", "assistant")
# Compile the graph into a runnable app
graph = builder.compile()
# 6. The Execution Loop
def run_researcher():
    print("--- Welcome to your Agentic Research Assistant ---")
    while True:
        user_input = input("\nAsk Research Question (type 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Start the graph with the user's message
        # 'stream' allows us to see each step the agent takes
        for event in graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                # Get the last message generated in this step
                last_msg = value["messages"][-1]
                
                # If it's a tool call, we print what tool is being used
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"\n[Agent]: Thinking... I need to use tools: {[tc['name'] for tc in last_msg.tool_calls]}")
                # If it's a tool result, we print that the tool finished
                elif last_msg.type == "tool":
                    print(f"[System]: Tool '{last_msg.name}' completed.")
                # If it's a final answer, we print it
                else:
                    print("\n[Final Answer]:")
                    print(last_msg.content)

if __name__ == "__main__":
    run_researcher()

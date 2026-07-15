
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from tools import web_search
from rag import load_rag
from langchain_core.messages import SystemMessage

# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(override=True)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Warning: GOOGLE_API_KEY not found in .env file!")
else:
    print(f"API Key loaded successfully (starts with: {api_key[:5]}...)")


# ============================================================
# 2. DEFINE AGENT STATE
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[str]


# ============================================================
# 3. INITIALIZE LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)


# ============================================================
# 4. DEFINE TOOLS
# ============================================================

@tool
def search_tool(query: str):
    """Search the web for latest information."""
    return web_search(query)


@tool
def rag_search(query: str):
    """Search from local documents for concepts and definitions."""
    
    db = load_rag()

    docs = db.similarity_search(
        query,
        k=3
    )

    return "\n".join([
        doc.page_content
        for doc in docs
    ])


tools = [
    search_tool,
    rag_search
]


# ============================================================
# 5. BIND TOOLS TO LLM
# ============================================================

llm_with_tools = llm.bind_tools(tools)

#=============================================================
# Define the Planner node 
#=============================================================

def planner(state: AgentState):

    user_question = state["messages"][-1].content

    planner_prompt = f"""
You are a planning agent.

Analyze the user's research question and create a clear step-by-step plan.

Available tools:
1. web_search - Search the web for current information.
2. rag_search - Search internal documents.
3. calculator - Perform mathematical calculations.

User question:
{user_question}

Return only a numbered step-by-step plan.
"""

    response = llm.invoke(planner_prompt)

    plan = [
        line.strip()
        for line in response.content.split("\n")
        if line.strip()
    ]

    print("\n[Planner]:")
    for step in plan:
        print(step)

    return {
        "plan": plan
    }


# ============================================================
# 6. DEFINE ASSISTANT NODE
# ============================================================

def assistant(state: AgentState):

    plan = state.get("plan", [])

    plan_text = "\n".join(plan)

    system_message = SystemMessage(
        content=f"""
You are an AI Research Assistant.

Follow the planner's plan when answering the user's question.

Plan:
{plan_text}

Use the available tools when necessary.
Do not claim that you used a tool unless you actually called it.
After collecting enough information, provide the final answer.
"""
    )

    response = llm_with_tools.invoke(
        [system_message] + state["messages"]
    )

    return {
        "messages": [response]
    }

# ============================================================
# 7. CREATE TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 8. BUILD GRAPH
# ============================================================

builder = StateGraph(AgentState)

builder.add_node(
    "planner",
    planner
)

builder.add_node(
    "assistant",
    assistant
)

builder.add_node(
    "tools",
    tool_node
)

builder.set_entry_point(
    "planner"
)

builder.add_edge(
    "planner",
    "assistant"
)

builder.add_conditional_edges(
    "assistant",
    tools_condition
)

builder.add_edge(
    "tools",
    "assistant"
)


# ============================================================
# 9. ADD SHORT-TERM MEMORY
# ============================================================

memory = InMemorySaver()

graph = builder.compile(
    checkpointer=memory
)


# ============================================================
# 10. EXECUTION LOOP
# ============================================================

def run_researcher():

    print("--- Welcome to your Agentic Research Assistant ---")

    # Same thread_id = same conversation memory
    config = {
        "configurable": {
            "thread_id": "research-session-1"
        }
    }

    while True:

        user_input = input(
            "\nAsk Research Question (type 'exit'): "
        )

        if user_input.lower() in ["exit", "quit"]:
            break

        for event in graph.stream(
            {
                "messages": [
                    ("user", user_input)
                ]
            },
            config=config
        ):

            for node_name, value in event.items():

                # Planner node doesn't return messages
                if "messages" not in value:
                    continue

                last_msg = value["messages"][-1]

                # Agent decided to use a tool
                if (
                    hasattr(last_msg, "tool_calls")
                    and last_msg.tool_calls
                ):

                    tool_names = [
                        tc["name"]
                        for tc in last_msg.tool_calls
                    ]

                    print(
                        f"\n[Agent]: Thinking... "
                        f"I need to use tools: {tool_names}"
                    )

                # Tool execution completed
                elif last_msg.type == "tool":

                    print(
                        f"[System]: Tool "
                        f"'{last_msg.name}' completed."
                    )

                # Final LLM response
                else:

                   print("\n[Final Answer]:")

                   content = last_msg.content

                   if isinstance(content, str):
                    print(content)

                   elif isinstance(content, list):
                    for block in content:
                     if isinstance(block, dict) and block.get("type") == "text":
                      print(block.get("text", ""))

if __name__ == "__main__":
    run_researcher()
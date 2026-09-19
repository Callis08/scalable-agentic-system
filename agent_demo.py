import os
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# 1. Define Mock PayPal API Tools
def create_invoice(amount: float, recipient: str) -> str:
    """Creates and sends an invoice for a specific amount to a recipient."""
    return f"Successfully created invoice for ${amount} to {recipient}."

def get_sales_volume(timeframe: str) -> str:
    """Retrieves total sales volume and revenue reports for a given timeframe like last month."""
    return f"Total sales volume for {timeframe}: $12,450.00."

def check_dispute(user_id: str) -> str:
    """Checks if there is an active payment dispute or chargeback open for a specific user ID."""
    return f"Dispute status for {user_id}: No active disputes found."

# Tool Registry mapping names to functions and descriptions
TOOL_REGISTRY = {
    "create_invoice": {
        "func": create_invoice,
        "description": "Creates and sends an invoice for a specific amount to a recipient."
    },
    "get_sales_volume": {
        "func": get_sales_volume,
        "description": "Retrieves total sales volume and revenue reports for a given timeframe."
    },
    "check_dispute": {
        "func": check_dispute,
        "description": "Checks if there is an active payment dispute open for a specific user ID."
    }
}

# 2. Dynamic Tool Selector (Simulating Vector/Semantic Retrieval)
def select_relevant_tools(query: str) -> List[str]:
    """
    In production, this queries a vector database (like Chroma or FAISS) 
    storing tool descriptions. Here we use basic keyword matching for simulation.
    """
    query_lower = query.lower()
    selected = []
    if "invoice" in query_lower or "send" in query_lower:
        selected.append("create_invoice")
    if "sales" in query_lower or "volume" in query_lower or "month" in query_lower:
        selected.append("get_sales_volume")
    if "dispute" in query_lower or "user" in query_lower:
        selected.append("check_dispute")
    
    # Fallback if no specific keyword matches: return all or a default set
    if not selected:
        selected = list(TOOL_REGISTRY.keys())
    return selected

# 3. Define LangGraph State
class AgentState(TypedDict):
    messages: List[str]
    user_query: str
    selected_tools: List[str]
    final_output: str

# 4. Define Graph Nodes
def router_node(state: AgentState) -> AgentState:
    query = state["user_query"]
    print(f"\n[Router] Analyzing query: '{query}'")
    
    # Dynamically retrieve tools to prevent context bloat
    relevant_keys = select_relevant_tools(query)
    print(f"[Router] Filtered down to {len(relevant_keys)} relevant tools out of hundreds: {relevant_keys}")
    
    state["selected_tools"] = relevant_keys
    return state

def execution_node(state: AgentState) -> AgentState:
    query = state["user_query"]
    tools_to_use = state["selected_tools"]
    
    print(f"[Executor] Simulating execution with tools: {tools_to_use}")
    
    # Safely simulate tool output and responses locally
    if "create_invoice" in tools_to_use and "invoice" in query.lower():
        output = "Successfully executed 'create_invoice': Created invoice for $50 to user_abc."
    elif "get_sales_volume" in tools_to_use:
        output = "Successfully executed 'get_sales_volume': Total sales volume for last month is $12,450.00."
    elif "check_dispute" in tools_to_use:
        output = "Successfully executed 'check_dispute': No active disputes found for user_123."
    else:
        output = f"Processed query using tools {tools_to_use} successfully."
        
    state["final_output"] = output
    return state

# 5. Build the LangGraph Workflow
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("executor", execution_node)

workflow.set_entry_point("router")
workflow.add_edge("router", "executor")
workflow.add_edge("executor", END)

app = workflow.compile()

# 6. Test the System with User Scenarios
if __name__ == "__main__":
    test_queries = [
        "Send an invoice for $50 to user_abc",
        "What was my total sales volume last month?",
        "Is there a dispute open from user_123?"
    ]
    
    for q in test_queries:
        print("--------------------------------------------------")
        result = app.invoke({"user_query": q, "messages": [], "selected_tools": [], "final_output": ""})
        print(f"User Query: {q}")
        print(f"Agent Output:\n{result['final_output']}")
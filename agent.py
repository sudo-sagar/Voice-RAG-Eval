import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. Configure your local Ollama model (acting as OpenAI-compatible)
# Make sure 'ollama serve' is running in the background
llm = ChatOpenAI(
    model="qwen2.5-coder:1.5b", # Using the faster model we tested
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1",
    temperature=0
)

# 2. Define the State
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    route: str

# 3. Define the Nodes
def router_node(state: AgentState):
    """Decides if the question is a product query or off-topic."""
    prompt = f"""You are a router. If the user's question is about Frames (health, navigation, translation), reply 'rag'. If it is random chatter, reply 'fallback'.
    Question: {state['question']}
    Reply with ONLY one word: 'rag' or 'fallback'."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    route = response.content.strip().lower()
    return {"route": route if route in ["rag", "fallback"] else "rag"}

def rag_node(state: AgentState):
    """Retrieves context and answers in Hinglish."""
    # In a real app, you would query a vector DB here. 
    # For this demo, we mock the retrieved context.
    #mock_context = "JioFrames supports health monitoring via built-in sensors for heart rate and steps."
    mock_context = "JioFrames supports health monitoring via built-in sensors that track heart rate and steps, and it syncs directly to the companion mobile app."
    system_prompt = """You are AI. You MUST answer in natural Hinglish (Hindi words in Latin script).
    Use ONLY the provided context. If the answer is not in the context, say 'Mujhe iska exact answer nahi pata' and do not guess.
    Keep your answer under 2 sentences."""
    
    user_prompt = f"Context: {mock_context}\n\nQuestion: {state['question']}"
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return {"answer": response.content, "context": mock_context}

def fallback_node(state: AgentState):
    """Graceful fallback for off-topic or low-confidence queries."""
    return {"answer": "Mujhe iska exact answer nahi pata. Main aapko human agent se connect kar raha hoon."}

# 4. Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("rag", rag_node)
workflow.add_node("fallback", fallback_node)

workflow.set_entry_point("router")

# Conditional routing
workflow.add_conditional_edges(
    "router",
    lambda x: x["route"],
    {"rag": "rag", "fallback": "fallback"}
)

workflow.add_edge("rag", END)
workflow.add_edge("fallback", END)

app = workflow.compile()

# 5. Test the Agent
if __name__ == "__main__":
    test_questions = [
        "Mera health monitor kaise kaam karta hai?", # Should route to RAG and answer in Hinglish
        "What is the capital of France?"             # Should route to Fallback
    ]
    
    for q in test_questions:
        print(f"\n--- User: {q} ---")
        result = app.invoke({"question": q, "context": "", "answer": "", "route": ""})
        print(f"Route: {result['route']}")
        print(f"AI: {result['answer']}")
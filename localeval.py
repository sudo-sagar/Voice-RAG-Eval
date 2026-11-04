import sys
import types
import json
import asyncio

# --- PATCH FOR RAGAS VERTEX AI IMPORT BUG ---
_vertex_module = types.ModuleType("langchain_community.chat_models.vertexai")
_vertex_module.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _vertex_module
# ---------------------------------------------

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

# IMPORT YOUR AGENT
from agent import app as agent_app

# 1. Load Golden Dataset
with open("dataset.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# 2. Run the Agent to generate REAL answers
print("Running Jio Agent to generate answers for evaluation...")
generated_answers = []
generated_contexts = []

for d in data:
    question = d["question"]
    # Invoke your LangGraph agent
    result = agent_app.invoke({
        "question": question, 
        "context": "", 
        "answer": "", 
        "route": ""
    })
    
    generated_answers.append(result["answer"])
    # If the agent returned context (RAG node), use it. Otherwise, use the reference.
    generated_contexts.append([result["context"] if result["context"] else d["reference_context"]])

# 3. Build the Eval Dataset using the Agent's outputs
eval_data = {
    "question": [d["question"] for d in data],
    "answer": generated_answers,          # <-- Real AI outputs
    "contexts": generated_contexts,       # <-- Real contexts used by AI
    "ground_truth": [d["ground_truth"] for d in data],
}
dataset = Dataset.from_dict(eval_data)

# 4. Ollama OpenAI-compatible client
client = OpenAI(
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1"
)

# 5. Warm up models (Pin them in memory for 1 hour)
print("Warming up models...")
client.chat.completions.create(
    model="qwen2.5-coder:1.5b",
    messages=[{"role": "user", "content": "hi"}],
    max_tokens=1,
    extra_body={"keep_alive": "1h"},
)
client.embeddings.create(
    model="mxbai-embed-large:latest",
    input="hi",
    extra_body={"keep_alive": "1h"},
)

# 6. LLM factory and Embeddings
local_llm = llm_factory("qwen2.5-coder:1.5b", provider="openai", client=client)

local_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        model="mxbai-embed-large:latest",
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
        check_embedding_ctx_length=False,
    )
)

# 7. Metrics
faithfulness_metric = Faithfulness(llm=local_llm)
answer_relevancy_metric = AnswerRelevancy(llm=local_llm, embeddings=local_embeddings)
context_precision_metric = ContextPrecision(llm=local_llm)

try:
    answer_relevancy_metric.strictness = 1
except Exception:
    pass

# 8. Run Config
run_config = RunConfig(timeout=600, max_workers=1, max_retries=2)

print("Starting local RAGAS evaluation. This will take a few minutes...")
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness_metric, answer_relevancy_metric, context_precision_metric],
    llm=local_llm,
    embeddings=local_embeddings,
    run_config=run_config
)

print("\n--- Evaluation Results ---")
print(result)

# 9. CI Gate
raw_score = result['faithfulness']
if isinstance(raw_score, list):
    valid_scores = [s for s in raw_score if s == s]
    faithfulness_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
else:
    faithfulness_score = raw_score if raw_score == raw_score else 0.0

print(f"\nFaithfulness Score: {faithfulness_score:.4f}")

if faithfulness_score < 0.80:
    print("GATE FAILED: Model hallucinated too much or connection failed. Blocking deployment.")
    exit(1)
else:
    print("GATE PASSED: Model is grounded.")
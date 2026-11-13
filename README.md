# Voice-RAG-Eval: CI-Gated Evaluation for JioFrames AI

## The Goal
To build a production-grade evaluation pipeline for the JioAI assistant, ensuring that multilingual voice responses (Hinglish) are grounded in facts and do not hallucinate. 

This project addresses the core JioFrames AI requirements: fine-tuning multilingual speech models, building agentic RAG workflows, and establishing robust CI-gated evaluation frameworks.

## The Stack
*   **ASR Fine-Tuning:** Whisper-small (LoRA/PEFT)
*   **Agentic Brain:** LangGraph (Hinglish routing, RAG, and graceful fallback)
*   **Evaluation:** RAGAS (Faithfulness, Answer Relevancy, Context Precision)
*   **Inference:** Local Ollama (Qwen 1.5B) for zero cloud dependencies.

---

## Phase 1: Multilingual ASR Fine-Tuning (Completed)

To address the JioFrames requirement for low Word Error Rates (WER) on Indian languages, I fine-tuned `openai/whisper-small` using **LoRA (PEFT)** on a Hindi audio dataset.

*   **Parameter Efficiency:** By targeting only the attention projection layers (`q_proj`, `v_proj`), I reduced trainable parameters to just **1.44%** (3.5M out of 245M parameters), allowing this to run in minutes on a free Colab T4 GPU.
*   **Results:** Achieved a **29.2% Word Error Rate (WER)** on a 100-sample evaluation set, a massive improvement over Whisper's zero-shot Hindi baseline (~50-60%).
*   **Why this matters:** This proves the exact LoRA methodology required to adapt open-weight ASR models to Jio's specific Hinglish acoustic environment.

--- LoRA Setup Complete ---
trainable params: 3,538,944 || all params: 245,273,856 || trainable%: 1.4429

--- Evaluation Results ---
{'eval_loss': 0.3395, 'eval_wer': 29.2098}

## Phase 2: Agentic Hinglish RAG (LangGraph)
To handle code-switching and multi-turn dialogue, I built a LangGraph state machine with three nodes: a Router (intent detection), a RAG node (grounded generation), and a Fallback node (graceful refusal).

The Implementation (agent.py)

## Phase 3: The RAGAS CI Gate (The Core FDE Value)
To prove the agent works, I built a RAGAS evaluation pipeline. It imports the compiled LangGraph agent, runs it against a Hinglish golden dataset, and blocks deployment if Faithfulness drops below 0.80.

The Implementation (localeval.py)

## The "Demo vs. Real World" Validation
To prove the gate actually works, I ran it three times:

Run	Pipeline	Faithfulness	Gate Status
1. Mocked	Ground truth passed as answer	0.90	✅ PASSED
2. Real Agent (Weak Context)	LangGraph agent with insufficient context	0.20	❌ FAILED
3. Real Agent (Fixed Context)	LangGraph agent with correct context	0.80	✅ PASSED
The Key Finding: When I wired the actual LangGraph agent into the gate, the score collapsed from 0.90 to 0.20. The gate successfully blocked the deployment because the agent was hallucinating beyond the retrieved context. After I fixed the retrieval logic, the gate passed. This proves the harness works as a real production safeguard.

## Bonus: Cost & Latency Tradeoff Analysis
To ensure the CI gate was viable, I benchmarked local models.

Metric	Qwen 1.5B	Qwen 3B
Faithfulness	0.90 (PASS)	0.70 (FAIL)
Latency	6m 35s	52m 39s
Finding: The smaller 1.5B model was 8x faster and passed the faithfulness gate, while the 3B model failed due to hallucinations. For local CI evaluation, smaller grounded models are better than larger hallucinators.
  
## Architecture

flowchart TD
    A[User Audio] -->|Whisper + LoRA| B(Hinglish Text)
    B --> C{LangGraph Router}
    C -->|On-Topic| D[RAG Node]
    C -->|Off-Topic| E[Fallback Node]
    D --> F[Grounded Hinglish Answer]
    E --> F
    F --> G{RAGAS CI Gate}
    G -->|Faithfulness < 0.80| H[Block Deployment]
    G -->|Faithfulness >= 0.80| I[Deploy]


# How to Run Locally
Install dependencies:
pip install -r requirements.txt

Start your local Ollama server:
ollama serve

Run the RAGAS Evaluation Gate:
python localeval.py

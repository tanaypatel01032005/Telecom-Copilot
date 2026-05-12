# 📡 Telecom AI Copilot: Agentic RAG Pipeline

A state-of-the-art AI Copilot designed for Telecom NOC (Network Operations Center) engineers and customer support agents. This system leverages **Agentic RAG**, **Fine-tuned LLMs**, and **Hybrid Retrieval** to provide grounded, tool-augmented technical support.

---

## 🚀 Key Features

*   **Hybrid Semantic Search**: Combines Dense (BGE-768) and Keyword (BM25) search with Reciprocal Rank Fusion (RRF).
*   **Agentic Tool Use**: ReAct-style reasoning to check **Live Network Outages**, create **Support Tickets**, and lookup authoritative **Regulatory Policies**.
*   **Fine-tuned Generator**: Llama-3-8B fine-tuned via **DoRA** (Weight-Decomposed Low-Rank Adaptation) for strict citation adherence and technical domain expertise.
*   **Authoritative Grounding**: Every response includes `[SOURCE: doc_id]` citations, with a **98%+ Groundedness score**.
*   **14-Metric Evaluation Suite**: Includes Retrieval (Recall@k, MRR), Generation (BERTScore, Groundedness), and Novel Telecom metrics (OARR, GEA).

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User Query]) --> Orchestrator[Telecom Copilot Orchestrator]
    Orchestrator --> ToolPolicy{Tool Policy / Routing}
    
    subgraph Retrieval Layer
        ToolPolicy --> HybridSearch[Hybrid Search: Dense + BM25]
        HybridSearch --> Reranker[Cross-Encoder Reranker]
    end
    
    subgraph Knowledge & Tools
        Reranker --> KB[(Knowledge Base: 25k Passages)]
        ToolPolicy --> NetworkAPI[Live Network Status API]
        ToolPolicy --> TicketSys[Automated Ticketing System]
    end
    
    KB --> Generator[Fine-tuned Llama-3-8B]
    NetworkAPI --> Generator
    Generator --> Response([Grounded Response + Citations])
```

---

## 🛠️ Setup & Installation

### 1. Environment Configuration
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

### 2. Initialization Sequence
To build the system from scratch, run the files in this order:

| Step | Command | Description |
| :--- | :--- | :--- |
| 1 | `python -m src.ingestion.kb_builder` | Builds the technical knowledge base. |
| 2 | `python -m src.retrieval.faiss_indexer --label finetuned` | Builds the FAISS vector index. |
| 3 | `python -m src.retrieval.train_retriever` | (Optional) Fine-tunes the BGE retriever. |
| 4 | `python -m src.retrieval.reranker --train` | (Optional) Trains the Cross-Encoder. |
| 5 | `streamlit run app/app.py` | **Launch the User Interface.** |

---

## 📊 Performance Benchmarks

The system was benchmarked against a Baseline (BM25 + Flan-T5) on 205 industry-standard test cases.

| Metric | Baseline | **Full System** | Delta |
| :--- | :--- | :--- | :--- |
| **Outage Awareness (OARR)** | 0.0% | **100.0%** | ⭐ Huge Win |
| **Groundedness** | 86.0% | **98.3%** | +12.3% |
| **Hallucination Rate** | 13.9% | **1.6%** | -88% Reduction |
| **Citation Recall@1** | 0.0% | **80.0%** | New Feature |

---

## 📂 Project Structure

*   `app/`: Streamlit chat interface and UI logic.
*   `src/retrieval/`: Hybrid search, FAISS indexing, and Cross-Encoder reranking.
*   `src/pipeline/`: Core ReAct orchestration and tool-calling policy.
*   `src/generation/`: DoRA fine-tuning scripts for the Llama-3 generator.
*   `src/evaluation/`: Automated 14-metric benchmarking harness.
*   `data/`: Raw technical documents, processed KB, and FAISS artifacts.

---

## 📝 License
This project is developed for the Telecom AI Copilot Technical Challenge. All rights reserved.
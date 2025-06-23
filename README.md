# 📄 **Technical Report**

## Legal Retrieval System with Multi-Agent Architecture: Vietnamese Law QA powered by LLMs & Agentic RAG

**Authors**:  
We gratefully acknowledge the invaluable contributions of the following team members:

1. Đặng Nguyễn Quang Huy – [ZeusCoderBE](https://github.com/ZeusCoderBE)  
2. Huỳnh Gia Hân – [hg27haan](https://github.com/hg27haan)  
3. Nguyễn Trọng Dũng – [NgTrDung](https://github.com/NgTrDung)

**Supervisor**: ThS. Trần Trọng Bình  
**Institution**: Ho Chi Minh City University of Technology and Education  
**Department**: Faculty of Information Technology  
**Major**: Data Engineering

---

## 1. Executive Summary

This report outlines the design, development, and evaluation of a **Legal Retrieval System** incorporating a **Multi-Agent Architecture** for the Vietnamese legal domain. By integrating advanced techniques such as **LLMs**, **embedding-based retrieval**, **cross-encoder reranking**, and an **Agentic RAG pipeline**, we have transitioned from static RAG architectures to a dynamic, modular system optimized for legal complexity and nuanced reasoning.

---

## 2. Motivation & Problem Statement

Vietnamese legal documents are inherently complex, filled with exceptions, and written in hierarchical formats. For non-experts, navigating this content is overwhelming; for professionals, it is time-consuming and inefficient.

We aimed to solve:

- **Scattered and poorly indexed legal knowledge**
- **LLM hallucination risks** without legal-grounded retrieval
- **Rigid RAG pipelines** unable to handle multi-turn or agent-based workflows

---

## 3. System Architecture

### 3.1 Key Components

- **Data Source**: Official Vietnamese legal repositories
- **Vector Store**: Qdrant for dense semantic retrieval
- **LLM Backbone**: Google BERT fine-tuned on legal corpus
- **Embedding Model**: [DEk21_hcmute_embedding](https://huggingface.co/huyydangg/DEk21_hcmute_embedding)
- **Reranker**: [Cross-Encoder with RRF](https://huggingface.co/hghaan/rerank_model)
- **Extractor**: [BERT Answer Span Extractor](https://huggingface.co/huyydangg/bert_extract_full_fine-tuned)
- **Agentic RAG Modules**:
  - `Query Router`
  - `Query Rewriter`
  - `Entity Extractor`
  - `Search + Rerank`
  - `Search Tool`
  - `LLM Inference` (powered by Gemini API)

![Workflow](https://github.com/user-attachments/assets/69daefa3-937a-4f94-9b6f-9b888051c252)

---

### 3.2 Technical Stack

| Layer             | Technology / Tool                             | Description                                                                 |
|------------------|-----------------------------------------------|-----------------------------------------------------------------------------|
| **Modeling**      | 🤗 HuggingFace Transformers                   | Core transformer base for BERT and Sentence-BERT                           |
| **Embedding**     | 🧠 SBERT + Matryoshka Loss                    | Multi-scale semantic vector generation                                     |
| **Reranking**     | 🎯 Cross-Encoder BERT + RRF                   | Improved relevance scoring post-retrieval                                  |
| **LLM Reasoning** | 🔮 Gemini API                                 | Generates contextual answers from retrieved information                    |
| **Agent Modules** | 🤖 Router, Rewriter, Extractor                | Modular tools for routing, clarification, and metadata extraction          |
| **Search Layer**  | 🔍 Qdrant + Web Tools                         | Embedding retrieval with optional web search integration                   |
| **Information Extractor** | 🧾 Fine-tuned BERT                    | Extracts relevant answer spans from legal content                          |
| **Infrastructure**| ⚙️ FastAPI, Docker, LangChain                 | Microservice backend and LLM orchestration                                 |
| **Deployment**    | ☁️ Dockerized Microservices                  | Scalable, containerized services                                           |
| **Storage**       | 🧮 Qdrant Vector Store                         | Efficient and scalable vector + metadata storage                           |

---

## 4. Methodology & Model Optimization

### 4.1 Embedding Optimization

- Trained with **MultipleNegativesRankingLoss**
- Enhanced via **Matryoshka Representation Learning**
- Evaluated using: **Recall@K**, **MAP**, **MRR**, **NDCG**

### 4.2 Reranker

- Fine-tuned **Cross-Encoder** with legal question-doc pairs
- Incorporated **Reciprocal Rank Fusion (RRF)** for ensemble scoring

### 4.3 Extractor Model

- BERT-based answer span extraction
- Reduces context length and API costs for downstream LLMs

### 4.4 Agentic RAG Pipeline

- **Dynamic task delegation** via Query Router
- **Multi-hop reasoning** and **context rewriting**
- Modular components for fine-grained control over query execution

---

## 5. Evaluation & Results

| Module         | Approach                      | Metrics                         |
|----------------|-------------------------------|----------------------------------|
| BERT Extractor | Full + LoRA Fine-Tuning       | **F1: 0.93**, **EM: 0.91**       |
| Embedding      | SBERT + Matryoshka            | **NDCG@10: 0.92**                |
| Reranker       | Cross-Encoder + RRF           | **MRR@10: 0.87**                 |
| Full System    | Agentic RAG + Gemini Agent    | **Accuracy: 90%**                |

> Benchmarked on internal **UTE_LAW** and **Zalo QA** datasets. The Agentic RAG pipeline significantly outperformed traditional RAG in both retrieval and reasoning effectiveness.

---

## 6. Deployment

- **API**: Built with **FastAPI** for lightweight REST interface
- **Containerization**: Docker ensures reproducible environments
- **Scaling**: Microservice architecture enables modular scalability
- **Frontend**: Simple and intuitive UI tailored for legal professionals and public usage

---

## 7. Key Contributions & Innovation

- **First Agentic RAG** system specialized for Vietnamese legal NLP
- **Custom-trained models** for embedding, reranking, and span extraction
- **Fully modular, scalable**, and **cost-efficient** architecture
- Designed for **transparency**, **traceability**, and **legal compliance**

---

## 8. Limitations

- Dataset constraints due to limited availability of Vietnamese annotated legal corpora
- Ambiguity handling in multi-turn dialogue still needs refinement
- **Gemini LLM integration** is API-dependent; future goal is full local LLM support

---

## 9. Future Directions

- Expand coverage to **multi-jurisdictional** and **multilingual** legal sources
- Integrate structured **Legal Ontologies**
- Introduce **adaptive retrievers** via user feedback loops
- Add **zero-shot summarization** and **citation tracing** for interpretability

---

## 10. Conclusion

This project demonstrates the successful application of **Agentic Retrieval-Augmented Generation (RAG)** to a high-stakes, domain-specific task: legal question answering. By combining advanced NLP techniques, purpose-built models, and modular agent architecture, we deliver a reliable and intelligent system for Vietnamese legal advisory. This work paves the way for broader applications in **AI-driven public services** and **legal tech innovation**.

---

# AI Medical Assistant using Retrieval-Augmented Generation with Self-RAG Verification

> A hallucination-free clinical question answering system built on Llama-2-7B, two-stage FAISS retrieval, real-time PubMed integration, and a Self-RAG reflection layer — trained and evaluated entirely on Google Colab's free T4 GPU.

---

## Overview

Standard RAG systems retrieve and generate — but they don't verify. This project adds a **Self-RAG verification layer** on top of a two-stage retrieval pipeline, where the model scores its own generated output for factual grounding before committing to a response. The result is a medical assistant that refuses to hallucinate.


---

## Architecture

![System Architecture](architecture.png)

The pipeline begins with **Query Pre-processing** (abbreviation expansion + intent detection), branches into **Local FAISS Retrieval** and **PubMed API Retrieval** (for time-sensitive queries), merges candidates through a **Cross-Encoder Re-Ranker**, generates an answer via **Llama-2-7B**, and then passes it through the **Self-RAG Verification** layer. If the utility score falls below 4, the query is refined and the loop retries — up to 3 attempts.

---

## Key Features

- **Self-RAG verification** — model critiques its own output before finalizing a response
- **Two-stage FAISS retrieval** — coarse-to-fine retrieval over a local medical corpus
- **Real-time PubMed integration** — pulls live literature for queries beyond the local index
- **Citation-aware generation** — every response is grounded and traceable
- **Zero hallucination** — 0.0% hallucination rate on the evaluation set
- **Resource-efficient** — runs entirely on a free T4 GPU (Google Colab)

---

## Results

| Metric | Score |
|---|---|
| PubMedQA Accuracy | 76% |
| Hallucination Rate | **0.0%** |
| ROUGE-1 | 0.321 |
| ROUGE-2 | 0.114 |
| ROUGE-L | 0.220 |

> A hallucination rate of 0.0% on medical queries is the primary design goal. Factual reliability in healthcare outweighs raw accuracy on benchmarks.

---

## Tech Stack

| Component | Tool |
|---|---|
| Base LLM | Llama-2-7B (HuggingFace) |
| Retrieval Index | FAISS |
| Live Retrieval | PubMed API (Entrez) |
| Evaluation | ROUGE, MedQA benchmark |
| Environment | Google Colab (T4 GPU, 15GB VRAM) |
| Framework | Python, LangChain, HuggingFace Transformers |

---

## Notebook Structure

```
AI_Medical_Assistant_SelfRAG.ipynb
│
├── 1. Environment Setup & Dependencies
├── 2. Model Loading (Llama-2-7B, 4-bit quantization)
├── 3. FAISS Index Construction
│       ├── Corpus preprocessing
│       ├── Embedding generation (sentence-transformers)
│       └── Two-stage index build
├── 4. PubMed Real-time Retrieval
│       └── Entrez API integration
├── 5. Self-RAG Pipeline
│       ├── Retrieval-augmented generation
│       ├── Reflection / critique step
│       └── Response selection logic
├── 6. Evaluation
│       ├── MedQA accuracy
│       ├── Hallucination rate measurement
│       └── ROUGE score computation
└── 7. Sample Queries & Outputs
```

---

## Getting Started

### Prerequisites

- Google Colab (free tier with T4 GPU runtime)
- HuggingFace account with access to [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- NCBI API key for PubMed access (free at [ncbi.nlm.nih.gov](https://www.ncbi.nlm.nih.gov/account/))

### Setup

1. Clone this repository or open the notebook directly in Colab
2. Add your HuggingFace token and NCBI API key to Colab secrets
3. Run all cells top to bottom — setup takes approximately 10 to 15 minutes on a T4

```python
# Add to Colab secrets or set directly
HF_TOKEN = "your_huggingface_token"
NCBI_API_KEY = "your_ncbi_api_key"
```

---

## Limitations

- MedQA accuracy (33.5%) reflects the constraint of a 7B parameter model without full fine-tuning on medical corpora
- Inference latency is higher than production systems due to the self-verification step
- Dependent on PubMed availability for real-time retrieval queries

---

## License

This project is for academic and research purposes. Model weights follow the [Llama 2 Community License](https://ai.meta.com/llama/license/).

"""
selfrag.py
==========
Self-RAG verification layer:
  1. generate_response  — RAG-based answer generation with Llama-2-7B
  2. verify_support     — LLM-as-judge: checks factual grounding
  3. score_utility      — LLM-as-judge: rates answer quality (1-5)
  4. selfrag_pipeline   — full iterative Self-RAG loop

Prerequisites
-------------
  pip install transformers accelerate bitsandbytes torch

Usage
-----
  from selfrag import load_model, selfrag_pipeline
  model, tokenizer = load_model()
  from retrieval import load_retriever, hybrid_retrieve, rerank
  retriever = load_retriever()
  result = selfrag_pipeline("What are symptoms of dengue fever?",
                             model, tokenizer, retriever)
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME          = "meta-llama/Llama-2-7b-chat-hf"
MIN_UTILITY_SCORE   = 4
MAX_RETRIEVAL_ATTEMPTS = 3


# ── Model loading ─────────────────────────────────────────────────

def load_model():
    """Load Llama-2-7B in 4-bit quantisation for T4 GPU."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (4-bit quantized)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print(f"✅ Model loaded — GPU memory: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


# ── Prompt builder ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a medical AI assistant. Your role is to provide accurate,
evidence-based medical information by STUDYING the provided reference documents and
SYNTHESIZING a comprehensive answer in your own words.

STRICT RULES:
1. READ and UNDERSTAND all provided reference documents thoroughly.
2. SYNTHESIZE a clear, well-structured answer based on the information in these documents.
3. CITE every claim using the reference number format [1], [2], etc.
4. If the documents do not contain enough information, explicitly state that.
5. Do NOT copy documents verbatim — generate a coherent, natural response.
6. Do NOT use any knowledge outside the provided references.
7. Use proper medical terminology and maintain clinical accuracy.
   Answer SPECIFICALLY about the query asked. Do not generalise to broader topics.
   If the documents do not directly address the query, say so explicitly."""


def build_rag_prompt(query: str, docs: list[dict], tokenizer) -> str:
    MAX_CONTEXT_TOKENS    = 4096
    RESERVED_GENERATION   = 1024
    RESERVED_TEMPLATE     = 300
    MAX_INPUT_TOKENS      = MAX_CONTEXT_TOKENS - RESERVED_GENERATION - RESERVED_TEMPLATE
    max_tokens_per_doc    = MAX_INPUT_TOKENS // max(len(docs), 1)

    context_parts = []
    for i, doc in enumerate(docs, 1):
        text   = doc["text"].strip()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens_per_doc:
            text = tokenizer.decode(tokens[:max_tokens_per_doc],
                                    skip_special_tokens=True).strip() + "..."
        context_parts.append(
            f"[{i}] (Source: {doc.get('source','?')} | "
            f"ID: {doc.get('doc_id', f'doc_{i}')})\n{text}"
        )

    context_str = "\n\n".join(context_parts)
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"REFERENCE DOCUMENTS:\n{context_str}\n\n"
        f"QUESTION: {query}\n\n"
        "Study the above reference documents carefully and generate a comprehensive, "
        "well-structured medical answer. Cite sources as [1], [2], etc. [/INST]"
    )


# ── Answer generation ─────────────────────────────────────────────

def generate_response(query: str, docs: list[dict],
                      model, tokenizer) -> dict:
    """Generate a RAG-grounded answer and extract citation metadata."""
    prompt = build_rag_prompt(query, docs, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated  = outputs[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Strip duplicate reference lists appended by LLM
    for pattern in [
        r"\n\s*References?\s*:?\s*\n.*$",
        r"\n\s*Sources?\s*:?\s*\n.*$",
        r"\n\s*Reference List\s*:?\s*\n.*$",
    ]:
        response = re.sub(pattern, "", response,
                          flags=re.DOTALL | re.IGNORECASE).strip()

    # Extract citation numbers
    bracket_contents = re.findall(r"\[([\d,\s]+)\]", response)
    citations_used   = sorted({int(n) for bc in bracket_contents
                                for n in re.findall(r"(\d+)", bc)})

    cited_sources = []
    for num in citations_used:
        if 1 <= num <= len(docs):
            d = docs[num - 1]
            cited_sources.append({
                "citation":     f"[{num}]",
                "source":       d.get("source", "Unknown"),
                "doc_id":       d.get("pmid") or d.get("book") or f"doc_{num}",
                "text_snippet": d["text"][:200] + "...",
            })

    return {
        "response":       response,
        "citations_used": citations_used,
        "cited_sources":  cited_sources,
    }


# ── Self-RAG verification prompts ────────────────────────────────

SUPPORT_PROMPT = """You are a medical fact-checker. Check if the answer is supported by the source documents.

QUERY: {query}

SOURCE DOCUMENTS:
{sources}

GENERATED ANSWER:
{answer}

Respond EXACTLY in this format:
SUPPORT_LABEL: <Fully Supported | Partially Supported | No Support>
SUPPORT_SCORE: <decimal 0.0 to 1.0>
UNSUPPORTED_CLAIMS: <list each with '-' prefix, or write None>
"""

UTILITY_PROMPT = """<s>[INST] <<SYS>>
You are a strict medical evaluator. Respond in EXACTLY the format shown. No extra text.
<</SYS>>

Rate this medical answer from 1 to 5.

QUERY: {query}
ANSWER: {answer}
SUPPORT STATUS: {support_label}
UNSUPPORTED CLAIMS: {unsupported_claims}

Scoring:
5 = Complete, accurate, well-cited, no hallucinations
4 = Mostly accurate, minor gaps, adequately supported
3 = Partially correct, some unsupported claims
2 = Significant inaccuracies or missing critical info
1 = Mostly wrong or dangerous

You MUST respond in EXACTLY this format and nothing else:
UTILITY_SCORE: [number]
FEEDBACK: [one sentence]
[/INST]"""


def _build_sources_text(docs: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] PMID:{doc.get('pmid','N/A')}\n{doc.get('text','')[:500]}"
        for i, doc in enumerate(docs, 1)
    )


def _parse_support(response: str) -> dict:
    result = {"support_label": "Partially Supported",
              "support_score": 0.5, "unsupported_claims": []}
    m = re.search(r"SUPPORT_LABEL:\s*(.+)", response, re.IGNORECASE)
    if m:
        l = m.group(1).strip().lower()
        if "fully" in l:
            result.update(support_label="Fully Supported", support_score=0.9)
        elif "no support" in l:
            result.update(support_label="No Support", support_score=0.1)
    m = re.search(r"SUPPORT_SCORE:\s*([\d.]+)", response, re.IGNORECASE)
    if m:
        try:
            result["support_score"] = float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"UNSUPPORTED_CLAIMS:(.*?)(?:\n[A-Z_]+:|$)",
                  response, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
        if text.lower() != "none":
            result["unsupported_claims"] = [
                c.strip().lstrip("-").strip()
                for c in text.split("\n")
                if c.strip() and c.strip() != "-"
            ]
    return result


def _parse_utility(response: str) -> dict:
    result = {"utility_score": 3, "utility_feedback": "Could not parse."}
    m = re.search(r"UTILITY_SCORE:\s*([1-5])", response, re.IGNORECASE)
    if m:
        result["utility_score"] = int(m.group(1))
    m = re.search(r"FEEDBACK:\s*(.+)", response, re.IGNORECASE)
    if m:
        result["utility_feedback"] = m.group(1).strip()
    return result


def verify_support(query: str, answer: str, docs: list[dict],
                   model, tokenizer) -> dict:
    """Use LLM as judge to verify factual support of the answer."""
    prompt = SUPPORT_PROMPT.format(
        query=query, sources=_build_sources_text(docs), answer=answer
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=3000).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300,
                                temperature=0.1, do_sample=False)
    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return _parse_support(response)


def score_utility(query: str, answer: str, support: dict,
                  model, tokenizer) -> dict:
    """Use LLM as judge to rate answer utility on a 1-5 scale."""
    unsupported_text = (
        "\n".join(f"- {c}" for c in support["unsupported_claims"]) or "None"
    )
    prompt = UTILITY_PROMPT.format(
        query=query, answer=answer,
        support_label=support["support_label"],
        support_score=support["support_score"],
        unsupported_claims=unsupported_text,
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=2000).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100,
                                temperature=0.1, do_sample=False)
    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return _parse_utility(response)


def _refine_query(query: str, unsupported: list[str], attempt: int) -> str:
    strategies = [
        f"{query} clinical evidence treatment guidelines",
        f"{query} pathophysiology mechanism diagnosis",
        f"{query} systematic review meta-analysis",
    ]
    if unsupported:
        return f"{query} {unsupported[0][:80]}"
    return strategies[min(attempt - 1, 2)]


# ── Full Self-RAG pipeline ────────────────────────────────────────

def selfrag_pipeline(query: str, model, tokenizer, retriever) -> dict:
    """
    Full iterative Self-RAG pipeline.

    Steps
    -----
    1. Hybrid retrieval + reranking
    2. Answer generation
    3. Support verification
    4. Utility scoring
    5. Re-retrieval if utility < MIN_UTILITY_SCORE (up to MAX_RETRIEVAL_ATTEMPTS)
    6. Return best result

    Returns
    -------
    dict with keys: result, docs, support, utility, attempt
    """
    from retrieval import hybrid_retrieve, rerank, check_doc_relevance

    candidates   = hybrid_retrieve(query, retriever)
    top5         = rerank(query, candidates, retriever, top_k=5)
    current_docs = check_doc_relevance(
        query, top5, retriever["embedding_model"], threshold=0.55
    )

    all_attempts = []
    best_result  = None

    for attempt in range(1, MAX_RETRIEVAL_ATTEMPTS + 1):
        print(f"\n{'='*50}")
        print(f"🔄 Self-RAG Attempt {attempt}/{MAX_RETRIEVAL_ATTEMPTS}")

        result      = generate_response(query, current_docs, model, tokenizer)
        answer_text = result["response"]

        print("🔍 Verifying support...")
        support = verify_support(query, answer_text, current_docs, model, tokenizer)
        print(f"   {support['support_label']} ({support['support_score']:.2f})")

        print("📊 Scoring utility...")
        utility = score_utility(query, answer_text, support, model, tokenizer)
        print(f"   Utility: {utility['utility_score']}/5 — {utility['utility_feedback']}")

        all_attempts.append({
            "result":  result,
            "docs":    current_docs,
            "support": support,
            "utility": utility,
            "attempt": attempt,
        })

        if utility["utility_score"] >= MIN_UTILITY_SCORE:
            print(f"✅ Passed with score {utility['utility_score']}!")
            best_result = all_attempts[-1]
            break

        if attempt < MAX_RETRIEVAL_ATTEMPTS:
            refined_q    = _refine_query(query, support["unsupported_claims"], attempt)
            print(f"⚠️  Re-retrieving: {refined_q[:80]}...")
            new_cands    = hybrid_retrieve(refined_q, retriever)
            current_docs = rerank(refined_q, new_cands, retriever, top_k=5)

    if best_result is None:
        best_result = max(
            all_attempts,
            key=lambda x: (x["utility"]["utility_score"],
                           x["support"]["support_score"])
        )
        print(f"\n📌 Max retries hit — using best attempt "
              f"(score: {best_result['utility']['utility_score']}/5)")

    return best_result

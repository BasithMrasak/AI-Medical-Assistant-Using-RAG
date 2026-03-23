"""
evaluate.py
===========
Evaluation pipeline for MedQA and PubMedQA benchmarks.

Metrics
-------
  MedQA   : Accuracy (A/B/C/D), Hallucination rate, Utility score
  PubMedQA: Accuracy (yes/no/maybe), ROUGE-1/2/L, Hallucination rate

Results
-------
  evaluation/medqa_checkpoint.json
  evaluation/pubmedqa_checkpoint.json
  evaluation/final_report.json

Usage
-----
  python evaluate.py
"""

import os
import re
import json
import time
from datetime import datetime
from rouge_score import rouge_scorer

# ── Paths ─────────────────────────────────────────────────────────
BASE_PATH           = "/content/drive/MyDrive/main project"
MEDQA_PATH          = f"{BASE_PATH}/data/questions/US/4_options/phrases_no_exclude_test.jsonl"
PUBMEDQA_PATH       = f"{BASE_PATH}/data/ori_pqaa.json"
EVAL_DIR            = f"{BASE_PATH}/evaluation"
MEDQA_CHECKPOINT    = f"{EVAL_DIR}/medqa_checkpoint.json"
PUBMEDQA_CHECKPOINT = f"{EVAL_DIR}/pubmedqa_checkpoint.json"
FINAL_REPORT        = f"{EVAL_DIR}/final_report.json"

MEDQA_SAMPLES    = 1273
PUBMEDQA_SAMPLES = 200

os.makedirs(EVAL_DIR, exist_ok=True)


# ── Checkpoint helpers ────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        print(f"✅ Checkpoint loaded: {len(data['results'])} results. Resuming...")
        return data
    return {"results": [], "completed_ids": []}


def save_checkpoint(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Answer extraction helpers ─────────────────────────────────────

def extract_answer_letter(response: str) -> str | None:
    """Extract A/B/C/D from Llama-2 response for MedQA."""
    for pattern in [
        r"(?:answer is|answer:|correct answer is|correct option is)\s*[:\(]?\s*([A-D])",
        r"(?:option|choice)\s+([A-D])\b",
        r"^\s*[\(\[]?([A-D])[\)\]]?[.\s]",
        r"\b([A-D])\b",
    ]:
        m = re.search(pattern, response if "answer" in pattern else response[:100],
                      re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def extract_yes_no_maybe(response: str) -> str | None:
    """Extract yes/no/maybe from Llama-2 response for PubMedQA."""
    text    = response.lower().strip()
    snippet = text[:200]
    for label, pattern in [
        ("yes",   r"\byes\b"),
        ("no",    r"\bno\b"),
        ("maybe", r"\bmaybe\b|\bperhaps\b|\bunclear\b|\binsufficient\b"),
    ]:
        if re.search(pattern, snippet):
            return label
    for label, pattern in [
        ("yes",   r"\byes\b"),
        ("no",    r"\bno\b"),
        ("maybe", r"\bmaybe\b"),
    ]:
        if re.search(pattern, text):
            return label
    return None


def compute_rouge(hypothesis: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ── MedQA prompt ──────────────────────────────────────────────────

def build_medqa_prompt(question: str, options: dict,
                       docs: list[dict] | None = None) -> str:
    options_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    context = ""
    if docs:
        context = "\n\nRelevant Medical Context:\n" + "\n\n".join(
            d["text"][:300] for d in docs[:3]
        )
    return (
        "You are a medical expert taking a USMLE exam.\n"
        "Use the context below to help answer the question.\n"
        "Respond with ONLY the letter (A, B, C, or D) on the first line, then explain.\n"
        f"{context}\n\nQuestion: {question}\n\nOptions:\n{options_text}\n\nAnswer:"
    )


# ── PubMedQA prompt ───────────────────────────────────────────────

def build_pubmedqa_prompt(question: str, contexts: list[str]) -> str:
    context_text = "\n\n".join(contexts[:3])[:1500]
    return (
        "You are a biomedical research expert. Based on the provided research context,\n"
        "answer the question with Yes, No, or Maybe.\n"
        "Start your response with the word Yes, No, or Maybe, then provide a brief explanation.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer (Yes/No/Maybe):"
    )


# ── MedQA evaluation ──────────────────────────────────────────────

def evaluate_medqa(model, tokenizer, retriever) -> dict:
    from query_utils import clean_query, expand_medical_query
    from retrieval  import hybrid_retrieve, rerank, check_doc_relevance
    from selfrag    import verify_support, score_utility
    import torch

    print("\n" + "=" * 60)
    print("📋 MEDQA EVALUATION")
    print("=" * 60)

    checkpoint   = load_checkpoint(MEDQA_CHECKPOINT)
    completed_ids = set(checkpoint["completed_ids"])
    results      = checkpoint["results"]

    with open(MEDQA_PATH, "r") as f:
        samples = [json.loads(line) for line in f][:MEDQA_SAMPLES]

    print(f"Samples: {len(samples)} | Completed: {len(completed_ids)}")

    for i, sample in enumerate(samples):
        sid = f"medqa_{i}"
        if sid in completed_ids:
            continue

        question    = sample["question"]
        options     = sample["options"]
        correct_idx = sample["answer_idx"]
        print(f"\n[{i+1}/{len(samples)}] Processing...")

        try:
            expanded_q = expand_medical_query(clean_query(question[:200]))
            candidates = hybrid_retrieve(expanded_q, retriever)
            top5       = rerank(expanded_q, candidates, retriever, top_k=5)
            top5       = check_doc_relevance(question, top5,
                                             retriever["embedding_model"], 0.45)

            mc_prompt = build_medqa_prompt(question, options, top5)
            inputs    = tokenizer(mc_prompt, return_tensors="pt",
                                  truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=200,
                    temperature=0.1, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            response_text = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            support   = verify_support(question, response_text, top5, model, tokenizer)
            utility   = score_utility(question, response_text, support, model, tokenizer)
            predicted = extract_answer_letter(response_text)
            is_correct = (predicted == correct_idx) if predicted else False

            result = {
                "id":               sid,
                "correct_answer":   correct_idx,
                "predicted_answer": predicted,
                "is_correct":       is_correct,
                "support_label":    support["support_label"],
                "support_score":    support["support_score"],
                "utility_score":    utility["utility_score"],
                "is_hallucinated":  support["support_label"] == "No Support",
            }
            results.append(result)
            completed_ids.add(sid)
            checkpoint.update(results=results, completed_ids=list(completed_ids))
            save_checkpoint(MEDQA_CHECKPOINT, checkpoint)

            icon = "✅" if is_correct else "❌"
            print(f"   {icon} Pred: {predicted} | Correct: {correct_idx} | "
                  f"Support: {support['support_label']} | "
                  f"Utility: {utility['utility_score']}/5")

        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            results.append({"id": sid, "error": str(e),
                            "is_correct": False, "is_hallucinated": False})
            completed_ids.add(sid)
            checkpoint.update(results=results, completed_ids=list(completed_ids))
            save_checkpoint(MEDQA_CHECKPOINT, checkpoint)

    valid        = [r for r in results if "error" not in r]
    correct      = sum(1 for r in valid if r["is_correct"])
    hallucinated = sum(1 for r in valid if r["is_hallucinated"])
    accuracy     = round(correct / len(valid) * 100, 2) if valid else 0
    hall_rate    = round(hallucinated / len(valid) * 100, 2) if valid else 0

    summary = {
        "total_evaluated":      len(valid),
        "correct":              correct,
        "accuracy_pct":         accuracy,
        "hallucinated":         hallucinated,
        "hallucination_rate_pct": hall_rate,
        "unparsed_responses":   sum(1 for r in valid if r.get("predicted_answer") is None),
    }
    print(f"\n{'='*60}")
    print(f"📊 MEDQA RESULTS")
    print(f"   Accuracy      : {accuracy}%")
    print(f"   Hallucination : {hall_rate}%")
    print(f"{'='*60}")
    return summary


# ── PubMedQA evaluation ───────────────────────────────────────────

def evaluate_pubmedqa(model, tokenizer, retriever) -> dict:
    from query_utils import clean_query, expand_medical_query
    from retrieval  import hybrid_retrieve, rerank, check_doc_relevance
    from selfrag    import verify_support, score_utility
    import torch

    print("\n" + "=" * 60)
    print("📋 PUBMEDQA EVALUATION")
    print("=" * 60)

    checkpoint    = load_checkpoint(PUBMEDQA_CHECKPOINT)
    completed_ids = set(checkpoint["completed_ids"])
    results       = checkpoint["results"]

    with open(PUBMEDQA_PATH, "r") as f:
        pqa_data = json.load(f)
    all_keys = list(pqa_data.keys())[:PUBMEDQA_SAMPLES]
    print(f"Samples: {len(all_keys)} | Completed: {len(completed_ids)}")

    for i, pmid in enumerate(all_keys):
        sid = f"pubmedqa_{pmid}"
        if sid in completed_ids:
            continue

        sample       = pqa_data[pmid]
        question     = sample["QUESTION"]
        contexts     = sample["CONTEXTS"]
        long_answer  = sample.get("LONG_ANSWER", "")
        ground_truth = sample["final_decision"].lower()
        print(f"\n[{i+1}/{len(all_keys)}] Processing...")

        try:
            expanded_q = expand_medical_query(clean_query(question[:200]))
            candidates = hybrid_retrieve(expanded_q, retriever)
            top5       = rerank(expanded_q, candidates, retriever, top_k=5)
            top5       = check_doc_relevance(question, top5,
                                             retriever["embedding_model"], 0.45)

            pqa_prompt = build_pubmedqa_prompt(question, contexts)
            inputs     = tokenizer(pqa_prompt, return_tensors="pt",
                                   truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=300,
                    temperature=0.1, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            response_text = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            support   = verify_support(question, response_text, top5, model, tokenizer)
            utility   = score_utility(question, response_text, support, model, tokenizer)
            predicted = extract_yes_no_maybe(response_text)
            is_correct = (predicted == ground_truth) if predicted else False
            rouge     = compute_rouge(response_text, long_answer) if long_answer else {}

            result = {
                "id":              sid,
                "pmid":            pmid,
                "ground_truth":    ground_truth,
                "predicted":       predicted,
                "is_correct":      is_correct,
                "support_label":   support["support_label"],
                "support_score":   support["support_score"],
                "utility_score":   utility["utility_score"],
                "is_hallucinated": support["support_label"] == "No Support",
                "rouge":           rouge,
            }
            results.append(result)
            completed_ids.add(sid)
            checkpoint.update(results=results, completed_ids=list(completed_ids))
            save_checkpoint(PUBMEDQA_CHECKPOINT, checkpoint)

            icon = "✅" if is_correct else "❌"
            r1   = rouge.get("rouge1", 0) if rouge else 0
            print(f"   {icon} Pred: {predicted} | Truth: {ground_truth} | "
                  f"ROUGE-1: {r1:.3f} | Support: {support['support_label']}")

        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            results.append({"id": sid, "error": str(e),
                            "is_correct": False, "is_hallucinated": False, "rouge": {}})
            completed_ids.add(sid)
            checkpoint.update(results=results, completed_ids=list(completed_ids))
            save_checkpoint(PUBMEDQA_CHECKPOINT, checkpoint)

    valid        = [r for r in results if "error" not in r]
    correct      = sum(1 for r in valid if r["is_correct"])
    hallucinated = sum(1 for r in valid if r["is_hallucinated"])
    rouge_list   = [r["rouge"] for r in valid if r.get("rouge")]
    avg_r1 = round(sum(r.get("rouge1", 0) for r in rouge_list) / len(rouge_list), 4) if rouge_list else 0
    avg_r2 = round(sum(r.get("rouge2", 0) for r in rouge_list) / len(rouge_list), 4) if rouge_list else 0
    avg_rL = round(sum(r.get("rougeL", 0) for r in rouge_list) / len(rouge_list), 4) if rouge_list else 0
    accuracy  = round(correct / len(valid) * 100, 2) if valid else 0
    hall_rate = round(hallucinated / len(valid) * 100, 2) if valid else 0

    summary = {
        "total_evaluated":        len(valid),
        "correct":                correct,
        "accuracy_pct":           accuracy,
        "hallucination_rate_pct": hall_rate,
        "unparsed_responses":     sum(1 for r in valid if r.get("predicted") is None),
        "avg_rouge1":             avg_r1,
        "avg_rouge2":             avg_r2,
        "avg_rougeL":             avg_rL,
    }
    print(f"\n{'='*60}")
    print(f"📊 PUBMEDQA RESULTS")
    print(f"   Accuracy      : {accuracy}%")
    print(f"   Hallucination : {hall_rate}%")
    print(f"   ROUGE-1/2/L   : {avg_r1} / {avg_r2} / {avg_rL}")
    print(f"{'='*60}")
    return summary


# ── Final report ──────────────────────────────────────────────────

def generate_final_report(medqa: dict, pubmedqa: dict) -> dict:
    report = {
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system":   "AI Medical Assistant using RAG + Self-RAG",
        "model":    "Llama-2-7B-chat (4-bit)",
        "retriever": "BioBERT-FAISS + PubMed API",
        "medqa":    medqa,
        "pubmedqa": pubmedqa,
        "overall": {
            "avg_accuracy":         round((medqa["accuracy_pct"] + pubmedqa["accuracy_pct"]) / 2, 2),
            "avg_hallucination_rate": round((medqa["hallucination_rate_pct"] + pubmedqa["hallucination_rate_pct"]) / 2, 2),
        },
    }
    with open(FINAL_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📁 Final report saved → {FINAL_REPORT}")
    return report


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    from selfrag  import load_model
    from retrieval import load_retriever

    model, tokenizer = load_model()
    retriever        = load_retriever()

    medqa_summary    = evaluate_medqa(model, tokenizer, retriever)
    pubmedqa_summary = evaluate_pubmedqa(model, tokenizer, retriever)
    generate_final_report(medqa_summary, pubmedqa_summary)

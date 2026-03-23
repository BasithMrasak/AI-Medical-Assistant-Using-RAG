"""
preprocess.py
=============
Normalises and chunks the raw medical corpora (MedQA textbooks +
PubMedQA artificial) into a single chunked_corpus.json file ready
for FAISS embedding.

Outputs
-------
  data/normalized_medqa_textbooks.json
  data/normalized_pubmed_artificial.json
  data/chunked_corpus.json
"""

import os
import json
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────
BASE_PATH   = "/content/drive/MyDrive/main project"
MEDQA_DIR   = f"{BASE_PATH}/data/textbooks/en"
PUBMED_FILE = f"{BASE_PATH}/data/ori_pqaa.json"

OUT_MEDQA   = f"{BASE_PATH}/data/normalized_medqa_textbooks.json"
OUT_PUBMED  = f"{BASE_PATH}/data/normalized_pubmed_artificial.json"
OUT_CHUNKS  = f"{BASE_PATH}/data/chunked_corpus.json"

CHUNK_SIZE  = 400   # words
CHUNK_OVERLAP = 50  # words


# ── Helpers ───────────────────────────────────────────────────────

def read_text_file(filepath: str) -> str | None:
    """Read a UTF-8 text file; return None if too short or unreadable."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text if len(text) > 50 else None
    except Exception:
        return None


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk_words = words[start: start + chunk_size]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap
    return chunks


# ── Step 1: Normalise MedQA textbooks ────────────────────────────

def normalise_medqa(input_dir: str, output_file: str) -> list[dict]:
    print("\n[1/3] Normalising MedQA textbooks...")
    docs = []
    counter = 1
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith((".txt", ".md")):
                continue
            text = read_text_file(os.path.join(root, file))
            if text is None:
                continue
            docs.append({
                "doc_id":   f"medqa_{counter:06d}",
                "text":     text,
                "source":   "MedQA_Textbook",
                "book":     os.path.splitext(file)[0],
                "language": "en"
            })
            counter += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(docs)} MedQA documents → {output_file}")
    return docs


# ── Step 2: Normalise PubMedQA artificial ────────────────────────

def normalise_pubmed(input_file: str, output_file: str) -> list[dict]:
    print("\n[2/3] Normalising PubMedQA artificial...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for pmid, record in tqdm(data.items()):
        contexts = record.get("CONTEXTS", [])
        combined = " ".join(c.strip() for c in contexts if len(c.strip()) > 20)
        if len(combined) < 100:
            continue
        docs.append({
            "doc_id":   f"pmid_{pmid}",
            "text":     combined,
            "source":   "PubMed_Artificial",
            "pmid":     pmid,
            "language": "en"
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(docs)} PubMed documents → {output_file}")
    return docs


# ── Step 3: Chunk all documents ───────────────────────────────────

def chunk_all(medqa_docs: list[dict], pubmed_docs: list[dict],
              output_file: str) -> list[dict]:
    print("\n[3/3] Chunking combined corpus...")
    all_docs = medqa_docs + pubmed_docs
    chunked  = []

    for doc in tqdm(all_docs):
        for idx, chunk in enumerate(chunk_text(doc["text"])):
            entry = {
                "chunk_id":      f"{doc['doc_id']}_chunk_{idx+1:03d}",
                "parent_doc_id": doc["doc_id"],
                "text":          chunk,
                "source":        doc["source"],
                "language":      doc["language"],
            }
            if doc["source"] == "PubMed_Artificial":
                entry["pmid"] = doc["pmid"]
            if doc["source"] == "MedQA_Textbook":
                entry["book"] = doc["book"]
            chunked.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunked, f, ensure_ascii=False, indent=2)

    pubmed_count = sum(1 for c in chunked if c["source"] == "PubMed_Artificial")
    medqa_count  = sum(1 for c in chunked if c["source"] == "MedQA_Textbook")
    print(f"  Total chunks : {len(chunked)}")
    print(f"    MedQA      : {medqa_count}")
    print(f"    PubMed     : {pubmed_count}")
    print(f"  Saved → {output_file}")
    return chunked


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    medqa_docs  = normalise_medqa(MEDQA_DIR, OUT_MEDQA)
    pubmed_docs = normalise_pubmed(PUBMED_FILE, OUT_PUBMED)
    chunk_all(medqa_docs, pubmed_docs, OUT_CHUNKS)
    print("\n✅ Preprocessing complete.")

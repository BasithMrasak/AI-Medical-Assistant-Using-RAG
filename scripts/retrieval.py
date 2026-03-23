"""
retrieval.py
============
Hybrid retrieval pipeline:
  - Two-stage FAISS (coarse → BioBERT rerank)
  - Real-time PubMed retrieval for time-sensitive queries
  - CrossEncoder reranker to select top-k candidates

Prerequisites
-------------
  pip install faiss-cpu sentence-transformers biopython

Usage
-----
  from retrieval import load_retriever, hybrid_retrieve, rerank
  retriever = load_retriever()
  candidates = hybrid_retrieve("treatment options for dengue fever", retriever)
  top5 = rerank("treatment options for dengue fever", candidates, top_k=5)
"""

import re
import json
import numpy as np
import faiss
from Bio import Entrez
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── PubMed credentials (replace with your own) ────────────────────
ENTREZ_EMAIL   = "your_email@example.com"
ENTREZ_API_KEY = "your_ncbi_api_key"       # optional but recommended

# ── Paths ─────────────────────────────────────────────────────────
BASE_PATH   = "/content/drive/MyDrive/main project"
INDEX_PATH  = f"{BASE_PATH}/vector_store/medical_faiss.index"
META_PATH   = f"{BASE_PATH}/vector_store/metadata.json"
CORPUS_PATH = f"{BASE_PATH}/data/chunked_corpus.json"

EMBED_MODEL  = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TIME_KEYWORDS = [
    "latest", "recent", "current", "new", "updated",
    "guideline", "recommendation", "approval"
]


# ── Retriever initialisation ──────────────────────────────────────

def load_retriever() -> dict:
    """Load FAISS index, metadata, corpus lookup, and models."""
    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    with open(CORPUS_PATH, "r") as f:
        corpus_chunks = json.load(f)

    chunk_lookup = {c["chunk_id"]: c["text"] for c in corpus_chunks}

    print("Loading embedding + reranker models...")
    embedding_model = SentenceTransformer(EMBED_MODEL)
    reranker        = CrossEncoder(RERANK_MODEL)

    Entrez.email   = ENTREZ_EMAIL
    Entrez.api_key = ENTREZ_API_KEY

    print(f"✅ Retriever ready — {index.ntotal} vectors in index")
    return {
        "index":           index,
        "metadata":        metadata,
        "chunk_lookup":    chunk_lookup,
        "embedding_model": embedding_model,
        "reranker":        reranker,
    }


# ── Time-sensitivity detection ────────────────────────────────────

def is_time_sensitive(query: str) -> bool:
    q = query.lower()
    if any(k in q for k in TIME_KEYWORDS):
        return True
    if re.search(r"\b(19|20)\d{2}\b", q):
        return True
    if re.search(r"last\s+\d+\s+(months|years)", q):
        return True
    if "this year" in q or "past few years" in q:
        return True
    return False


# ── FAISS retrieval ───────────────────────────────────────────────

def retrieve_faiss(query: str, retriever: dict, k: int = 32) -> list[dict]:
    """Retrieve top-k candidates from the local FAISS index."""
    emb = retriever["embedding_model"].encode(
        query, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    distances, indices = retriever["index"].search(emb.reshape(1, -1), k)
    results = []
    for idx, score in zip(indices[0], distances[0]):
        meta = retriever["metadata"][idx]
        cid  = meta["chunk_id"]
        results.append({
            "chunk_id":   cid,
            "text":       retriever["chunk_lookup"].get(cid, ""),
            "source":     meta["source"],
            "pmid":       meta.get("pmid"),
            "book":       meta.get("book"),
            "faiss_score": float(score),
        })
    return results


# ── PubMed live retrieval ─────────────────────────────────────────

def pubmed_search(query: str, retmax: int = 20, year_from: int = 2020) -> list[str]:
    """Search PubMed and return a list of PMIDs."""
    q = f'{query} AND ("{year_from}"[Date - Publication] : "3000"[DP])'
    handle = Entrez.esearch(db="pubmed", term=q, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def pubmed_fetch(pmids: list[str]) -> list[dict]:
    """Fetch and parse abstracts for a list of PMIDs."""
    if not pmids:
        return []
    handle  = Entrez.efetch(db="pubmed", id=",".join(pmids),
                             rettype="abstract", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    docs = []
    for art in records["PubmedArticle"]:
        med      = art["MedlineCitation"]
        art_data = med["Article"]
        title    = art_data.get("ArticleTitle", "")
        abstract = ""
        if "Abstract" in art_data:
            abstract = " ".join(art_data["Abstract"]["AbstractText"])
        docs.append({
            "doc_id":   f"pubmed_live_{med['PMID']}",
            "text":     f"{title}. {abstract}",
            "source":   "PubMed_Live",
            "pmid":     str(med["PMID"]),
            "language": "en",
        })
    return docs


def _simple_splitter(text: str, chunk_size: int = 512,
                     overlap: int = 100) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - overlap
    return chunks


def process_live_docs(docs: list[dict], query: str,
                      embedding_model) -> list[dict]:
    """Chunk and score live PubMed documents against the query."""
    chunks = []
    for doc in docs:
        for i, text in enumerate(_simple_splitter(doc["text"])):
            chunks.append({
                "chunk_id":      f"{doc['doc_id']}_chunk_{i+1:03d}",
                "parent_doc_id": doc["doc_id"],
                "text":          text,
                "source":        doc["source"],
                "pmid":          doc["pmid"],
            })
    if not chunks:
        return []

    texts  = [c["text"] for c in chunks]
    embs   = embedding_model.encode(texts, convert_to_numpy=True,
                                     normalize_embeddings=True)
    q_emb  = embedding_model.encode(query, convert_to_numpy=True,
                                     normalize_embeddings=True)
    for c, e in zip(chunks, embs):
        c["faiss_score"] = float(np.dot(q_emb, e))
    return chunks


# ── Hybrid retrieval ──────────────────────────────────────────────

def hybrid_retrieve(query: str, retriever: dict) -> list[dict]:
    """
    Retrieve from FAISS; append live PubMed results for
    time-sensitive queries.
    """
    faiss_results = retrieve_faiss(query, retriever)
    if not is_time_sensitive(query):
        return faiss_results

    pmids      = pubmed_search(query)
    live_docs  = pubmed_fetch(pmids)
    live_chunks = process_live_docs(live_docs, query,
                                    retriever["embedding_model"])
    return faiss_results + live_chunks


# ── CrossEncoder reranking ────────────────────────────────────────

def rerank(query: str, candidates: list[dict],
           retriever: dict, top_k: int = 5) -> list[dict]:
    """Rerank candidates with a CrossEncoder and return top-k."""
    pairs  = [(query, c["text"]) for c in candidates]
    scores = retriever["reranker"].predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: x["rerank_score"],
                  reverse=True)[:top_k]


# ── Topic relevance filter ────────────────────────────────────────

def check_doc_relevance(query: str, docs: list[dict],
                        embedding_model,
                        threshold: float = 0.3) -> list[dict]:
    """Filter documents below a cosine similarity threshold."""
    q_emb    = embedding_model.encode(query, convert_to_numpy=True,
                                       normalize_embeddings=True)
    relevant = []
    for doc in docs:
        d_emb = embedding_model.encode(doc["text"][:300],
                                        convert_to_numpy=True,
                                        normalize_embeddings=True)
        sim = float(np.dot(q_emb, d_emb))
        doc["topic_similarity"] = sim
        if sim >= threshold:
            relevant.append(doc)

    if not relevant:
        docs.sort(key=lambda x: x.get("topic_similarity", 0), reverse=True)
        return docs[:3]
    return relevant

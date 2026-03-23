"""
build_index.py
==============
Encodes chunked_corpus.json with BioBERT sentence embeddings and
builds a FAISS IndexFlatL2 vector store saved to disk.

Prerequisites
-------------
  pip install faiss-cpu sentence-transformers tqdm

Outputs
-------
  vector_store/medical_faiss.index
  vector_store/metadata.json
"""

import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────
BASE_PATH    = "/content/drive/MyDrive/main project"
CORPUS_PATH  = f"{BASE_PATH}/data/chunked_corpus.json"
SAVE_DIR     = f"{BASE_PATH}/vector_store"
INDEX_PATH   = f"{SAVE_DIR}/medical_faiss.index"
META_PATH    = f"{SAVE_DIR}/metadata.json"

EMBED_MODEL  = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
BATCH_SIZE   = 64


def build_faiss_index(corpus_path: str, index_path: str, meta_path: str) -> None:
    print("\n[1/3] Loading corpus...")
    with open(corpus_path, "r") as f:
        all_chunks = json.load(f)
    print(f"  Total chunks: {len(all_chunks)}")

    print("\n[2/3] Loading BioBERT embedding model...")
    model     = SentenceTransformer(EMBED_MODEL, device="cuda")
    embed_dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {embed_dim}")

    index    = faiss.IndexFlatL2(embed_dim)
    metadata = []
    vector_id = 0

    print("\n[3/3] Encoding and indexing...")
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE)):
        batch  = all_chunks[i: i + BATCH_SIZE]
        texts  = [chunk["text"] for chunk in batch]

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        index.add(embeddings)

        for chunk in batch:
            metadata.append({
                "vector_id":    vector_id,
                "chunk_id":     chunk["chunk_id"],
                "parent_doc_id": chunk["parent_doc_id"],
                "source":       chunk["source"],
                "language":     chunk["language"],
                "pmid":         chunk.get("pmid"),
                "book":         chunk.get("book"),
            })
            vector_id += 1

    os.makedirs(SAVE_DIR, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ FAISS index built")
    print(f"  Vectors   : {index.ntotal}")
    print(f"  Metadata  : {len(metadata)}")
    print(f"  Index     → {index_path}")
    print(f"  Metadata  → {meta_path}")


if __name__ == "__main__":
    build_faiss_index(CORPUS_PATH, INDEX_PATH, META_PATH)

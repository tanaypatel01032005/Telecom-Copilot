"""
src/retrieval/faiss_indexer.py

Week 2 — Build a FAISS index over all KB passages using the fine-tuned retriever.

What this does:
  1. Loads all passages from kb_passages.jsonl
  2. Encodes them with the fine-tuned sentence-transformers model
  3. Builds a FAISS IndexFlatIP (inner product = cosine on unit vectors)
  4. Saves the index + passage metadata for fast retrieval at inference

Why IndexFlatIP?
  - Exact search — no approximation error
  - Suitable for our corpus size (~3000-15000 passages after MD2D)
  - If corpus grows >100K, switch to IndexIVFFlat (approximate but faster)

Run:
    python -m src.retrieval.faiss_indexer
    python -m src.retrieval.faiss_indexer --model checkpoints/retriever   (fine-tuned)
    python -m src.retrieval.faiss_indexer --model sentence-transformers/all-MiniLM-L6-v2 (base)
"""
import os

os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import faiss
import numpy as np

import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


# ─── Embed passages ───────────────────────────────────────────────────────────

def embed_passages(
    passages:    List[Dict],
    model_path:  str,
    batch_size:  int = 128,
    field:       str = "full_text",
) -> np.ndarray:
    """
    Encodes all passages using the sentence-transformers model.
    Uses 'full_text' (heading + text) for richer context.
    Returns a float32 numpy array of shape (n_passages, embedding_dim).
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading encoder: {model_path}")
    model = SentenceTransformer(model_path)

    texts = [p.get(field, p.get("text", "")) for p in passages]
    print(f"  Encoding {len(texts):,} passages (batch_size={batch_size})...")

    t0         = time.time()
    embeddings = model.encode(
        texts,
        batch_size        = batch_size,
        show_progress_bar = True,
        convert_to_numpy  = True,
        normalize_embeddings = True,   # normalise → inner product = cosine similarity
    )
    elapsed = time.time() - t0
    print(f"  Encoded {len(texts):,} passages in {elapsed:.1f}s "
          f"({len(texts)/elapsed:.0f} passages/sec)")
    print(f"  Embedding shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


# ─── Build FAISS index ────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    """
    Builds a FAISS IndexFlatIP (exact inner product search).
    Since embeddings are L2-normalised, inner product = cosine similarity.
    """
    import faiss

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Wrap with IDMap so we can store integer IDs for each vector
    # (IDs = passage row index → passage metadata lookup)
    id_index = faiss.IndexIDMap(index)
    ids      = np.arange(len(embeddings), dtype=np.int64)

    print(f"  Building FAISS IndexFlatIP (dim={dim})...")
    id_index.add_with_ids(embeddings, ids)
    print(f"  Index contains {id_index.ntotal:,} vectors")
    return id_index


# ─── Save / Load index ────────────────────────────────────────────────────────

def save_index(
    index,
    passages:     List[Dict],
    output_dir:   str = "data/index",
    label:        str = "finetuned",
):
    """
    Saves:
      {label}_faiss.index        ← FAISS binary index
      {label}_passage_store.json ← passage metadata (id → passage)
      {label}_index_meta.json    ← build config & stats
    """
    import faiss

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # FAISS index
    idx_path = out / f"{label}_faiss.index"
    faiss.write_index(index, str(idx_path))
    print(f"  Saved FAISS index → {idx_path}")

    # Passage store: row_id → passage record
    # Store only the fields needed at retrieval time to keep file small
    store = {}
    for i, p in enumerate(passages):
        store[str(i)] = {
            "passage_id": p["passage_id"],
            "doc_id":     p["doc_id"],
            "section_id": p["section_id"],
            "title":      p["title"],
            "heading":    p.get("heading", ""),
            "text":       p["text"],
            "category":   p.get("category", ""),
            "domain":     p.get("domain", ""),
            "source":     p.get("source", ""),
        }
    store_path = out / f"{label}_passage_store.json"
    with open(store_path, "w") as f:
        json.dump(store, f, ensure_ascii=False)
    print(f"  Saved passage store ({len(store):,} entries) → {store_path}")

    # Meta
    meta = {
        "label":           label,
        "n_passages":      len(passages),
        "embedding_dim":   index.d,
        "index_type":      "IndexIDMap(IndexFlatIP)",
        "sources": {
            s: sum(1 for p in passages if p.get("source") == s)
            for s in set(p.get("source", "unknown") for p in passages)
        }
    }
    meta_path = out / f"{label}_index_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved index meta → {meta_path}")


def load_index(index_dir: str = "data/index", label: str = "finetuned"):
    """Loads a previously built FAISS index + passage store."""
    import faiss

    idx_path   = Path(index_dir) / f"{label}_faiss.index"
    store_path = Path(index_dir) / f"{label}_passage_store.json"

    if not idx_path.exists():
        raise FileNotFoundError(f"Index not found: {idx_path}")

    index = faiss.read_index(str(idx_path))
    with open(store_path) as f:
        store = json.load(f)

    print(f"  Loaded FAISS index ({index.ntotal:,} vectors) from {idx_path}")
    return index, store


# ─── Dense Retriever class (used by full system in Week 3+) ───────────────────

class DenseRetriever:
    """
    Wraps a FAISS index + passage store for easy retrieval.
    Used by the full system pipeline from Week 3 onward.

    Replaces BM25 from the baseline.
    """

    def __init__(
        self,
        model_path:  str = "checkpoints/retriever",
        index_dir:   str = "data/index",
        label:       str = "finetuned",
    ):
        from sentence_transformers import SentenceTransformer
        import faiss

        print(f"  [DenseRetriever] Loading encoder: {model_path}")
        self.model = SentenceTransformer(model_path)

        print(f"  [DenseRetriever] Loading FAISS index (label={label})...")
        self.index, self.store = load_index(index_dir, label)
        self.label             = label

    def search(
        self,
        query:           str,
        top_k:           int = 5,
        category_filter: str = "any",
    ) -> List[Dict]:
        """
        Encodes the query and retrieves top-k passages from FAISS.

        Args:
            query:           User query string
            top_k:           Number of passages to return
            category_filter: Restrict to a specific category ("any" = no filter)

        Returns:
            List of passage dicts with an added "dense_score" field
        """
        import numpy as np

        # Encode query (single, no batching needed)
        q_emb = self.model.encode(
            [query],
            normalize_embeddings = True,
            convert_to_numpy     = True,
        ).astype(np.float32)

        # Over-retrieve if filtering, to still get top_k after filter
        fetch_k = top_k * 5 if category_filter != "any" else top_k

        scores, ids = self.index.search(q_emb, fetch_k)
        scores = scores[0]
        ids    = ids[0]

        results = []
        for score, idx in zip(scores, ids):
            if idx < 0:   # FAISS returns -1 for empty slots
                continue
            passage = dict(self.store[str(idx)])
            passage["dense_score"] = round(float(score), 4)

            if category_filter != "any":
                if passage.get("category", "") != category_filter:
                    continue

            results.append(passage)
            if len(results) >= top_k:
                break

        return results

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """Encode multiple queries in one batch for efficiency."""
        import numpy as np

        q_embs = self.model.encode(
            queries,
            normalize_embeddings = True,
            convert_to_numpy     = True,
            batch_size           = 64,
        ).astype(np.float32)

        all_scores, all_ids = self.index.search(q_embs, top_k)

        results = []
        for scores, ids in zip(all_scores, all_ids):
            passages = []
            for score, idx in zip(scores, ids):
                if idx < 0:
                    continue
                p = dict(self.store[str(idx)])
                p["dense_score"] = round(float(score), 4)
                passages.append(p)
            results.append(passages)
        return results


# ─── Full pipeline: embed + index ─────────────────────────────────────────────

def build_index_pipeline(
    kb_path:     str = "data/processed/kb_passages.jsonl",
    model_path:  str = "checkpoints/retriever",
    output_dir:  str = "data/index",
    label:       str = "finetuned",
    batch_size:  int = 128,
):
    """
    Full pipeline:
      1. Load KB passages
      2. Embed with fine-tuned model
      3. Build FAISS index
      4. Save index + store

    Also builds a "base" index (un-finetuned model) for comparison.
    """
    print(f"\n{'='*60}")
    print(f"  FAISS INDEX BUILDER")
    print(f"  KB     : {kb_path}")
    print(f"  Model  : {model_path}")
    print(f"  Label  : {label}")
    print(f"{'='*60}\n")

    # Load passages
    if not Path(kb_path).exists():
        raise FileNotFoundError(f"KB not found: {kb_path}")
    with open(kb_path) as f:
        passages = [json.loads(line) for line in f]
    print(f"  Loaded {len(passages):,} passages from {kb_path}")

    # Embed + index
    embeddings = embed_passages(passages, model_path, batch_size=batch_size)
    index      = build_faiss_index(embeddings)
    save_index(index, passages, output_dir, label)

    return index, passages


# ─── Retrieval benchmark (BM25 vs Dense) ──────────────────────────────────────

def benchmark_retrieval(
    test_cases_path: str = "data/processed/test_cases.jsonl",
    index_dir:       str = "data/index",
    kb_path:         str = "data/processed/kb_passages.jsonl",
    retriever_path:  str = "checkpoints/retriever",
    top_k:           int = 5,
) -> Dict:
    """
    Compares BM25 (baseline) vs Dense Retriever (fine-tuned)
    on the test set. Reports Recall@1, Recall@5, MRR@10.

    This is the KEY Week 2 result table for your report.
    """
    import sys
    sys.path.insert(0, ".")
    from src.baseline.baseline_system import BM25

    # Load test cases
    if not Path(test_cases_path).exists():
        print(f"  Test cases not found: {test_cases_path}")
        return {}
    with open(test_cases_path) as f:
        cases = [json.loads(line) for line in f]

    # Filter cases that have a gold_section_id (needed for retrieval eval)
    cases = [c for c in cases if c.get("gold_section_id")]
    print(f"\n  Benchmarking retrieval on {len(cases)} test cases...")

    # Load KB for BM25
    with open(kb_path) as f:
        passages = [json.loads(line) for line in f]

    bm25 = BM25(passages)

    # Load dense retriever
    try:
        dense = DenseRetriever(retriever_path, index_dir, label="finetuned")
        has_dense = True
    except Exception as e:
        print(f"  Warning: Dense retriever not available ({e}). BM25 only.")
        has_dense = False

    def recall_at_k(ranked_ids: List[str], gold_id: str, k: int) -> float:
        return 1.0 if gold_id in ranked_ids[:k] else 0.0

    bm25_r1, bm25_r5, bm25_mrr  = [], [], []
    dense_r1, dense_r5, dense_mrr = [], [], []

    for case in cases:
        query    = case["query"]
        gold_id  = case.get("gold_section_id", "")

        # BM25
        bm25_results = bm25.search(query, top_k=top_k)
        bm25_ids     = [r["section_id"] for r in bm25_results]
        bm25_r1.append(recall_at_k(bm25_ids, gold_id, 1))
        bm25_r5.append(recall_at_k(bm25_ids, gold_id, 5))
        rank_b = next((i+1 for i, sid in enumerate(bm25_ids) if sid == gold_id), top_k+1)
        bm25_mrr.append(1.0 / rank_b)

        # Dense
        if has_dense:
            dense_results = dense.search(query, top_k=top_k)
            dense_ids     = [r["section_id"] for r in dense_results]
            dense_r1.append(recall_at_k(dense_ids, gold_id, 1))
            dense_r5.append(recall_at_k(dense_ids, gold_id, 5))
            rank_d = next((i+1 for i, sid in enumerate(dense_ids) if sid == gold_id), top_k+1)
            dense_mrr.append(1.0 / rank_d)

    def avg(lst): return round(sum(lst)/len(lst), 4) if lst else 0.0

    print(f"\n{'='*60}")
    print(f"  RETRIEVAL BENCHMARK (top_k={top_k})")
    print(f"  {'Metric':<22} {'BM25':>10} {'Dense (FT)':>12} {'Delta':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*8}")
    metrics = [
        ("Recall@1",  avg(bm25_r1),  avg(dense_r1) if has_dense else None),
        ("Recall@5",  avg(bm25_r5),  avg(dense_r5) if has_dense else None),
        ("MRR@10",    avg(bm25_mrr), avg(dense_mrr) if has_dense else None),
    ]
    for name, b, d in metrics:
        if d is not None:
            delta  = d - b
            arrow  = "↑" if delta > 0 else "↓"
            print(f"  {name:<22} {b:>10.4f} {d:>12.4f} {arrow}{abs(delta):>6.4f}")
        else:
            print(f"  {name:<22} {b:>10.4f} {'N/A':>12}")
    print(f"{'='*60}")

    report = {
        "bm25":  {"recall_at_1": avg(bm25_r1), "recall_at_5": avg(bm25_r5), "mrr_at_10": avg(bm25_mrr)},
        "dense": {"recall_at_1": avg(dense_r1), "recall_at_5": avg(dense_r5), "mrr_at_10": avg(dense_mrr)} if has_dense else {},
    }
    out_path = Path("data/processed/retrieval_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Benchmark saved → {out_path}")
    return report


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="checkpoints/retriever",
                        help="Path to fine-tuned model (or HuggingFace model name)")
    parser.add_argument("--label",      default="finetuned",
                        help="Label for index files (e.g. 'finetuned', 'base')")
    parser.add_argument("--kb",         default="data/processed/kb_passages.jsonl")
    parser.add_argument("--output-dir", default="data/index")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--benchmark",  action="store_true",
                        help="Run retrieval benchmark after indexing")
    parser.add_argument("--base-only",  action="store_true",
                        help="Build index with BASE (un-finetuned) model only")
    args = parser.parse_args()

    if args.base_only:
        build_index_pipeline(
            kb_path    = args.kb,
            model_path = "sentence-transformers/all-MiniLM-L6-v2",
            output_dir = args.output_dir,
            label      = "base",
            batch_size = args.batch_size,
        )
    else:
        build_index_pipeline(
            kb_path    = args.kb,
            model_path = args.model,
            output_dir = args.output_dir,
            label      = args.label,
            batch_size = args.batch_size,
        )

    if args.benchmark:
        benchmark_retrieval(
            index_dir      = args.output_dir,
            retriever_path = args.model,
        )

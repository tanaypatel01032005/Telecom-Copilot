"""
src/retrieval/reranker.py

Week 2 — Cross-Encoder Reranker.

Architecture:
  Base model : cross-encoder/ms-marco-MiniLM-L-6-v2
               (pre-trained on MS-MARCO passage retrieval — good zero-shot)
  Fine-tuning: Binary classification (query, passage) → relevant (1) / not (0)
  Data       : MD2D retriever triples
               positive = (query, gold_span) → label 1
               negative = (query, hard_neg_span) → label 0

Why rerank?
  - Dense retriever returns top-20 candidates efficiently
  - Reranker re-scores top-20 with a full cross-attention model
  - Cross-encoder sees both query AND passage together → much better precision
  - Typical pipeline: BM25/Dense retrieves top-20 → reranker selects top-3 → generator

Run:
    python -m src.retrieval.reranker --train
    python -m src.retrieval.reranker --eval
    python -m src.retrieval.reranker --demo "How do I dispute a billing error?"
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple

random.seed(42)


# ─── Training data builder for reranker ───────────────────────────────────────

def build_reranker_dataset(
    triples_path:    str = "data/processed/retriever_train.jsonl",
    span_index_path: str = "data/processed/span_index.json",
    max_samples:     int = 8000,
    neg_per_pos:     int = 3,
) -> Tuple[List, List]:
    """
    Converts retriever triples into (query, passage, label) pairs for reranker.

    For each triple:
      + 1 positive example: (query, gold_passage, label=1)
      + neg_per_pos negative examples: (query, hard_neg_passage, label=0)

    Returns: train_data, val_data
    Each sample = {"query": str, "passage": str, "label": int}
    """
    with open(span_index_path) as f:
        span_index = json.load(f)

    triples = []
    with open(triples_path) as f:
        for line in f:
            triples.append(json.loads(line))
            if len(triples) >= max_samples:
                break

    random.shuffle(triples)
    data = []

    for t in triples:
        query         = t.get("query", "").strip()
        positive_text = t.get("positive_text", "")
        if not positive_text:
            positive_text = span_index.get(t.get("positive_id", ""), {}).get("text", "")

        if len(query) < 5 or len(positive_text) < 20:
            continue

        # Positive
        data.append({"query": query, "passage": positive_text, "label": 1})

        # Negatives
        for neg_id in t.get("hard_negatives", [])[:neg_per_pos]:
            neg_text = span_index.get(neg_id, {}).get("text", "")
            if neg_text and len(neg_text) > 20:
                data.append({"query": query, "passage": neg_text, "label": 0})

    random.shuffle(data)
    split     = int(len(data) * 0.9)
    train_data = data[:split]
    val_data   = data[split:]

    print(f"  Reranker dataset: {len(train_data)} train / {len(val_data)} val")
    pos = sum(1 for d in train_data if d["label"] == 1)
    neg = sum(1 for d in train_data if d["label"] == 0)
    print(f"  Train balance: {pos} positive / {neg} negative")
    return train_data, val_data


# ─── Train reranker ───────────────────────────────────────────────────────────

def train_reranker(
    base_model:  str = "BAAI/bge-reranker-base",
    output_dir:  str = "checkpoints/reranker",
    max_samples: int = 8000,
    num_epochs:  int = 2,
    batch_size:  int = 16,
    lr:          float = 2e-5,
    max_length:  int = 512,
):
    """
    Fine-tunes a cross-encoder reranker.

    The cross-encoder takes [CLS] query [SEP] passage [SEP] as input
    and outputs a relevance score. This is more powerful than bi-encoders
    because query and passage attend to each other in every layer.

    We use sentence-transformers' CrossEncoder class which handles:
      - Tokenization of (query, passage) pairs
      - Binary cross-entropy loss
      - Evaluation with accuracy / AP score
    """
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

    print(f"\n{'='*60}")
    print(f"  RERANKER FINE-TUNING")
    print(f"  Base model : {base_model}")
    print(f"  Epochs     : {num_epochs}")
    print(f"  Batch size : {batch_size}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────
    train_data, val_data = build_reranker_dataset(max_samples=max_samples)

    # Convert to CrossEncoder format: list of (texts, label) tuples
    train_samples = [([d["query"], d["passage"]], d["label"]) for d in train_data]
    val_samples   = [([d["query"], d["passage"]], d["label"]) for d in val_data]

    # ── Load model ────────────────────────────────────────────────
    print(f"  Loading base model: {base_model}")
    model = CrossEncoder(
        base_model,
        num_labels  = 1,
        max_length  = max_length,
    )

    # ── Evaluate before training ───────────────────────────────────
    print("\n  Base model evaluation (before fine-tuning)...")
    base_scores = _evaluate_cross_encoder(model, val_data[:200])
    print(f"    Accuracy  : {base_scores['accuracy']:.4f}")
    print(f"    Avg score (pos): {base_scores['avg_pos_score']:.4f}")
    print(f"    Avg score (neg): {base_scores['avg_neg_score']:.4f}")

    # ── Train ──────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader = _make_dataloader(train_samples, batch_size),
        epochs           = num_epochs,
        warmup_steps     = int(len(train_samples) / batch_size * 0.1),
        output_path      = output_dir,
        show_progress_bar= True,
        optimizer_params = {"lr": lr},
    )
    print(f"\n  Reranker saved -> {output_dir}")

    # ── Evaluate after training ────────────────────────────────────
    print("\n  Fine-tuned model evaluation (after training)...")
    ft_model = CrossEncoder(output_dir, max_length=max_length)
    ft_scores = _evaluate_cross_encoder(ft_model, val_data[:200])
    print(f"    Accuracy  : {ft_scores['accuracy']:.4f}")
    print(f"    Avg score (pos): {ft_scores['avg_pos_score']:.4f}")
    print(f"    Avg score (neg): {ft_scores['avg_neg_score']:.4f}")

    # ── Save report ───────────────────────────────────────────────
    report = {
        "base_model":        base_model,
        "before_finetuning": base_scores,
        "after_finetuning":  ft_scores,
        "delta": {
            k: round(ft_scores[k] - base_scores[k], 4)
            for k in base_scores
        }
    }
    with open(Path(output_dir) / "reranker_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    return ft_model, report


def _make_dataloader(samples, batch_size):
    """Wraps samples in a simple DataLoader for CrossEncoder.fit()."""
    from torch.utils.data import DataLoader, Dataset

    class PairDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    return DataLoader(PairDataset(samples), batch_size=batch_size, shuffle=True)


def _evaluate_cross_encoder(model, val_data: List[Dict]) -> Dict:
    """Simple accuracy + avg score evaluation."""
    import numpy as np

    pairs  = [[d["query"], d["passage"]] for d in val_data]
    labels = [d["label"] for d in val_data]
    scores = model.predict(pairs, show_progress_bar=False)

    # Threshold at 0.5 for accuracy
    preds    = (scores > 0.5).astype(int)
    accuracy = float(np.mean(preds == np.array(labels)))

    pos_scores = [scores[i] for i, l in enumerate(labels) if l == 1]
    neg_scores = [scores[i] for i, l in enumerate(labels) if l == 0]

    return {
        "accuracy":      round(accuracy, 4),
        "avg_pos_score": round(float(np.mean(pos_scores)), 4) if pos_scores else 0.0,
        "avg_neg_score": round(float(np.mean(neg_scores)), 4) if neg_scores else 0.0,
    }


# ─── Reranker Inference Class ──────────────────────────────────────────────────

class Reranker:
    """
    Wraps the fine-tuned cross-encoder for inference.
    Used in the full pipeline: dense_results → reranker → top_k.
    """

    def __init__(self, model_path: str = "checkpoints/reranker", max_length: int = 512):
        from sentence_transformers.cross_encoder import CrossEncoder
        print(f"  [Reranker] Loading: {model_path}")
        self.model      = CrossEncoder(model_path, max_length=max_length)
        self.model_path = model_path

    def rerank(self, query: str, passages: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Re-scores retrieved passages with the cross-encoder.

        Args:
            query:   The user's query
            passages: Candidates from dense retriever (top-20)
            top_k:   How many to return after reranking

        Returns:
            Re-ranked passages with "rerank_score" field added
        """
        if not passages:
            return []

        pairs  = [[query, p["text"]] for p in passages]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for p, score in zip(passages, scores):
            # 1. Base cross-encoder score
            final_score = float(score)
            
            # 2. Add retriever RRF boost
            final_score += p.get("rrf_score", 0.0) * 5.0
            
            # 3. Domain Bias: Huge boost for telecom documents
            if p.get("source") == "telecom_overlay":
                final_score += 1.5  # Ensure telecom docs always beat general ones if relevant
            
            p["rerank_score"] = round(final_score, 4)

        reranked = sorted(passages, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--eval",        action="store_true")
    parser.add_argument("--demo",        type=str, help="Demo query")
    parser.add_argument("--base-model",  default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--output-dir",  default="checkpoints/reranker")
    parser.add_argument("--epochs",      type=int,   default=2)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--max-samples", type=int,   default=8000)
    args = parser.parse_args()

    if args.train:
        train_reranker(
            base_model  = args.base_model,
            output_dir  = args.output_dir,
            num_epochs  = args.epochs,
            batch_size  = args.batch_size,
            max_samples = args.max_samples,
        )

    elif args.eval:
        _, val_data = build_reranker_dataset(max_samples=1000)
        from sentence_transformers.cross_encoder import CrossEncoder
        model   = CrossEncoder(args.output_dir)
        metrics = _evaluate_cross_encoder(model, val_data[:300])
        print("Reranker metrics:", metrics)

    elif args.demo:
        reranker = Reranker(args.output_dir)
        # Build mock candidates for demo
        candidates = [
            {"text": "You can dispute a charge by calling 198 or using the portal.", "section_id": "s1"},
            {"text": "Data plans are available from Rs. 99 to Rs. 999 per month.", "section_id": "s2"},
            {"text": "Billing disputes must be raised within 60 days.", "section_id": "s3"},
        ]
        results = reranker.rerank(args.demo, candidates, top_k=2)
        print(f"Query: {args.demo}")
        for r in results:
            print(f"  [{r['rerank_score']:.3f}] {r['text'][:80]}")
    else:
        parser.print_help()

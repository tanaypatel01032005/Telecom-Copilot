"""
src/retrieval/train_retriever.py

Week 2 — Fine-tune a Dense Retriever on MultiDoc2Dial triples.

Architecture:
  Base model  : sentence-transformers/all-MiniLM-L6-v2  (22M params, fast)
  Loss        : MultipleNegativesRankingLoss (MNRL)
               - treats every other positive in the batch as a negative
               - ideal for (query, positive) pairs
  Hard negs   : In-batch + explicitly mined hard negatives from retriever_train.jsonl
  Evaluation  : Recall@1, Recall@5, MRR@10 on MD2D validation triples

Why all-MiniLM-L6-v2?
  - Fast enough to embed the full KB (~3000+ passages) in seconds
  - Strong zero-shot baseline to fine-tune from
  - 384-dim embeddings → FAISS IndexFlatIP works well at this scale
  - Recommended in sentence-transformers docs for asymmetric retrieval

Training time estimates (Google Colab T4 GPU):
  - 10,000 triples, batch_size=32, 3 epochs ≈ 25–35 minutes
  - Can reduce to 5,000 triples + 2 epochs (~12 mins) if compute-constrained

Run:
    python -m src.retrieval.train_retriever --quick   (2000 samples, 1 epoch — for smoke test)
    python -m src.retrieval.train_retriever           (full training)
    python -m src.retrieval.train_retriever --eval    (evaluate saved model)
"""

import json
import os

os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import argparse
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple

random.seed(42)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_retriever_triples(
    path: str = "data/processed/retriever_train.jsonl",
    span_index_path: str = "data/processed/span_index.json",
    max_samples: int = 10000,
    val_ratio: float = 0.1,
) -> Tuple[List, List]:
    """
    Loads retriever training triples and converts them to
    sentence-transformers InputExample format.

    Each triple: (query, positive_text, [hard_negative_texts])

    Returns:
        train_examples, val_examples
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Retriever triples not found: {path}\n"
            "Run: python -m src.ingestion.training_data_builder first."
        )

    # Load span index for text lookup
    with open(span_index_path) as f:
        span_index = json.load(f)

    triples = []
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            triples.append(t)
            if len(triples) >= max_samples:
                break

    random.shuffle(triples)
    split = int(len(triples) * (1 - val_ratio))
    train_triples = triples[:split]
    val_triples   = triples[split:]

    print(f"  Loaded {len(triples)} triples → "
          f"{len(train_triples)} train / {len(val_triples)} val")

    return train_triples, val_triples, span_index


def triples_to_input_examples(triples: List[Dict], span_index: Dict):
    """
    Converts raw triples to sentence-transformers InputExample objects.
    Each example = (query, positive_text).
    Hard negatives are handled by MultipleNegativesRankingLoss in-batch.
    """
    from sentence_transformers import InputExample

    examples = []
    skipped  = 0

    for t in triples:
        query         = t.get("query", "").strip()
        positive_text = t.get("positive_text", "").strip()

        # Fallback: look up from span_index if positive_text is missing
        if not positive_text:
            pos_id        = t.get("positive_id", "")
            passage       = span_index.get(pos_id, {})
            positive_text = passage.get("text", "")

        if len(query) < 5 or len(positive_text) < 20:
            skipped += 1
            continue

        examples.append(InputExample(texts=[query, positive_text]))

    print(f"  Converted {len(examples)} examples ({skipped} skipped — too short)")
    return examples


# ─── Retrieval evaluation (Recall@k, MRR) ────────────────────────────────────

def evaluate_retriever(
    model,
    val_triples: List[Dict],
    span_index: Dict,
    k_values: List[int] = [1, 5, 10],
    batch_size: int = 64,
) -> Dict:
    """
    Computes Recall@k and MRR@10 on validation triples.

    For each query in val_triples:
      1. Embed the query
      2. Embed a candidate pool: positive + all hard negatives
      3. Rank by cosine similarity
      4. Check if positive is in top-k

    This is a "micro" eval — not full KB retrieval (that happens after FAISS indexing).
    """
    import torch
    import numpy as np

    model.eval()
    queries, positives, negatives_list = [], [], []

    for t in val_triples[:500]:   # cap at 500 for speed
        q   = t.get("query", "").strip()
        pos = t.get("positive_text", "")
        if not pos:
            pos = span_index.get(t.get("positive_id", ""), {}).get("text", "")
        negs = [
            span_index.get(nid, {}).get("text", "")
            for nid in t.get("hard_negatives", [])
        ]
        negs = [n for n in negs if n]

        if not q or not pos:
            continue
        queries.append(q)
        positives.append(pos)
        negatives_list.append(negs)

    print(f"  Evaluating on {len(queries)} val queries...")

    # Encode queries
    q_embs  = model.encode(queries, batch_size=batch_size,
                            show_progress_bar=False, convert_to_tensor=True)

    recalls = {k: 0 for k in k_values}
    mrr     = 0.0

    for i, (q_emb, pos, negs) in enumerate(zip(q_embs, positives, negatives_list)):
        # Build candidate set: [positive] + negatives (up to 9 negatives)
        candidates  = [pos] + negs[:9]
        cand_embs   = model.encode(candidates, batch_size=batch_size,
                                    show_progress_bar=False, convert_to_tensor=True)

        # Cosine similarity
        sims   = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), cand_embs
        ).cpu().numpy()
        ranked = np.argsort(-sims)   # descending
        pos_rank = int(np.where(ranked == 0)[0][0]) + 1   # rank of positive (1-indexed)

        for k in k_values:
            if pos_rank <= k:
                recalls[k] += 1
        mrr += 1.0 / pos_rank

    n = len(queries)
    metrics = {
        f"recall_at_{k}": round(recalls[k] / n, 4)
        for k in k_values
    }
    metrics["mrr_at_10"] = round(mrr / n, 4)
    return metrics


# ─── Main training function ───────────────────────────────────────────────────

def train_retriever(
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    triples_path:    str = "data/processed/retriever_train.jsonl",
    span_index_path: str = "data/processed/span_index.json",
    output_dir:      str = "checkpoints/retriever",
    max_samples:     int = 10000,
    num_epochs:      int = 3,
    batch_size:      int = 32,
    warmup_ratio:    float = 0.1,
    lr:              float = 2e-5,
    quick:           bool = False,
):
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from torch.utils.data import DataLoader

    if quick:
        max_samples = 2000
        num_epochs  = 1
        batch_size  = 16
        print("  [QUICK MODE] 2000 samples, 1 epoch")

    print(f"\n{'='*60}")
    print(f"  RETRIEVER FINE-TUNING")
    print(f"  Base model : {base_model_name}")
    print(f"  Samples    : {max_samples}")
    print(f"  Epochs     : {num_epochs}")
    print(f"  Batch size : {batch_size}")
    print(f"  LR         : {lr}")
    print(f"{'='*60}\n")

    # ── Load data ──────────────────────────────────────────────────
    train_triples, val_triples, span_index = load_retriever_triples(
        triples_path, span_index_path, max_samples=max_samples
    )

    # ── Load base model ────────────────────────────────────────────
    print(f"  Loading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name)

    # ── Evaluate BEFORE fine-tuning (baseline retriever numbers) ───
    print("\n  Evaluating BASE model (before fine-tuning)...")
    base_metrics = evaluate_retriever(model, val_triples, span_index)
    print("  Base model metrics:")
    for k, v in base_metrics.items():
        print(f"    {k:<20} {v:.4f}")

    # ── Build training examples ────────────────────────────────────
    train_examples = triples_to_input_examples(train_triples, span_index)
    train_loader   = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size
    )

    # ── Loss: MultipleNegativesRankingLoss ─────────────────────────
    # MNRL: for a batch of (q_i, p_i) pairs, treats all other p_j (j≠i)
    # as negatives for q_i. Scales with batch size — larger batch = harder negatives.
    # Reference: Henderson et al. (2017), Karpukhin et al. (2020)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # ── Warmup steps ──────────────────────────────────────────────
    total_steps  = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"\n  Total steps  : {total_steps}")
    print(f"  Warmup steps : {warmup_steps}")

    # ── Train ──────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n  Training started...")
    model.fit(
        train_objectives   = [(train_loader, train_loss)],
        epochs             = num_epochs,
        warmup_steps       = warmup_steps,
        optimizer_params   = {"lr": lr},
        output_path        = output_dir,
        save_best_model    = True,
        show_progress_bar  = True,
    )
    print(f"\n  Model saved → {output_dir}")

    # ── Evaluate AFTER fine-tuning ─────────────────────────────────
    print("\n  Evaluating FINE-TUNED model (after training)...")
    finetuned_model = SentenceTransformer(output_dir)
    ft_metrics      = evaluate_retriever(finetuned_model, val_triples, span_index)
    print("  Fine-tuned model metrics:")
    for k, v in ft_metrics.items():
        print(f"    {k:<20} {v:.4f}")

    # ── Delta table ────────────────────────────────────────────────
    print("\n  Improvement over base model:")
    print(f"  {'Metric':<20} {'Base':>8} {'Fine-tuned':>12} {'Delta':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*8}")
    for k in base_metrics:
        b  = base_metrics[k]
        ft = ft_metrics[k]
        print(f"  {k:<20} {b:>8.4f} {ft:>12.4f} {ft-b:>+8.4f}")

    # ── Save metrics report ────────────────────────────────────────
    report = {
        "base_model":       base_model_name,
        "training_config": {
            "max_samples": max_samples,
            "num_epochs":  num_epochs,
            "batch_size":  batch_size,
            "lr":          lr,
        },
        "before_finetuning": base_metrics,
        "after_finetuning":  ft_metrics,
        "delta": {k: round(ft_metrics[k] - base_metrics[k], 4) for k in base_metrics},
    }
    report_path = Path(output_dir) / "retriever_metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Metrics saved → {report_path}")
    return finetuned_model, report


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",      action="store_true",
                        help="Quick smoke test (2000 samples, 1 epoch)")
    parser.add_argument("--eval",       action="store_true",
                        help="Evaluate saved model only (no training)")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--max-samples",type=int,   default=10000)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--output-dir", default="checkpoints/retriever")
    args = parser.parse_args()

    if args.eval:
        from sentence_transformers import SentenceTransformer
        import json
        span_idx_path = "data/processed/span_index.json"
        with open(span_idx_path) as f:
            span_index = json.load(f)
        _, val_triples, _ = load_retriever_triples(
            "data/processed/retriever_train.jsonl", span_idx_path
        )
        model   = SentenceTransformer(args.output_dir)
        metrics = evaluate_retriever(model, val_triples, span_index)
        print("Saved model metrics:", metrics)
    else:
        train_retriever(
            base_model_name = args.base_model,
            num_epochs      = args.epochs,
            batch_size      = args.batch_size,
            max_samples     = args.max_samples,
            lr              = args.lr,
            output_dir      = args.output_dir,
            quick           = args.quick,
        )

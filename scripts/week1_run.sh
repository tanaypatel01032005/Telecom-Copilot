#!/usr/bin/env bash
# =============================================================================
#  Week 1 Master Script — Telecom Copilot (Real Dataset Edition)
#  Run: bash scripts/week1_run.sh
# =============================================================================
set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   TELECOM COPILOT — WEEK 1 (Real Dataset)               ║"
echo "║   MultiDoc2Dial + SHP-2 + Telecom Overlay               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 0: Install dependencies ────────────────────────────────
echo "──── [0/5] Installing core dependencies ────"
pip install datasets transformers sentencepiece anthropic \
            torch tqdm --quiet
echo "  Done."

# ── Step 1: Download real datasets ───────────────────────────────
echo ""
echo "──── [1/5] Downloading MultiDoc2Dial + SHP-2 ────"
echo "  (This downloads ~500MB — takes 3-5 mins on Colab)"
python scripts/download_datasets.py

# ── Step 2: Build KB from real documents ─────────────────────────
echo ""
echo "──── [2/5] Building KB (MD2D docs + telecom overlay) ────"
python -m src.ingestion.kb_builder

# ── Step 3: Extract training data from dialogues ─────────────────
echo ""
echo "──── [3/5] Extracting training data from dialogues ────"
echo "  (retriever triples, generator SFT pairs, DPO pairs, test set)"
python -m src.ingestion.training_data_builder

# ── Step 4: Run baseline system ──────────────────────────────────
echo ""
echo "──── [4/5] Running baseline on test set ────"
python -m src.baseline.baseline_system --eval

# ── Step 5: Compute baseline metrics ─────────────────────────────
echo ""
echo "──── [5/5] Computing and saving baseline metrics ────"
python -m src.evaluation.evaluator \
  --results data/processed/baseline_results.jsonl \
  --label baseline

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   WEEK 1 COMPLETE                                       ║"
echo "║                                                          ║"
echo "║   Key files produced:                                    ║"
echo "║   data/raw/multidoc2dial/         ← real dataset        ║"
echo "║   data/raw/shp2/                  ← real preferences    ║"
echo "║   data/processed/kb_passages.jsonl                       ║"
echo "║   data/processed/retriever_train.jsonl  (Week 2 input)  ║"
echo "║   data/processed/generator_sft_train.jsonl (Week 3)     ║"
echo "║   data/processed/dpo_pairs.jsonl        (Week 4 input)  ║"
echo "║   data/processed/test_cases.jsonl       (eval set)      ║"
echo "║   data/processed/baseline_results.jsonl                  ║"
echo "║   data/processed/baseline_results.eval.json  ← SAVE THIS║"
echo "║                                                          ║"
echo "║   Next: Week 2 — Fine-tune retriever on retriever_train  ║"
echo "╚══════════════════════════════════════════════════════════╝"

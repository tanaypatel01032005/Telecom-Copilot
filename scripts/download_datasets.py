"""
scripts/download_datasets.py

Downloads MultiDoc2Dial and SHP-2 from HuggingFace and saves them locally.
Run this ONCE before anything else.

Usage:
    python scripts/download_datasets.py

Saves to:
    data/raw/multidoc2dial/     ← dialogue + document splits
    data/raw/shp2/              ← preference pairs (filtered subset)
"""

# =========================================================
# FORCE HUGGINGFACE CACHE TO D DRIVE
# =========================================================

import os

HF_CACHE_PATH = "D:/huggingface"

os.environ["HF_HOME"] = HF_CACHE_PATH
os.environ["HF_DATASETS_CACHE"] = f"{HF_CACHE_PATH}/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_PATH}/transformers"

print("\n==================================================")
print(" HuggingFace Cache Configuration")
print("==================================================")
print(f" HF_HOME              : {os.environ['HF_HOME']}")
print(f" HF_DATASETS_CACHE    : {os.environ['HF_DATASETS_CACHE']}")
print(f" TRANSFORMERS_CACHE   : {os.environ['TRANSFORMERS_CACHE']}")
print("==================================================")
print(" Cached files will now use D drive.")
print(" Existing cache will be reused if available.")
print("==================================================\n")

import json
from pathlib import Path


def download_multidoc2dial():
    print("\n── Downloading MultiDoc2Dial ──────────────────────────")

    from datasets import load_dataset

    # ── Dialogue split (train / validation) ──────────────────────
    # Fields per turn:
    #   role        : "agent" | "user"
    #   utterance   : the actual text spoken
    #   da          : dialogue act, e.g. "respond_solution"
    #   references  : [{"id_sp": "6", "label": "solution", "doc_id": "..."}]
    #
    # We care ONLY about agent turns with da=="respond_solution"
    # because those are grounded answers with citations.

    print("Checking HuggingFace cache for MultiDoc2Dial...")

    dial_ds = load_dataset(
        "IBM/multidoc2dial",
        "dialogue_domain",
        trust_remote_code=True,
        download_mode="reuse_dataset_if_exists"
    )

    print("✓ Dialogue dataset loaded.")
    print("✓ Cache reused if already downloaded.")

    print(f"  Dialogue train : {len(dial_ds['train']):,} turns")
    print(f"  Dialogue val   : {len(dial_ds['validation']):,} turns")

    # ── Document split ────────────────────────────────────────────
    # Fields per document:
    #   doc_id      : e.g. "Top 5 DMV Mistakes and How to Avoid Them#3_0"
    #   title       : human-readable title
    #   domain      : "dmv" | "va" | "ssa" | "studentaid"
    #   doc_text    : full document text
    #   spans       : {"1": {"id_sp":"1","tag":"p","start_sp":0,"end_sp":143,"text":"..."}, ...}

    doc_ds = load_dataset(
        "IBM/multidoc2dial",
        "document_domain",
        trust_remote_code=True,
        download_mode="reuse_dataset_if_exists"
    )

    print("✓ Document dataset loaded.")
    print("✓ Cache reused if already downloaded.")

    print(f"  Documents      : {len(doc_ds['train']):,}")

    # Save both
    out = Path("data/raw/multidoc2dial")
    out.mkdir(parents=True, exist_ok=True)

    print("\nSaving MultiDoc2Dial locally...")

    dial_ds.save_to_disk(str(out / "dialogues"))
    doc_ds.save_to_disk(str(out / "documents"))

    print("✓ MultiDoc2Dial saved successfully.")

    # Print one real example so you can see the structure
    print("\n  Sample dialogue record:")

    sample = dial_ds["train"][0]

    print("Available Keys:")
    print(sample.keys())

    print("\nSample:")
    print(json.dumps(sample, indent=4, default=str)[:1500])

    print("\n  Sample document span:")
    print("\n  Sample document:")

    doc_sample = doc_ds["train"][0]

    print("Available Keys:")
    print(doc_sample.keys())

    print("\nSample Document:")
    print(json.dumps({
        "doc_id": doc_sample.get("doc_id"),
        "domain": doc_sample.get("domain"),
        "title": doc_sample.get("title"),
        "sample_spans": doc_sample.get("spans", [])[:2]
    }, indent=4, default=str))

    return dial_ds, doc_ds


def download_shp2(n_samples: int = 5000):
    print("\n── Downloading SHP-2 (customer-service subset) ─────────")

    from datasets import load_dataset

    # SHP-2 fields:
    #   history       : the question / prompt (Reddit post)
    #   human_ref_A   : first response candidate
    #   human_ref_B   : second response candidate
    #   labels        : 1 if A is preferred, 0 if B is preferred
    #   score_A/B     : Reddit upvote scores
    #   score_ratio   : ratio of winner to loser score
    #   domain        : subreddit name (e.g. "legaladvice", "personalfinance")

    # Filter: keep only customer-service-adjacent domains
    USEFUL_DOMAINS = {
        "legaladvice",
        "personalfinance",
        "techsupport",
        "explainlikeimfive",
        "Advice",
        "NoStupidQuestions",
        "AskTechnology",
        "Android",
        "ios"
    }

    print("Checking HuggingFace cache for SHP-2...")
    print("If dataset already exists, cache will be reused.")
    print("Only missing files will download.\n")

    ds = load_dataset(
        "stanfordnlp/SHP-2",
        split="train",
        trust_remote_code=True,
        download_mode="reuse_dataset_if_exists"
    )

    print("✓ SHP-2 dataset loaded successfully.")
    print("✓ Cache reused if already downloaded.")

    print(f"  Total SHP-2 samples: {len(ds):,}")

    # DEBUG: Show first few domains
    print("\nChecking first 10 domain names from dataset:")

    for i in range(10):
        try:
            print(f"  {i+1}. {ds[i]['domain']}")
        except Exception:
            pass

    # Filter to useful domains AND strong preference signal (ratio > 2.0)

    print("\nFiltering useful domains...")

    filtered = ds.filter(
        lambda x: str(x["domain"]).lower() in {
            d.lower() for d in USEFUL_DOMAINS
        }
        and float(x["score_ratio"]) >= 2.0,
        num_proc=4
    )

    print(f"✓ After domain+ratio filter: {len(filtered):,}")

    # Take a capped subset to keep disk/memory usage reasonable

    subset = filtered.select(range(min(n_samples, len(filtered))))

    print(f"✓ Using subset: {len(subset):,} samples")

    out = Path("data/raw/shp2")
    out.mkdir(parents=True, exist_ok=True)

    print("\nSaving SHP-2 subset locally...")

    subset.save_to_disk(str(out))

    print("✓ SHP-2 saved successfully.")

    # Print sample safely

    if len(subset) > 0:

        print("\n  Sample SHP-2 record:")

        s = subset[0]

        print(json.dumps({
            "domain":       s["domain"],
            "history":      s["history"][:120] + "...",
            "human_ref_A":  s["human_ref_A"][:80] + "...",
            "human_ref_B":  s["human_ref_B"][:80] + "...",
            "labels":       s["labels"],          # 1 = A preferred
            "score_ratio":  s["score_ratio"]
        }, indent=4))

    else:

        print("\n⚠ No rows matched filter.")
        print("⚠ Please check actual domain names printed above.")

    return subset


if __name__ == "__main__":

    print("\n==================================================")
    print(" DATASET DOWNLOAD STARTED ")
    print("==================================================")

    dial_ds, doc_ds = download_multidoc2dial()

    shp2_ds = download_shp2(n_samples=5000)

    print("\n✓ All datasets downloaded.")
    print("  Next step: python src/ingestion/kb_builder.py")
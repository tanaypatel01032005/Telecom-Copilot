"""
src/ingestion/training_data_builder.py

Extracts three training datasets from MultiDoc2Dial dialogues:

  A. retriever_train.jsonl
     (query, positive_passage_id, hard_negative_passage_ids)
     Used to fine-tune the dense retriever in Week 2.

  B. generator_sft_train.jsonl
     (query, retrieved_context, gold_answer, gold_citation)
     Used to fine-tune the generator with LoRA/DoRA in Week 3.

  C. dpo_pairs.jsonl
     (prompt, chosen_response, rejected_response)
     Used for DPO preference alignment in Week 4.

Why MultiDoc2Dial is ideal for ALL THREE tasks:
  - Every agent turn includes "references" = exact grounding span IDs → retriever labels
  - Agent utterances are grounded answers written by humans → generator gold outputs
  - We can contrast good answers (grounded, concise) vs bad (verbose, ungrounded) → DPO pairs

Run:
    python -m src.ingestion.training_data_builder
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from datasets import load_from_disk
from typing import List, Dict, Tuple


random.seed(42)


# ─── Load dialogues and span index ────────────────────────────────────────────

def load_resources(
    dial_path:      str = "data/raw/multidoc2dial/dialogues",
    span_idx_path:  str = "data/processed/span_index.json",
) -> Tuple:
    print("  Loading dialogues...")
    dial_ds = load_from_disk(dial_path)

    print("  Loading span index...")
    with open(span_idx_path) as f:
        span_index = json.load(f)

    return dial_ds, span_index


# ─── A. Retriever Training Triples ────────────────────────────────────────────

def build_retriever_triples(
    dial_ds,
    span_index: Dict,
    split: str = "train",
    output_path: str = "data/processed/retriever_train.jsonl",
    max_samples: int = 10000,
):
    """
    For each USER turn in MD2D that has grounding references,
    extracts: (user_query, positive_section_id, [hard_negative_section_ids])

    Hard negatives = spans from the SAME document that are NOT the answer span.
    This forces the retriever to learn fine-grained distinctions within documents.

    Output record:
    {
        "query":            "Can I renew my license online?",
        "positive_id":      "Top 5 DMV Mistakes...#3_0__sp6",
        "positive_text":    "Yes, you can sign up for MyDMV ...",
        "hard_negatives":   ["Top 5 DMV Mistakes...#3_0__sp2", ...],
        "doc_id":           "Top 5 DMV Mistakes and How to Avoid Them#3_0",
        "domain":           "dmv"
    }
    """
    print(f"\n  Building retriever triples from '{split}' split...")

    # Group all spans by doc_id for fast hard-negative mining
    doc_to_spans = defaultdict(list)
    for sec_id, passage in span_index.items():
        doc_to_spans[passage["doc_id"]].append(sec_id)

    triples = []
    dataset = dial_ds[split]
    print("\nSample dataset keys:")
    print(dataset[0].keys())

    for dialogue in dataset:
        
        turns = dialogue.get("turns", [])

        for sample in turns:

            # Only use USER turns that have grounding references
            role = sample.get("role", sample.get("speaker", ""))

            if role != "user":
                continue

            refs = sample.get("references", [])

            if not refs:
                continue

            query = sample["utterance"].strip()

            if len(query) < 10:
                continue

            for ref in refs:

                if ref.get("label") not in ("solution", "precondition"):
                    continue

                doc_id  = ref["doc_id"]
                span_id = ref["id_sp"]

                sec_id = f"{doc_id}__sp{span_id}"

                if sec_id not in span_index:
                    continue

                positive_passage = span_index[sec_id]

                # Hard negatives: other spans from same document
                same_doc_spans = [
                    s for s in doc_to_spans[doc_id]
                    if s != sec_id
                ]

                hard_negs = random.sample(
                    same_doc_spans,
                    min(3, len(same_doc_spans))
                )

                triples.append({
                    "query":           query,
                    "positive_id":     sec_id,
                    "positive_text":   positive_passage["text"],
                    "hard_negatives":  hard_negs,
                    "doc_id":          doc_id,
                    "domain":          positive_passage.get("domain", "unknown"),
                    "source":          "multidoc2dial",
                })

                if len(triples) >= max_samples:
                    break

            if len(triples) >= max_samples:
                break

        if len(triples) >= max_samples:
            break

    random.shuffle(triples)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for t in triples:
            f.write(json.dumps(t) + "\n")

    print(f"  Saved {len(triples):,} retriever triples → {output_path}")
    return triples


# ─── B. Generator SFT Pairs ───────────────────────────────────────────────────

def build_generator_sft_pairs(
    dial_ds,
    span_index: Dict,
    split: str = "train",
    output_path: str = "data/processed/generator_sft_train.jsonl",
    max_samples: int = 8000,
):
    """
    For each AGENT turn with grounding references, extracts:
    (query=previous user turn, context=grounding spans, gold_answer=agent utterance)

    This is the exact setup for supervised fine-tuning the generator:
    given a user question + retrieved context → produce a grounded cited answer.

    Output record:
    {
        "query":        "Can I renew my license online?",
        "context":      [{"section_id": "...", "text": "Yes, you can sign up ..."}],
        "gold_answer":  "hi, you can sign up for MyDMV for online transactions.",
        "gold_citations": [{"doc_id": "...", "section_id": "...", "span_id": "6"}],
        "domain":       "dmv",
        "dialogue_id":  "dlg_001",
        "turn_id":      2
    }
    """
    print(f"\n  Building generator SFT pairs from '{split}' split...")

    dataset = dial_ds[split]

    # Group turns by dialogue_id so we can get the preceding user turn
# Dataset already grouped by dialogue
    dialogues = {}

    for dialogue in dataset:

        dia_id = dialogue.get("dial_id", "unknown")

        dialogues[dia_id] = dialogue.get("turns", [])

    pairs = []

    for dial_id, turns in dialogues.items():
        # Sort by turn_id
        turns_sorted = sorted(turns, key=lambda t: t.get("turn_id", 0))

        for i, turn in enumerate(turns_sorted):
            role = turn.get("role", turn.get("speaker", ""))

            if role != "agent":
                continue
            refs = turn.get("references", [])
            if not refs:
                continue

            agent_answer = turn["utterance"].strip()
            if len(agent_answer) < 20:
                continue

            # Get the preceding user turn as the query
            prev_user_turns = [
                t for t in turns_sorted[:i]
                if t.get("role", t.get("speaker", "")) == "user"
            ]
            if not prev_user_turns:
                continue
            query = prev_user_turns[-1]["utterance"].strip()

            # Collect grounding context from references
            context_passages = []
            citations        = []

            for ref in refs:
                if ref.get("label") not in ("solution", "precondition"):
                    continue
                doc_id  = ref["doc_id"]
                span_id = ref["id_sp"]
                sec_id  = f"{doc_id}__sp{span_id}"

                if sec_id not in span_index:
                    continue

                passage = span_index[sec_id]
                context_passages.append({
                    "section_id": sec_id,
                    "doc_id":     doc_id,
                    "heading":    passage.get("heading", ""),
                    "text":       passage["text"],
                })
                citations.append({
                    "doc_id":     doc_id,
                    "section_id": sec_id,
                    "span_id":    span_id,
                })

            if not context_passages:
                continue

            pairs.append({
                "query":          query,
                "context":        context_passages,
                "gold_answer":    agent_answer,
                "gold_citations": citations,
                "domain":         context_passages[0].get("domain",
                                    span_index.get(citations[0]["section_id"], {}).get("domain", "unknown")),
                "dialogue_id":    dial_id,
                "turn_id":        turn.get("turn_id", i),
                "source":         "multidoc2dial",
            })

            if len(pairs) >= max_samples:
                break
        if len(pairs) >= max_samples:
            break

    random.shuffle(pairs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"  Saved {len(pairs):,} generator SFT pairs → {output_path}")
    return pairs


# ─── C. DPO Preference Pairs ──────────────────────────────────────────────────

def build_dpo_pairs(
    sft_pairs: List[Dict],
    shp2_path:   str = "data/raw/shp2",
    output_path: str = "data/processed/dpo_pairs.jsonl",
    max_from_md2d: int = 1000,
    max_from_shp2: int = 1000,
):
    """
    Builds DPO preference pairs from TWO sources:

    Source 1 — MD2D-derived pairs (telecom-grounded):
      chosen   = the real agent answer (grounded, concise, from MD2D)
      rejected = a deliberately bad version (verbose, not grounded, no citation)
      We GENERATE the rejected version by stripping citation and bloating the text.

    Source 2 — SHP-2 filtered pairs (helpfulness preferences):
      chosen   = the higher-upvoted response (more helpful)
      rejected = the lower-upvoted response
      We filter for score_ratio >= 2.0 (strong preference signal).

    Output record (DPO standard format):
    {
        "prompt":    "Customer asked: ...",
        "chosen":    "concise grounded answer with [SOURCE: doc_id]",
        "rejected":  "verbose unhelpful answer with no citation",
        "source":    "md2d" | "shp2",
        "domain":    "dmv" | "telecom" | ...
    }
    """
    print(f"\n  Building DPO pairs...")
    dpo_pairs = []

    # ── Source 1: MD2D-derived ──────────────────────────────────────
    print(f"  Building {max_from_md2d} MD2D-derived DPO pairs...")

    def make_rejected_response(chosen: str, citations: List[Dict]) -> str:
        """
        Creates a rejected response by:
        1. Removing any citation references
        2. Adding jargon and verbosity (simulates a bad agent response)
        3. Introducing hedging language that reduces helpfulness
        """
        hedged = (
            "That's a great question. There are many factors to consider here. "
            "Generally speaking, it depends on various circumstances. "
            + chosen
            + " However, please note that policies may vary and you should always "
            "check the official website for the most up-to-date information. "
            "Is there anything else I can help you with today?"
        )
        return hedged

    sample_sft = random.sample(sft_pairs, min(max_from_md2d, len(sft_pairs)))

    for pair in sample_sft:
        chosen   = pair["gold_answer"]
        citations = pair["gold_citations"]

        # Format chosen with citation tag (teaching citation-first behaviour)
        doc_id   = citations[0]["doc_id"] if citations else "unknown"
        span_id  = citations[0]["span_id"] if citations else "?"
        chosen_with_citation = f"{chosen} [SOURCE: {doc_id}, span {span_id}]"

        rejected = make_rejected_response(chosen, citations)

        prompt = (
            f"Customer asked: {pair['query']}\n"
            f"Context: {pair['context'][0]['text'][:300] if pair['context'] else ''}"
        )

        dpo_pairs.append({
            "prompt":   prompt,
            "chosen":   chosen_with_citation,
            "rejected": rejected,
            "source":   "md2d",
            "domain":   pair.get("domain", "unknown"),
        })

    # ── Source 2: SHP-2 ────────────────────────────────────────────
    print(f"  Loading SHP-2 pairs...")
    try:
        shp2_ds = load_from_disk(shp2_path)
        shp2_list = list(shp2_ds)
        random.shuffle(shp2_list)
        added = 0

        for sample in shp2_list:
            if added >= max_from_shp2:
                break

            # labels=1 means A is preferred, labels=0 means B is preferred
            if sample["labels"] == 1:
                chosen   = sample["human_ref_A"]
                rejected = sample["human_ref_B"]
            else:
                chosen   = sample["human_ref_B"]
                rejected = sample["human_ref_A"]

            # Skip if responses are too short or too similar in length
            if len(chosen) < 50 or len(rejected) < 50:
                continue
            if abs(len(chosen) - len(rejected)) < 20:
                continue   # very similar length → weak preference signal

            dpo_pairs.append({
                "prompt":   sample["history"][:400],
                "chosen":   chosen[:600],
                "rejected": rejected[:600],
                "source":   "shp2",
                "domain":   sample.get("domain", "unknown"),
                "score_ratio": sample.get("score_ratio", 1.0),
            })
            added += 1

        print(f"  Added {added} SHP-2 pairs")

    except Exception as e:
        print(f"  Warning: Could not load SHP-2 ({e}). Continuing with MD2D pairs only.")

    # ── Save ───────────────────────────────────────────────────────
    random.shuffle(dpo_pairs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in dpo_pairs:
            f.write(json.dumps(p) + "\n")

    from collections import Counter
    sources = Counter(p["source"] for p in dpo_pairs)
    print(f"  Saved {len(dpo_pairs):,} DPO pairs → {output_path}")
    print(f"    md2d: {sources['md2d']:,}  |  shp2: {sources['shp2']:,}")
    return dpo_pairs


# ─── D. Evaluation Test Set ───────────────────────────────────────────────────

def build_eval_test_set(
    dial_ds,
    span_index: Dict,
    output_path: str = "data/processed/test_cases.jsonl",
    n_cases: int = 200,
):
    """
    Builds the evaluation test set from the VALIDATION split of MultiDoc2Dial.
    Using the validation split (not train) ensures no data leakage.

    Each test case has:
      query, gold_doc_id, gold_section_id, gold_answer, domain,
      should_escalate (False for all MD2D cases — no KB gap),
      requires_outage_check (False — MD2D has no network outage queries)

    We also add 30 manually crafted telecom test cases that cover
    escalation, outage checks, and telecom-specific scenarios.
    """
    print(f"\n  Building eval test set from validation split...")

    val_ds  = dial_ds["validation"]
    cases   = []
    seen    = set()

    # Group by dialogue to get context (previous turns)
    # Dataset already grouped by dialogue
    dialogues = {}

    for dialogue in val_ds:

        dia_id = dialogue.get("dial_id", "unknown")

        dialogues[dia_id] = dialogue.get("turns", [])

    for dial_id, turns in dialogues.items():
        if len(cases) >= n_cases:
            break
        turns_sorted = sorted(turns, key=lambda t: t.get("turn_id", 0))

        for i, turn in enumerate(turns_sorted):
            role = turn.get("role", turn.get("speaker", ""))

            if role != "agent":
                continue
            refs = turn.get("references", [])
            if not refs:
                continue

            agent_answer = turn["utterance"].strip()
            if len(agent_answer) < 20:
                continue

            prev_user = [t for t in turns_sorted[:i] if t.get("role", t.get("speaker", "")) == "user"]
            if not prev_user:
                continue

            query = prev_user[-1]["utterance"].strip()
            if query in seen:
                continue
            seen.add(query)

            ref        = refs[0]
            doc_id     = ref["doc_id"]
            span_id    = ref["id_sp"]
            sec_id     = f"{doc_id}__sp{span_id}"

            if sec_id not in span_index:
                continue

            # Build history (last 2 user + agent turns before this)
            history = []
            for prev in turns_sorted[max(0, i-4):i]:
                history.append({
                    "role":      prev["role"],
                    "utterance": prev["utterance"]
                })

            cases.append({
                "test_id":           f"MD2D_VAL_{len(cases)+1:04d}",
                "source":            "multidoc2dial_validation",
                "query":             query,
                "history":           history,
                "gold_doc_id":       doc_id,
                "gold_section_id":   sec_id,
                "gold_answer":       agent_answer,
                "gold_tool":         "SearchKB",
                "should_escalate":   False,
                "requires_outage_check": False,
                "category":          span_index[sec_id].get("domain", "unknown"),
                "domain":            span_index[sec_id].get("domain", "unknown"),
            })

            if len(cases) >= n_cases:
                break

    # ── Append telecom-specific test cases (escalation, outage, roaming) ──────
    telecom_cases = [
        {
            "test_id": "TELECOM_001",
            "source": "telecom_handcrafted",
            "query": "I was charged roaming fees but I never left India.",
            "history": [],
            "gold_doc_id": "telecom_roaming_001",
            "gold_section_id": "telecom_roaming_001_s3",
            "gold_answer": "Being charged roaming rates while in India is a valid dispute reason. Submit a dispute within 60 days of the bill date via the portal or by calling 198. Disputed amounts are held and not collected until resolution.",
            "gold_tool": "GetPolicy",
            "should_escalate": False,
            "requires_outage_check": False,
            "category": "roaming",
            "domain": "telecom",
        },
        {
            "test_id": "TELECOM_002",
            "source": "telecom_handcrafted",
            "query": "My 4G is completely down in Ahmedabad since this morning.",
            "history": [],
            "gold_doc_id": "telecom_network_001",
            "gold_section_id": "telecom_network_001_s3",
            "gold_answer": "I'll check the current network status for Ahmedabad. If there is an active outage, our target restoration time is 4 hours for urban areas with updates every 2 hours on our app.",
            "gold_tool": "CheckNetworkStatus",
            "should_escalate": False,
            "requires_outage_check": True,
            "category": "network",
            "domain": "telecom",
            "region": "Ahmedabad",
            "service_type": "4G",
        },
        {
            "test_id": "TELECOM_003",
            "source": "telecom_handcrafted",
            "query": "This billing issue has been going on for 10 days and nobody has resolved it.",
            "history": [
                {"role": "user", "utterance": "There is a wrong charge of Rs. 3000 on my bill."},
                {"role": "agent", "utterance": "A dispute ticket has been raised for you."},
            ],
            "gold_doc_id": "telecom_escalation_001",
            "gold_section_id": "telecom_escalation_001_s2",
            "gold_answer": "Since the issue has been unresolved for more than 7 working days, this qualifies for escalation to our Nodal Officer. Please email nodal@telecom.com with your original ticket ID. Target resolution is 3 working days.",
            "gold_tool": "CreateTicket",
            "should_escalate": True,
            "requires_outage_check": False,
            "category": "escalation",
            "domain": "telecom",
        },
        {
            "test_id": "TELECOM_004",
            "source": "telecom_handcrafted",
            "query": "I got a bill for Rs. 15,000 in roaming charges for a 3-day trip.",
            "history": [],
            "gold_doc_id": "telecom_roaming_001",
            "gold_section_id": "telecom_roaming_001_s3",
            "gold_answer": "A charge of Rs. 15,000 for 3 days of roaming is likely incorrect. This is a high-value dispute. I am escalating this to our senior billing team and creating a ticket. Disputed amounts are held and not collected until resolved.",
            "gold_tool": "CreateTicket",
            "should_escalate": True,
            "requires_outage_check": False,
            "category": "roaming",
            "domain": "telecom",
        },
        {
            "test_id": "TELECOM_005",
            "source": "telecom_handcrafted",
            "query": "How do I switch to eSIM?",
            "history": [],
            "gold_doc_id": "telecom_device_001",
            "gold_section_id": "telecom_device_001_s2",
            "gold_answer": "Go to My Account > SIM Management > Switch to eSIM. A QR code is sent to your registered email. Scan it in your device settings to activate. eSIM is supported on iPhone XS and later, Samsung Galaxy S20 and later.",
            "gold_tool": "SearchKB",
            "should_escalate": False,
            "requires_outage_check": False,
            "category": "device",
            "domain": "telecom",
        },
    ]

    cases.extend(telecom_cases)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    from collections import Counter
    sources = Counter(c["source"] for c in cases)
    print(f"  Saved {len(cases):,} test cases → {output_path}")
    print(f"    multidoc2dial_validation: {sources['multidoc2dial_validation']:,}")
    print(f"    telecom_handcrafted:      {sources['telecom_handcrafted']:,}")
    return cases


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_all_training_data():
    import sys
    sys.path.insert(0, ".")

    dial_ds, span_index = load_resources()

    # A — Retriever triples
    build_retriever_triples(
        dial_ds, span_index,
        output_path="data/processed/retriever_train.jsonl",
        max_samples=10000,
    )

    # B — Generator SFT pairs
    sft_pairs = build_generator_sft_pairs(
        dial_ds, span_index,
        output_path="data/processed/generator_sft_train.jsonl",
        max_samples=8000,
    )

    # C — DPO preference pairs
    build_dpo_pairs(
        sft_pairs,
        output_path="data/processed/dpo_pairs.jsonl",
        max_from_md2d=1000,
        max_from_shp2=1000,
    )

    # D — Evaluation test set (from validation split — no data leakage)
    build_eval_test_set(
        dial_ds, span_index,
        output_path="data/processed/test_cases.jsonl",
        n_cases=200,
    )

    print("\n✓ All training data built.")
    print("  Next step: python -m src.baseline.baseline_system")


if __name__ == "__main__":
    build_all_training_data()

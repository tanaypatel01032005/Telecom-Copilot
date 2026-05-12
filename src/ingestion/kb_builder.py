"""
src/ingestion/kb_builder.py

Builds the Telecom KB in two layers:

  Layer 1 — Real MultiDoc2Dial documents
    MultiDoc2Dial has 488 real support documents across 4 domains:
    dmv, va, ssa, studentaid.  These are genuine policy/procedure articles
    written by government agencies.  We keep them verbatim as the retrieval
    corpus and give each span a doc_id + section_id — exactly like a real
    telecom KB would look.

    Why?  Because the dialogue turns in MultiDoc2Dial are GROUNDED IN these
    documents, so using them as the KB means our retriever training signal
    (query → correct span) is directly aligned with what is actually in the KB.
    This is the scientifically correct setup.

  Layer 2 — Telecom overlay documents (your handcrafted articles)
    These 19 telecom-specific articles are added as supplementary KB entries.
    They give domain flavour and are used in the demo / serving layer.
    They are NOT used as training signal for the retriever or generator —
    MultiDoc2Dial's real dialogue data handles that.

Output:
    data/processed/kb_passages.jsonl   ← every passage ready for FAISS
    data/processed/doc_index.json      ← doc_id → title / domain / text lookup
    data/processed/span_index.json     ← span_id → passage record
    data/processed/dataset_stats.json  ← counts for your report

Run:
    python -m src.ingestion.kb_builder
"""

import json
import hashlib
from pathlib import Path
from datasets import load_from_disk
from typing import Dict, List


# ─── Layer 1: MultiDoc2Dial document processing ───────────────────────────────

def load_multidoc2dial_documents(docs_path: str) -> List[Dict]:
    """
    Loads the document_domain split of MultiDoc2Dial and converts
    each span into a passage record matching our schema.

    MultiDoc2Dial document structure:
        doc_id   : "Title of Article#domain_number"
        title    : human-readable title
        domain   : "dmv" | "va" | "ssa" | "studentaid"
        doc_text : full text
        spans    : {"1": {"id_sp":"1", "tag":"p/li/h2",
                          "start_sp": int, "end_sp": int,
                          "text": str, "title": str}, ...}

    We treat each span as one passage (retrieval unit).
    This matches how MultiDoc2Dial's baseline models index the corpus.
    """
    print("  Loading MultiDoc2Dial documents...")
    doc_ds = load_from_disk(docs_path)

    # Map MD2D domains to telecom-adjacent category labels
    # This is a mapping for readability in citations — the content is unchanged
    DOMAIN_TO_CATEGORY = {
        "dmv":         "procedures",   # licences, renewals → analogous to account/device
        "va":          "benefits",     # claims, eligibility → analogous to plans/billing
        "ssa":         "benefits",     # social security → analogous to billing/account
        "studentaid":  "procedures",   # aid applications → analogous to plans/account
    }

    passages = []
    doc_index = {}

    for doc in doc_ds["train"]:
        doc_id   = doc["doc_id"]
        title    = doc["title"]
        domain   = doc["domain"]
        category = DOMAIN_TO_CATEGORY.get(domain, "general")
        doc_text = doc["doc_text"]

        # Store in doc_index for fast lookup at inference
        doc_index[doc_id] = {
            "doc_id":   doc_id,
            "title":    title,
            "domain":   domain,
            "category": category,
            "source":   "multidoc2dial",
        }

        # Convert spans → passages 

        spans = doc["spans"]

        # Handle both dict-style and list-style spans
        if isinstance(spans, dict):
            span_iter = spans.values()
        else:
            span_iter = spans

        for span in span_iter:

            # Some dataset versions use text_sp instead of text
            text = (
                span.get("text", "")
                or span.get("text_sp", "")
            ).strip()

            if len(text) < 30:          # skip trivially short spans
                continue

            if span.get("tag") in ("h1", "h2", "h3"):  # skip headers-only
                if len(text) < 60:
                    continue

            section_id = f"{doc_id}__sp{span['id_sp']}"
            heading = span.get("title", title)

            passage = {
                "passage_id":  hashlib.md5(section_id.encode()).hexdigest()[:12],
                "doc_id":      doc_id,
                "section_id":  section_id,
                "span_id":     span["id_sp"],           # original MD2D span id
                "title":       title,
                "heading":     heading,
                "category":    category,
                "domain":      domain,                   # original MD2D domain
                "source":      "multidoc2dial",
                "tags":        [domain, category],
                "text":        text,
                "full_text":   f"{heading}. {text}" if heading != title else text,
                "char_start":  span.get("start_sp", 0),
                "char_end":    span.get("end_sp", len(text)),
                "word_count":  len(text.split()),
            }

            passages.append(passage)

    print(f"  Loaded {len(passages):,} passages from {len(doc_index)} MD2D documents")
    return passages, doc_index


# ─── Layer 2: Telecom overlay documents ───────────────────────────────────────

TELECOM_OVERLAY_DOCS = [
    {
        "doc_id": "telecom_billing_001",
        "title": "Understanding Your Monthly Bill",
        "category": "billing",
        "domain": "telecom",
        "source": "telecom_overlay",
        "tags": ["billing", "charges", "invoice"],
        "sections": [
            {"section_id": "telecom_billing_001_s1", "heading": "Bill Components",
             "text": "Your monthly bill consists of three main components: your base plan charge, any add-on services, and usage-based charges. The base plan charge is fixed and billed on the same date each month. Add-on services are charged in full for the billing cycle in which they were activated."},
            {"section_id": "telecom_billing_001_s2", "heading": "Taxes and Regulatory Fees",
             "text": "All telecom services are subject to 18% GST as mandated by the Government of India. A Universal Service Obligation (USO) fund levy of 5% applies to all voice and data services. These charges are non-negotiable and cannot be waived."},
        ]
    },
    {
        "doc_id": "telecom_billing_002",
        "title": "How to Dispute a Charge",
        "category": "billing",
        "domain": "telecom",
        "source": "telecom_overlay",
        "tags": ["dispute", "wrong charge", "billing error", "refund"],
        "sections": [
            {"section_id": "telecom_billing_002_s1", "heading": "Eligibility for Dispute",
             "text": "A billing dispute can be raised for any charge that appears incorrect, unauthorized, or duplicated on your invoice. Disputes must be raised within 60 days of the bill generation date. Disputes raised after 60 days will not be accepted except in cases where the error was caused by our systems."},
            {"section_id": "telecom_billing_002_s2", "heading": "How to Raise a Dispute",
             "text": "To raise a billing dispute: (1) Use the self-service portal at myaccount.telecom.com under Billing > Dispute a Charge. (2) Call customer care at 198. (3) Visit the nearest store with your bill copy. All disputes are assigned a ticket ID and resolved within 7 working days."},
            {"section_id": "telecom_billing_002_s3", "heading": "Resolution and Refunds",
             "text": "If the dispute is resolved in your favor, the amount is adjusted in your next billing cycle. For prepaid customers, the refund is credited to the wallet within 3 working days. Cash refunds are not provided."},
        ]
    },
    {
        "doc_id": "telecom_billing_003",
        "title": "Late Payment and Suspension Policy",
        "category": "billing",
        "domain": "telecom",
        "source": "telecom_overlay",
        "tags": ["late payment", "suspension", "reconnection", "overdue"],
        "sections": [
            {"section_id": "telecom_billing_003_s1", "heading": "Due Date and Grace Period",
             "text": "Your bill is due within 21 days of the bill generation date. A grace period of 5 additional days is provided. No late payment fee is charged during the grace period. During the grace period all services remain active."},
            {"section_id": "telecom_billing_003_s2", "heading": "Service Suspension",
             "text": "If payment is not received within the grace period, outgoing services are suspended first. Incoming calls remain active for an additional 15 days. After 15 days of partial suspension without payment, all services are suspended. Account termination occurs after 90 days of zero payment."},
            {"section_id": "telecom_billing_003_s3", "heading": "Reconnection Process",
             "text": "To restore services after suspension, pay the outstanding amount plus a reconnection fee of Rs. 50 via the app, portal, or store. Services are restored within 4 hours of payment confirmation."},
        ]
    },
    {
        "doc_id": "telecom_plans_001",
        "title": "Prepaid and Postpaid Plan Details",
        "category": "plans",
        "domain": "telecom",
        "source": "telecom_overlay",
        "tags": ["prepaid", "postpaid", "plan", "recharge", "upgrade", "downgrade"],
        "sections": [
            {"section_id": "telecom_plans_001_s1", "heading": "Prepaid Plans Overview",
             "text": "Prepaid plans offer validity of 14 to 365 days. Entry plans (Rs. 99-199) include 1-2GB total data and unlimited voice. Mid-range plans (Rs. 249-449) include 1.5-2GB daily data with speed throttled to 64 Kbps after limit. Premium plans (Rs. 499-999) include 2-3GB daily data and OTT bundles."},
            {"section_id": "telecom_plans_001_s2", "heading": "Postpaid Plans Overview",
             "text": "Postpaid plans start at Rs. 399/month with unlimited voice, 40GB to unlimited data, and international SMS. Family plans allow 2 to 6 members with a shared data pool starting at Rs. 999 for 2 members."},
            {"section_id": "telecom_plans_001_s3", "heading": "Plan Change Policy",
             "text": "Postpaid plan downgrades take effect from the next billing cycle. Upgrading takes effect immediately and the difference is pro-rated. For prepaid, a new recharge applies after the current plan expires. Two active prepaid plans simultaneously are not permitted."},
        ]
    },
    {
        "doc_id": "telecom_network_001",
        "title": "Network Coverage and Outage Policy",
        "category": "network",
        "domain": "telecom",
        "source": "telecom_overlay",
        "tags": ["outage", "coverage", "signal", "4G", "5G", "compensation"],
        "sections": [
            {"section_id": "telecom_network_001_s1", "heading": "Coverage Types",
             "text": "We offer outdoor coverage (open areas), indoor coverage (urban and semi-urban), and deep indoor coverage (metro cities only). Coverage maps reflect outdoor coverage unless stated. 5G requires a 5G-capable device and is available in select cities."},
            {"section_id": "telecom_network_001_s2", "heading": "Reporting a Coverage Issue",
             "text": "Report persistent signal issues through the app under Network > Report Coverage Issue. Include your exact location and time of issue. Coverage complaints are reviewed within 5 working days and escalated to the infrastructure team if confirmed."},
            {"section_id": "telecom_network_001_s3", "heading": "Outage Compensation Policy",
             "text": "Postpaid customers are eligible for a service credit if a network outage lasts more than 4 continuous hours at their registered address. The credit is (monthly plan cost / 30) x disrupted days, applied automatically within 2 billing cycles. Prepaid customers receive a validity extension equivalent to the outage duration."},
        ]
    },
    {
        "doc_id": "telecom_network_002",
        "title": "Data Speed and Fair Usage Policy",
        "category": "network",
        "domain": "telecom",
        "source": "telecom_overlay",
        "tags": ["data speed", "FUP", "fair usage", "throttling", "slow internet", "data add-on"],
        "sections": [
            {"section_id": "telecom_network_002_s1", "heading": "Fair Usage Policy (FUP)",
             "text": "Once the daily or monthly high-speed data limit is exhausted, data speeds are reduced to 64 Kbps for the remainder of that day or cycle. Speed resets automatically at midnight IST for daily plans and on the bill cycle start date for monthly plans."},
            {"section_id": "telecom_network_002_s2", "heading": "Data Add-ons",
             "text": "Additional high-speed data can be purchased at any time through the app or portal under Manage > Buy Data. Add-on packs: 1GB for Rs. 19, 5GB for Rs. 49, 10GB for Rs. 79. Valid for 30 days or until exhausted. Add-on data does not roll over."},
        ]
    },
    {
        "doc_id": "telecom_roaming_001",
        "title": "International Roaming Policy and Packs",
        "category": "roaming",
        "domain": "telecom",
        "source": "telecom_overlay",
        "tags": ["roaming", "international", "abroad", "roaming charges", "roaming pack"],
        "sections": [
            {"section_id": "telecom_roaming_001_s1", "heading": "Activating Roaming",
             "text": "International roaming must be activated before travel. Go to My Account > Roaming > Activate or call 198. Prepaid customers need a minimum wallet balance of Rs. 500 and at least 7 days of remaining plan validity."},
            {"section_id": "telecom_roaming_001_s2", "heading": "Roaming Rates Without a Pack",
             "text": "Without a roaming pack, pay-per-use rates apply: outgoing calls Rs. 5-20/min, incoming Rs. 2-8/min, data Rs. 5-10/MB, SMS Rs. 5/message depending on destination. We strongly recommend purchasing a roaming pack before travel."},
            {"section_id": "telecom_roaming_001_s3", "heading": "Disputing Roaming Charges",
             "text": "Roaming charges can be disputed within 60 days of the bill date. Valid reasons include being charged roaming rates while in India (border area billing), incorrect pack activation, and charges continuing after return from travel. Disputed amounts are held until resolution."},
        ]
    },
    {
        "doc_id": "telecom_device_001",
        "title": "SIM Card and Device Management",
        "category": "device",
        "domain": "telecom",
        "source": "telecom_overlay",
        "tags": ["SIM", "SIM swap", "eSIM", "lost SIM", "IMEI", "device return"],
        "sections": [
            {"section_id": "telecom_device_001_s1", "heading": "Lost or Stolen SIM",
             "text": "If your SIM is lost or stolen, call 198 immediately (24/7) to block it. A replacement SIM retains your existing number, plan, and balance. Visit any store with a valid government ID. Fee: Rs. 100 for physical SIM, Rs. 150 for eSIM. Services transfer within 2 hours."},
            {"section_id": "telecom_device_001_s2", "heading": "eSIM Activation",
             "text": "eSIM is supported on iPhone XS and later, Samsung Galaxy S20 and later. Go to My Account > SIM Management > Switch to eSIM. A QR code is sent to your registered email. Scan it in device settings to activate. The physical SIM is deactivated automatically."},
            {"section_id": "telecom_device_001_s3", "heading": "Device Return Policy",
             "text": "Devices purchased through our stores can be returned within 10 days of purchase in original condition with all accessories. Defective devices within 30 days are eligible for like-for-like exchange. After 30 days, manufacturer warranty applies via authorized service centers."},
        ]
    },
    {
        "doc_id": "telecom_account_001",
        "title": "Account Security and Port-In/Port-Out",
        "category": "account",
        "domain": "telecom",
        "source": "telecom_overlay",
        "tags": ["security", "2FA", "OTP", "port", "MNP", "KYC", "account security"],
        "sections": [
            {"section_id": "telecom_account_001_s1", "heading": "Two-Factor Authentication",
             "text": "2FA is mandatory for all account changes including plan changes, payment method updates, and SIM swaps. OTPs are sent to your registered mobile and email, valid for 10 minutes. If you suspect unauthorized access, call 198 immediately to freeze your account."},
            {"section_id": "telecom_account_001_s2", "heading": "Port-Out Process (MNP)",
             "text": "To port your number to another operator, send PORT <your 10-digit number> to 1900 to get a UPC code (valid 4 days). Submit the UPC to your new operator. Porting completes within 2 working days. Services may be interrupted for up to 2 hours during porting."},
            {"section_id": "telecom_account_001_s3", "heading": "KYC Requirements",
             "text": "Accepted KYC documents: Aadhaar card, PAN card, Voter ID, Passport, Driving License. Aadhaar e-KYC is fastest and can be done digitally. Failure to complete re-KYC results in outgoing service suspension followed by full suspension after 30 days."},
        ]
    },
    {
        "doc_id": "telecom_escalation_001",
        "title": "Escalation Process and Customer Rights",
        "category": "escalation",
        "domain": "telecom",
        "source": "telecom_overlay",
        "tags": ["escalate", "TRAI", "complaint", "grievance", "rights"],
        "sections": [
            {"section_id": "telecom_escalation_001_s1", "heading": "Escalation Triggers",
             "text": "Escalate when: (1) A ticket has been open >7 working days without resolution. (2) Same issue recurred >3 times within 30 days. (3) Financial dispute exceeds Rs. 5,000. (4) Suspected fraud or unauthorized account activity. (5) Service suspended in error."},
            {"section_id": "telecom_escalation_001_s2", "heading": "Escalation Levels",
             "text": "Level 1: Customer Care (198) — 24 hours target. Level 2: Nodal Officer (nodal@telecom.com) — 3 working days target. Level 3: Appellate Authority — 39 days per TRAI mandate. Level 4: TRAI Consumer Complaint Portal (consumercomplaints.trai.gov.in) — regulatory escalation."},
            {"section_id": "telecom_escalation_001_s3", "heading": "Customer Rights under TRAI",
             "text": "Under TRAI regulations you have the right to: a clear itemized bill; compensation for unplanned outages exceeding defined thresholds; mobile number portability without unreasonable delay; a grievance redressal mechanism with defined timelines; and information about network coverage before purchase."},
        ]
    },
]


def process_telecom_overlay(docs: list) -> List[Dict]:
    """Converts telecom overlay doc list into passage records (same schema as MD2D)."""
    passages = []
    for doc in docs:
        for sec in doc["sections"]:
            text = sec["text"].strip()
            passage = {
                "passage_id": hashlib.md5(sec["section_id"].encode()).hexdigest()[:12],
                "doc_id":     doc["doc_id"],
                "section_id": sec["section_id"],
                "span_id":    None,               # no MD2D span — telecom overlay
                "title":      doc["title"],
                "heading":    sec["heading"],
                "category":   doc["category"],
                "domain":     "telecom",
                "source":     "telecom_overlay",
                "tags":       doc["tags"],
                "text":       text,
                "full_text":  f"{sec['heading']}. {text}",
                "char_start": 0,
                "char_end":   len(text),
                "word_count": len(text.split()),
            }
            passages.append(passage)
    return passages


# ─── Statistics printer ────────────────────────────────────────────────────────

def print_kb_stats(passages: List[Dict]):
    from collections import Counter
    sources  = Counter(p["source"]   for p in passages)
    domains  = Counter(p["domain"]   for p in passages)
    cats     = Counter(p["category"] for p in passages)

    print("\n" + "=" * 60)
    print("  KNOWLEDGE BASE STATISTICS")
    print("=" * 60)
    print(f"  Total passages   : {len(passages):,}")
    print(f"  Total words      : {sum(p['word_count'] for p in passages):,}")

    print(f"\n  By source:")
    for src, count in sources.most_common():
        print(f"    {src:<25} {count:>6,} passages")

    print(f"\n  By domain:")
    for dom, count in domains.most_common():
        print(f"    {dom:<25} {count:>6,} passages")

    print(f"\n  By category (telecom overlay):")
    tc = {p["category"]: 0 for p in passages if p["source"] == "telecom_overlay"}
    for p in passages:
        if p["source"] == "telecom_overlay":
            tc[p["category"]] = tc.get(p["category"], 0) + 1
    for cat, count in sorted(tc.items()):
        print(f"    {cat:<25} {count:>6,} passages")
    print("=" * 60)


# ─── Main build function ───────────────────────────────────────────────────────

def build_kb(
    md2d_docs_path: str = "data/raw/multidoc2dial/documents",
    output_dir:     str = "data/processed",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Layer 1: Real MultiDoc2Dial documents ──────────────────────
    md2d_passages, doc_index = load_multidoc2dial_documents(md2d_docs_path)

    # ── Layer 2: Telecom overlay ───────────────────────────────────
    print("  Processing telecom overlay documents...")
    telecom_passages = process_telecom_overlay(TELECOM_OVERLAY_DOCS)
    print(f"  Telecom overlay: {len(telecom_passages)} passages from "
          f"{len(TELECOM_OVERLAY_DOCS)} documents")

    # Add telecom overlay docs to doc_index
    for doc in TELECOM_OVERLAY_DOCS:
        doc_index[doc["doc_id"]] = {
            "doc_id":   doc["doc_id"],
            "title":    doc["title"],
            "domain":   "telecom",
            "category": doc["category"],
            "source":   "telecom_overlay",
        }

    # ── Combine ────────────────────────────────────────────────────
    all_passages = md2d_passages + telecom_passages
    print_kb_stats(all_passages)

    # ── Save ───────────────────────────────────────────────────────
    # passages.jsonl — one passage per line, ready for FAISS
    kb_path = Path(output_dir) / "kb_passages.jsonl"
    with open(kb_path, "w", encoding="utf-8") as f:
        for p in all_passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n  Saved {len(all_passages):,} passages -> {kb_path}")

    # doc_index.json — for fast lookup at inference
    doc_idx_path = Path(output_dir) / "doc_index.json"
    with open(doc_idx_path, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(doc_index)} doc records -> {doc_idx_path}")

    # span_index.json — section_id → passage record (for GetPolicy tool)
    span_index = {p["section_id"]: p for p in all_passages}
    span_idx_path = Path(output_dir) / "span_index.json"
    with open(span_idx_path, "w", encoding="utf-8") as f:
        json.dump(span_index, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(span_index)} span records -> {span_idx_path}")

    # dataset_stats.json — for your report
    stats = {
        "total_passages":         len(all_passages),
        "md2d_passages":          len(md2d_passages),
        "telecom_overlay_passages": len(telecom_passages),
        "total_documents":        len(doc_index),
        "md2d_documents":         len(doc_index) - len(TELECOM_OVERLAY_DOCS),
        "telecom_overlay_docs":   len(TELECOM_OVERLAY_DOCS),
        "total_words":            sum(p["word_count"] for p in all_passages),
    }
    stats_path = Path(output_dir) / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats -> {stats_path}")

    return all_passages, doc_index


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    build_kb()

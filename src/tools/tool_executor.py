"""
src/tools/tool_executor.py

Real tool execution logic — wired to the actual KB and span index.
Replaces the mock executors from the Week 1 schema definition.

Tools:
  SearchKB(query, category_filter, top_k)
    → calls DenseRetriever (Week 2) + Reranker (Week 2)
    → returns top passages with doc_id, section_id, score

  GetPolicy(section_id, doc_id)
    → direct lookup in span_index.json
    → returns full passage text for authoritative citation

  CreateTicket(summary, category, severity, customer_mobile)
    → writes to data/processed/tickets.jsonl (persistent local store)
    → returns ticket_id, eta_hours, queue

  CheckNetworkStatus(region, service_type)   ← NOVEL TOOL
    → reads from data/raw/network_status.json (mock live feed)
    → returns status, active_incident, compensation_eligible

Escalation logic (used by full pipeline):
  should_escalate(query, retrieved, confidence) → bool

Run standalone demo:
    python -m src.tools.tool_executor
"""

import json
import uuid
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

random.seed(int(time.time()))   # vary each run for realistic outage simulation


# ─── Tool schemas (JSON, for prompt injection) ────────────────────────────────

TOOL_SCHEMAS = {
    "SearchKB": {
        "name": "SearchKB",
        "description": (
            "Searches the knowledge base for passages relevant to the customer query. "
            "Always call this first. Returns passages with doc_id and section_id for citation."
        ),
        "parameters": {
            "query":           {"type": "str",  "required": True},
            "category_filter": {"type": "str",  "required": False, "default": "any",
                                "enum": ["billing","plans","network","device",
                                         "account","roaming","escalation","any"]},
            "top_k":           {"type": "int",  "required": False, "default": 5},
        },
    },
    "GetPolicy": {
        "name": "GetPolicy",
        "description": (
            "Fetches the full authoritative policy text for a known section. "
            "Use when SearchKB returns a relevant doc_id and you need the complete text."
        ),
        "parameters": {
            "section_id": {"type": "str", "required": True},
            "doc_id":     {"type": "str", "required": False},
        },
    },
    "CreateTicket": {
        "name": "CreateTicket",
        "description": (
            "Creates an escalation ticket when: KB has no sufficient answer, "
            "financial dispute > Rs. 5000, suspected fraud, or issue unresolved > 7 days. "
            "Returns ticket_id for tracking."
        ),
        "parameters": {
            "summary":          {"type": "str",  "required": True},
            "category":         {"type": "str",  "required": True,
                                 "enum": ["billing","network","device","account",
                                          "roaming","fraud","general"]},
            "severity":         {"type": "str",  "required": True,
                                 "enum": ["low","medium","high","critical"]},
            "customer_mobile":  {"type": "str",  "required": False},
            "preferred_contact":{"type": "str",  "required": False, "default": "call"},
        },
    },
    "CheckNetworkStatus": {
        "name": "CheckNetworkStatus",
        "description": (
            "NOVEL TOOL — Checks real-time network status for a region and service type. "
            "Call this for ALL network/signal/speed complaints BEFORE suggesting device fixes. "
            "If an outage exists, cite it in the answer instead of troubleshooting steps."
        ),
        "parameters": {
            "region":       {"type": "str", "required": True},
            "service_type": {"type": "str", "required": False, "default": "all",
                             "enum": ["4G","5G","voice","sms","all"]},
        },
    },
}


def get_tool_descriptions_for_prompt() -> str:
    """Returns compact tool descriptions for injection into the generator prompt."""
    lines = ["AVAILABLE TOOLS (emit as JSON: {\"tool\": name, \"params\": {...}}):\n"]
    for name, schema in TOOL_SCHEMAS.items():
        req = [k for k, v in schema["parameters"].items() if v.get("required")]
        lines.append(f"  {name}: {schema['description'][:100]}")
        lines.append(f"    required: {req}")
    return "\n".join(lines)


# ─── Tool 1: SearchKB ─────────────────────────────────────────────────────────

class SearchKBTool:
    """
    Real SearchKB implementation.
    Uses DenseRetriever (Week 2 fine-tuned model) + Reranker.
    Falls back to BM25 if checkpoints are not available.
    """

    def __init__(
        self,
        retriever_path: str = "checkpoints/retriever",
        reranker_path:  str = "checkpoints/reranker",
        index_dir:      str = "data/index",
        kb_path:        str = "data/processed/kb_passages.jsonl",
    ):
        self._retriever = None
        self._reranker  = None
        self._bm25      = None

        # Try to load dense retriever
        try:
            from src.retrieval.faiss_indexer import DenseRetriever, HybridRetriever
            dense = DenseRetriever(retriever_path, index_dir, label="finetuned")
            self._retriever = HybridRetriever(dense)
            print("  [SearchKB] Hybrid retriever (Dense + BM25) loaded.")
        except Exception as e:
            print(f"  [SearchKB] Dense retriever unavailable ({e}). Using BM25 fallback.")
            import sys
            sys.path.insert(0, ".")
            from src.baseline.baseline_system import BM25
            with open(kb_path) as f:
                passages = [json.loads(line) for line in f]
            self._bm25 = BM25(passages)

        # Try to load reranker
        try:
            from src.retrieval.reranker import Reranker
            self._reranker = Reranker(reranker_path)
            print("  [SearchKB] Reranker loaded.")
        except Exception as e:
            print(f"  [SearchKB] Reranker unavailable ({e}). Skipping reranking.")

    def execute(self, params: Dict) -> Dict:
        query           = params.get("query", "").strip()
        category_filter = params.get("category_filter", "any")
        force_telecom   = params.get("force_telecom", False)
        top_k           = int(params.get("top_k", 5))

        if not query:
            return {"passages": [], "total_found": 0, "error": "Empty query"}

        t0 = time.time()

        # Step 1: retrieve
        if self._retriever:
            candidates = self._retriever.search(
                query, top_k=top_k * 4,
                category_filter=category_filter
            )
        else:
            raw = self._bm25.search(query, top_k=top_k * 4)
            # Normalise BM25 field names to dense schema
            for r in raw:
                r["dense_score"] = r.pop("retrieval_score", 0.0)
            if category_filter != "any":
                raw = [r for r in raw if r.get("category") == category_filter]
            candidates = raw

        # Step 1b: Domain Guard - if force_telecom, throw away everything else
        if force_telecom:
            telecom_candidates = [c for c in candidates if c.get("source") == "telecom_overlay"]
            if telecom_candidates:
                candidates = telecom_candidates

        # Step 2: rerank if available
        if self._reranker and candidates:
            candidates = self._reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        latency_ms = round((time.time() - t0) * 1000, 1)

        return {
            "passages":    candidates,
            "total_found": len(candidates),
            "latency_ms":  latency_ms,
            "_tool":       "SearchKB",
            "_query":      query,
        }


# ─── Tool 2: GetPolicy ───────────────────────────────────────────────────────

class GetPolicyTool:
    """
    Direct lookup in span_index.json.
    Returns the full authoritative passage for citation.
    """

    def __init__(self, span_index_path: str = "data/processed/span_index.json"):
        self._span_index = {}
        if Path(span_index_path).exists():
            with open(span_index_path) as f:
                self._span_index = json.load(f)
            print(f"  [GetPolicy] Loaded {len(self._span_index):,} sections.")
        else:
            print(f"  [GetPolicy] span_index not found: {span_index_path}")

    def execute(self, params: Dict) -> Dict:
        section_id = params.get("section_id", "").strip()
        doc_id     = params.get("doc_id", "")

        if not section_id:
            return {"error": "section_id is required", "_tool": "GetPolicy"}

        passage = self._span_index.get(section_id)

        if not passage:
            # Try doc_id prefix match as fallback
            if doc_id:
                matches = [
                    v for k, v in self._span_index.items()
                    if k.startswith(doc_id)
                ]
                if matches:
                    passage = matches[0]

        if not passage:
            return {
                "error":      f"Section '{section_id}' not found in span index.",
                "section_id": section_id,
                "_tool":      "GetPolicy",
            }

        return {
            "section_id":  section_id,
            "doc_id":      passage.get("doc_id", doc_id),
            "title":       passage.get("title", ""),
            "heading":     passage.get("heading", ""),
            "policy_text": passage.get("text", ""),
            "category":    passage.get("category", ""),
            "domain":      passage.get("domain", ""),
            "source":      passage.get("source", ""),
            "last_updated": "2025-01-15",
            "_tool":       "GetPolicy",
        }


# ─── Tool 3: CreateTicket ────────────────────────────────────────────────────

class CreateTicketTool:
    """
    Creates a support escalation ticket.
    Writes to data/processed/tickets.jsonl for persistence.
    """

    SEVERITY_ETA  = {"low": 48, "medium": 24, "high": 8, "critical": 2}
    CATEGORY_QUEUE = {
        "billing":  "billing-disputes",
        "network":  "network-ops",
        "device":   "device-support",
        "account":  "account-security",
        "roaming":  "roaming-support",
        "fraud":    "fraud-response",
        "general":  "general-support",
    }

    def __init__(self, tickets_path: str = "data/processed/tickets.jsonl"):
        self._tickets_path = tickets_path
        Path(tickets_path).parent.mkdir(parents=True, exist_ok=True)

    def execute(self, params: Dict) -> Dict:
        summary     = params.get("summary", "").strip()
        category    = params.get("category", "general")
        severity    = params.get("severity", "medium")
        mobile      = params.get("customer_mobile", "")
        contact     = params.get("preferred_contact", "call")

        if not summary:
            return {"error": "summary is required", "_tool": "CreateTicket"}

        ticket_id  = f"TKT-{str(uuid.uuid4())[:8].upper()}"
        created_at = datetime.utcnow().isoformat() + "Z"
        eta_hours  = self.SEVERITY_ETA.get(severity, 24)
        queue      = self.CATEGORY_QUEUE.get(category, "general-support")

        ticket = {
            "ticket_id":      ticket_id,
            "summary":        summary,
            "category":       category,
            "severity":       severity,
            "customer_mobile": mobile,
            "preferred_contact": contact,
            "status":         "open",
            "queue":          queue,
            "eta_hours":      eta_hours,
            "created_at":     created_at,
            "reference_url":  f"https://support.telecom.com/tickets/{ticket_id}",
        }

        # Persist ticket
        with open(self._tickets_path, "a") as f:
            f.write(json.dumps(ticket) + "\n")

        return {
            "ticket_id":     ticket_id,
            "status":        "created",
            "queue":         queue,
            "eta_hours":     eta_hours,
            "reference_url": ticket["reference_url"],
            "message":       (
                f"Ticket {ticket_id} created. Our {queue} team will contact "
                f"you within {eta_hours} hours via {contact}."
            ),
            "_tool":         "CreateTicket",
        }


# ─── Tool 4: CheckNetworkStatus (NOVEL) ──────────────────────────────────────

class CheckNetworkStatusTool:
    """
    NOVEL TOOL — Checks network status for a region.

    In production: calls a real network operations API.
    Here: reads from data/raw/network_status.json (pre-seeded mock feed)
    with a 20% random incident rate to simulate live variation.

    Key insight: calling this BEFORE device troubleshooting prevents the
    system from giving wrong advice ("restart your phone") when there's
    actually an active tower outage.
    """

    STATUS_LEVELS = ["operational", "degraded", "partial_outage",
                     "major_outage", "planned_maintenance"]

    def __init__(self, status_feed_path: str = "data/raw/network_status.json"):
        self._feed = {}
        if Path(status_feed_path).exists():
            with open(status_feed_path) as f:
                self._feed = json.load(f)

    def execute(self, params: Dict) -> Dict:
        region       = params.get("region", "Unknown").strip()
        service_type = params.get("service_type", "all")
        now          = datetime.utcnow()

        # Check pre-seeded feed first
        feed_key = region.lower().replace(" ", "_")
        if feed_key in self._feed:
            record = self._feed[feed_key]
        else:
            # Simulate: 20% chance of active incident for unseeded regions
            has_incident = random.random() < 0.20
            if has_incident:
                status     = random.choice(["degraded", "partial_outage", "major_outage"])
                started    = (now - timedelta(hours=random.randint(1, 5))).isoformat() + "Z"
                eta_hours  = random.randint(1, 6)
                est_res    = (now + timedelta(hours=eta_hours)).isoformat() + "Z"
                affected   = (
                    ["4G data"] if status == "degraded" else
                    ["4G data", "voice"] if status == "partial_outage" else
                    ["4G data", "5G data", "voice", "SMS"]
                )
                record = {
                    "status":               status,
                    "active_incident":      True,
                    "incident_id":          f"INC-{random.randint(1000,9999)}",
                    "incident_summary":     f"Unplanned {status.replace('_',' ')} in {region}.",
                    "started_at":           started,
                    "estimated_resolution": est_res,
                    "affected_services":    affected,
                    "compensation_eligible": eta_hours >= 4,
                }
            else:
                record = {
                    "status":               "operational",
                    "active_incident":      False,
                    "incident_id":          None,
                    "incident_summary":     None,
                    "started_at":           None,
                    "estimated_resolution": None,
                    "affected_services":    [],
                    "compensation_eligible": False,
                }

        return {
            "region":               region,
            "service_type":         service_type,
            "last_updated":         now.isoformat() + "Z",
            **record,
            "_tool":                "CheckNetworkStatus",
        }


# ─── Tool registry & dispatcher ──────────────────────────────────────────────

class ToolExecutor:
    """
    Central dispatcher. Instantiated once at startup, shared across requests.
    """

    def __init__(
        self,
        retriever_path:  str = "checkpoints/retriever",
        reranker_path:   str = "checkpoints/reranker",
        index_dir:       str = "data/index",
        kb_path:         str = "data/processed/kb_passages.jsonl",
        span_index_path: str = "data/processed/span_index.json",
        tickets_path:    str = "data/processed/tickets.jsonl",
        status_feed_path:str = "data/raw/network_status.json",
    ):
        print("\n  [ToolExecutor] Initialising tools...")
        self.tools = {
            "SearchKB":           SearchKBTool(retriever_path, reranker_path,
                                               index_dir, kb_path),
            "GetPolicy":          GetPolicyTool(span_index_path),
            "CreateTicket":       CreateTicketTool(tickets_path),
            "CheckNetworkStatus": CheckNetworkStatusTool(status_feed_path),
        }
        print("  [ToolExecutor] All tools ready.\n")

    def execute(self, tool_name: str, params: Dict) -> Dict:
        """
        Executes a tool by name. Raises ValueError for unknown tools.
        Always adds _tool key for trace logging.
        """
        if tool_name not in self.tools:
            raise ValueError(
                f"Unknown tool '{tool_name}'. "
                f"Available: {list(self.tools.keys())}"
            )
        result          = self.tools[tool_name].execute(params)
        result["_tool"] = tool_name
        return result

    def get_schema_prompt(self) -> str:
        return get_tool_descriptions_for_prompt()


# ─── Escalation logic ────────────────────────────────────────────────────────

def should_escalate(
    query:       str,
    retrieved:   List[Dict],
    confidence:  Optional[float] = None,
    history:     List[Dict]      = None,
) -> tuple[bool, str]:
    """
    Decides whether to escalate to a human agent.

    Rules (in priority order):
      1. No relevant passages found (confidence too low or empty retrieval)
      2. Query mentions fraud / hacked / unauthorized
      3. High-value financial dispute (Rs. > 5000 mentioned)
      4. Recurring issue pattern in history (3+ turns about same topic)
      5. Explicit escalation request ("speak to human", "supervisor")

    Returns:
      (should_escalate: bool, reason: str)
    """
    import re
    query_lower = query.lower()

    # Rule 1: No relevant KB evidence
    if not retrieved:
        return True, "no_kb_evidence"
    top_score = retrieved[0].get("dense_score",
                retrieved[0].get("rerank_score",
                retrieved[0].get("retrieval_score", 0.0)))
    if confidence is not None and confidence < 0.30:
        return True, "low_confidence"
    if top_score < 0.20 and confidence is None:
        return True, "low_retrieval_score"

    # Rule 2: Fraud / security keywords
    fraud_kw = ["fraud", "hack", "hacked", "unauthorized", "stolen", "not me",
                "someone else", "suspicious", "security breach"]
    if any(kw in query_lower for kw in fraud_kw):
        return True, "security_concern"

    # Rule 3: High-value dispute
    amounts = re.findall(r"rs\.?\s*(\d[\d,]+)", query_lower)
    for amt_str in amounts:
        amt = int(amt_str.replace(",", ""))
        if amt > 5000:
            return True, f"high_value_dispute_rs_{amt}"

    # Rule 4: Recurring issue in history
    if history and len(history) >= 6:
        recent_topics = [t.get("utterance", "") for t in history[-6:]]
        if len(set(recent_topics)) < 3:   # very repetitive conversation
            return True, "recurring_issue"

    # Rule 5: Explicit escalation request
    escalation_kw = ["speak to human", "human agent", "supervisor",
                     "manager", "escalate", "nodal", "trai complaint"]
    if any(kw in query_lower for kw in escalation_kw):
        return True, "explicit_escalation_request"

    return False, "kb_sufficient"


# ─── Seed network status feed ────────────────────────────────────────────────

def seed_network_status_feed(output_path: str = "data/raw/network_status.json"):
    """
    Seeds a mock network status feed with pre-defined incidents.
    In a real system this is replaced by a live API call.
    """
    feed = {
        "mumbai": {
            "status": "partial_outage",
            "active_incident": True,
            "incident_id": "INC-4821",
            "incident_summary": "Partial 4G data outage in Mumbai due to fiber cut.",
            "started_at": "2025-07-15T08:00:00Z",
            "estimated_resolution": "2025-07-15T16:00:00Z",
            "affected_services": ["4G data"],
            "compensation_eligible": True,
        },
        "delhi": {
            "status": "operational",
            "active_incident": False,
            "incident_id": None,
            "incident_summary": None,
            "started_at": None,
            "estimated_resolution": None,
            "affected_services": [],
            "compensation_eligible": False,
        },
        "bangalore": {
            "status": "planned_maintenance",
            "active_incident": True,
            "incident_id": "MAINT-112",
            "incident_summary": "Planned maintenance in Bangalore. 4G unavailable 2–5 AM.",
            "started_at": "2025-07-15T20:30:00Z",
            "estimated_resolution": "2025-07-15T23:00:00Z",
            "affected_services": ["4G data", "5G data"],
            "compensation_eligible": False,
        },
        "ahmedabad": {
            "status": "operational",
            "active_incident": False,
            "incident_id": None,
            "incident_summary": None,
            "started_at": None,
            "estimated_resolution": None,
            "affected_services": [],
            "compensation_eligible": False,
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(feed, f, indent=2)
    print(f"  Network status feed seeded -> {output_path}")
    return feed


# ─── CLI demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # Seed network feed
    seed_network_status_feed()

    # Initialise executor
    executor = ToolExecutor()

    # Demo calls
    demos = [
        ("SearchKB",           {"query": "how to dispute a roaming charge", "top_k": 3}),
        ("GetPolicy",          {"section_id": "telecom_billing_002_s2"}),
        ("CheckNetworkStatus", {"region": "Mumbai", "service_type": "4G"}),
        ("CheckNetworkStatus", {"region": "Ahmedabad", "service_type": "4G"}),
        ("CreateTicket",       {"summary": "Customer charged Rs. 12000 for roaming unfairly",
                                "category": "billing", "severity": "high",
                                "customer_mobile": "9876543210"}),
    ]

    print("\n" + "="*55)
    print("  TOOL EXECUTOR DEMO")
    print("="*55)

    for tool_name, params in demos:
        print(f"\n── {tool_name} ──")
        print(f"  Input : {params}")
        result = executor.execute(tool_name, params)
        # Print concise output
        if tool_name == "SearchKB":
            for p in result.get("passages", [])[:2]:
                score_key = "rerank_score" if "rerank_score" in p else "dense_score"
                print(f"  [{p.get(score_key,'?'):.3f}] {p['doc_id']} — {p['heading']}")
        elif tool_name == "GetPolicy":
            print(f"  heading : {result.get('heading')}")
            print(f"  text    : {result.get('policy_text','')[:100]}...")
        elif tool_name == "CheckNetworkStatus":
            print(f"  status  : {result['status']}")
            print(f"  incident: {result['active_incident']} — {result.get('incident_summary','')}")
        elif tool_name == "CreateTicket":
            print(f"  ticket_id : {result['ticket_id']}")
            print(f"  eta_hours : {result['eta_hours']}")
            print(f"  message   : {result['message']}")

    # Demo escalation logic
    print("\n── Escalation logic demo ──")
    cases = [
        ("How do I dispute a charge?",              [],              None,  []),
        ("I think my account was hacked",           [{"dense_score": 0.8}], None, []),
        ("I was billed Rs. 12000 for roaming",      [{"dense_score": 0.7}], None, []),
        ("No signal for 3 days (4th complaint)",    [{"dense_score": 0.6}], None,
         [{"role":"user","utterance":"no signal"} for _ in range(7)]),
    ]
    for query, retrieved, conf, history in cases:
        esc, reason = should_escalate(query, retrieved, conf, history)
        print(f"  {'ESCALATE' if esc else 'NO ESC  '} [{reason}] {query[:50]}")

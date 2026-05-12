# src/policy/tool_policy_classifier.py

"""
Trained tool-policy classifier (Component E).
Upgrades the rule-based tool_policy() in inference_pipeline.py.

Model: fine-tuned bert-base-uncased
Input: user query text
Output: primary tool to call (SearchKB / GetPolicy / CreateTicket / CheckNetworkStatus)

Training data: ~400 labeled examples generated from KB queries
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

LABEL2ID = {
    "SearchKB": 0,
    "GetPolicy": 1,
    "CreateTicket": 2,
    "CheckNetworkStatus": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ─── Training data generation ────────────────────────────────────────────────

def generate_training_data() -> List[Dict]:
    """
    Creates ~400 labeled (query, tool) examples covering all 4 tools.
    Rule: primary tool = the first tool that should be called for this query.
    """
    examples = []

    # SearchKB — general support queries (label 0)
    searchkb_queries = [
        "How do I dispute a charge on my bill?",
        "What are the prepaid plans available?",
        "How do I port my number to another operator?",
        "Can I upgrade my postpaid plan mid-month?",
        "What documents do I need for a new SIM?",
        "How do I set up autopay for my account?",
        "What is the data rollover policy?",
        "How do I activate international roaming?",
        "What is the return policy for devices?",
        "How do I unlock my device IMEI?",
        "How do I check my remaining data balance?",
        "What are the roaming rates for USA?",
        "How do I change my plan?",
        "Where can I find my bill online?",
        "What is the late payment fee?",
        "How do I add a family member to my plan?",
        "What is the validity of my prepaid recharge?",
        "How do I deactivate my SIM temporarily?",
        "What happens if I exceed my data limit?",
        "How do I get a duplicate SIM?",
        "Can I keep my number when switching plans?",
        "What is the minimum recharge amount?",
        "How do I check if 5G is available in my area?",
        "What is the fair usage policy?",
        "How do I cancel my subscription?",
        "What are the HD calling charges?",
        "How do I get a paper bill?",
        "How do I update my address in account?",
        "What is the SIM swap process?",
        "How long does porting take?",
    ] * 3  # repeat for ~90 examples

    # GetPolicy — specific regulation/policy section queries (label 1)
    getpolicy_queries = [
        "What exactly does section 4.2 say about roaming?",
        "What is the TRAI regulation on billing disputes?",
        "Show me the full terms for data rollover policy.",
        "What are the exact terms of the fair usage policy?",
        "What does the subscriber agreement say about cancellation?",
        "Quote the exact policy on SIM replacement charges.",
        "What is the official policy for compensation during outages?",
        "What does the contract say about early termination fees?",
        "Show me the exact regulation on number porting timelines.",
        "What is the official policy on unauthorized charges?",
        "What does the TRAI order say about service quality?",
        "Show me the warranty terms for devices.",
        "What exactly are the terms for international roaming charges?",
        "Quote the policy on prepaid validity extension.",
        "What does the agreement say about data throttling?",
        "Show me the terms and conditions for cashback offers.",
        "What is the exact refund policy for cancelled orders?",
        "What does the contract say about auto-renewal?",
        "Show me the privacy policy for account data.",
        "What are the exact eligibility criteria for a plan upgrade?",
    ] * 5  # ~100 examples

    # CreateTicket — escalation/complaint queries (label 2)
    createticket_queries = [
        "I want to file a formal complaint about my billing.",
        "This issue has not been resolved for 15 days.",
        "I think someone hacked my account and made calls.",
        "I was charged Rs. 8000 for roaming I never used.",
        "I need to speak to a supervisor right now.",
        "Please escalate my issue to the nodal officer.",
        "I want to register a complaint with TRAI.",
        "My account was compromised and unauthorized recharges were made.",
        "I have been calling for 2 weeks and nothing is fixed.",
        "I need a formal escalation for this billing dispute.",
        "This is the 5th time I am calling about the same problem.",
        "I want compensation for 3 days of no service.",
        "Someone made unauthorized changes to my plan.",
        "I need a written acknowledgement of my complaint.",
        "Please create a ticket for my unresolved issue.",
        "I want to escalate to the grievance redressal cell.",
        "I was charged Rs. 12000 for international calls I didn't make.",
        "Please connect me to the fraud investigation team.",
        "My SIM was swapped without my permission.",
        "I need to report a fraudulent recharge on my account.",
        "Please register my complaint for network issues in writing.",
        "I want this escalated — I have been waiting too long.",
        "This is a serious security issue with my account.",
    ] * 4  # ~92 examples

    # CheckNetworkStatus — network/signal queries (label 3)
    checknetwork_queries = [
        "My 4G is not working since this morning.",
        "There is no signal in my area.",
        "Is there a network outage in Mumbai right now?",
        "My internet speed is very slow today.",
        "I cannot make calls — is there a tower issue?",
        "Network is down in Bangalore.",
        "Is there maintenance going on in Delhi?",
        "I have been getting no service for 2 hours.",
        "Why is my 5G not connecting in Ahmedabad?",
        "The network has been unstable all day.",
        "Is there an outage affecting voice calls?",
        "My SMS is not getting delivered — network issue?",
        "Data is not working on 4G in Pune.",
        "Network dropped suddenly in Hyderabad.",
        "Is there any planned maintenance tonight?",
        "Signal is weak at my location in Chennai.",
        "Why is my data speed reduced to 2G?",
        "No network since last night in Kolkata.",
        "Is there a tower outage near me?",
        "Internet keeps disconnecting every few minutes.",
    ] * 5  # ~100 examples

    for q in searchkb_queries:
        examples.append({"query": q, "label": "SearchKB"})
    for q in getpolicy_queries:
        examples.append({"query": q, "label": "GetPolicy"})
    for q in createticket_queries:
        examples.append({"query": q, "label": "CreateTicket"})
    for q in checknetwork_queries:
        examples.append({"query": q, "label": "CheckNetworkStatus"})

    return examples


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ToolPolicyDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 128):
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex  = self.examples[idx]
        enc = self.tokenizer(
            ex["query"],
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(LABEL2ID[ex["label"]], dtype=torch.long),
        }


# ─── Training ─────────────────────────────────────────────────────────────────

def train_tool_policy(
    output_dir:  str   = "checkpoints/tool_policy",
    epochs:      int   = 5,
    batch_size:  int   = 16,
    lr:          float = 2e-5,
):
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from sklearn.model_selection import train_test_split
    import evaluate

    print("\n" + "="*50)
    print("  TOOL POLICY CLASSIFIER TRAINING")
    print("="*50)

    # Generate data
    examples = generate_training_data()
    print(f"  Total examples: {len(examples)}")

    # Split
    train_ex, val_ex = train_test_split(examples, test_size=0.15, random_state=42,
                                         stratify=[e["label"] for e in examples])
    print(f"  Train: {len(train_ex)} | Val: {len(val_ex)}")

    # Tokenizer + model
    model_name = "google/bert_uncased_L-4_H-256_A-4"  # tiny BERT — fast on CPU
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels = len(LABEL2ID),
        id2label   = ID2LABEL,
        label2id   = LABEL2ID,
    )

    train_ds = ToolPolicyDataset(train_ex, tokenizer)
    val_ds   = ToolPolicyDataset(val_ex,   tokenizer)

    # Metrics
    accuracy_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=preds, references=labels)

    # Training args
    args = TrainingArguments(
        output_dir              = output_dir,
        num_train_epochs        = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        learning_rate           = lr,
        weight_decay            = 0.01,
        eval_strategy           = "epoch",
        save_strategy           = "best",
        load_best_model_at_end  = True,
        metric_for_best_model   = "accuracy",
        logging_steps           = 20,
        report_to               = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
    )

    print("\n  Training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Final eval
    results = trainer.evaluate()
    print(f"\n  Final Val Accuracy: {results['eval_accuracy']:.4f}")
    print(f"  Checkpoint saved → {output_dir}")

    # Save label map
    with open(f"{output_dir}/label_map.json", "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)

    return results


# ─── Inference class (replaces rule-based tool_policy) ───────────────────────

class TrainedToolPolicy:
    """
    Drop-in replacement for the rule_based tool_policy() function.
    Loads fine-tuned BERT and classifies the primary tool to call.
    Falls back to rule-based if checkpoint not available.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints/tool_policy"):
        self._model     = None
        self._tokenizer = None
        self._available = False

        if Path(checkpoint_dir).exists():
            try:
                from transformers import (
                    AutoTokenizer,
                    AutoModelForSequenceClassification,
                )
                self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
                self._model     = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint_dir
                )
                self._model.eval()
                self._available = True
                print(f"  [ToolPolicy] Trained classifier loaded from {checkpoint_dir}")
            except Exception as e:
                print(f"  [ToolPolicy] Could not load classifier ({e}). Using rule-based.")
        else:
            print(f"  [ToolPolicy] No checkpoint at {checkpoint_dir}. Using rule-based.")

    def predict(self, query: str) -> Tuple[str, float]:
        """Returns (tool_name, confidence)."""
        if not self._available:
            return self._rule_based_predict(query)

        inputs = self._tokenizer(
            query,
            return_tensors = "pt",
            max_length     = 128,
            truncation     = True,
            padding        = True,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs      = torch.softmax(logits, dim=-1).squeeze()
        top_id     = int(probs.argmax())
        confidence = float(probs[top_id])
        return ID2LABEL[top_id], confidence

    def get_tool_calls(
        self,
        query:   str,
        history: List[Dict],
    ) -> List[Tuple[str, Dict]]:
        """
        Same interface as the rule-based tool_policy().
        Primary tool from classifier; secondary tools from rules.
        """
        primary_tool, confidence = self.predict(query)
        q = query.lower()

        calls = []

        # Primary tool from classifier
        if primary_tool == "SearchKB":
            category = self._infer_category(q)
            calls.append(("SearchKB", {"query": query, "category_filter": category, "top_k": 5}))
        elif primary_tool == "GetPolicy":
            category = self._infer_category(q)
            calls.append(("SearchKB", {"query": query, "category_filter": category, "top_k": 5}))
            calls.append(("GetPolicy", {"_deferred": True}))
        elif primary_tool == "CheckNetworkStatus":
            calls.append(("SearchKB", {"query": query, "category_filter": "network", "top_k": 5}))
            region = self._extract_region(query)
            svc    = "5G" if "5g" in q else "4G" if "4g" in q else "all"
            calls.append(("CheckNetworkStatus", {"region": region, "service_type": svc}))
        elif primary_tool == "CreateTicket":
            calls.append(("SearchKB", {"query": query, "category_filter": "any", "top_k": 5}))
            # CreateTicket is added by escalation logic in pipeline, not here

        # Low confidence → fall back to rule-based
        if confidence < 0.60:
            return self._rule_based_calls(query, history)

        return calls

    def _infer_category(self, q: str) -> str:
        if any(w in q for w in ["bill","charge","invoice","payment","dispute","refund"]):
            return "billing"
        if any(w in q for w in ["signal","4g","5g","network","outage","slow","speed"]):
            return "network"
        if any(w in q for w in ["roam","international","abroad"]):
            return "roaming"
        if any(w in q for w in ["plan","recharge","prepaid","postpaid"]):
            return "plans"
        if any(w in q for w in ["sim","device","phone","imei"]):
            return "device"
        if any(w in q for w in ["account","kyc","port","otp"]):
            return "account"
        return "any"

    def _extract_region(self, query: str) -> str:
        import re
        cities = ["mumbai","delhi","bangalore","bengaluru","hyderabad","chennai",
                  "kolkata","pune","ahmedabad","surat","jaipur","lucknow"]
        q_lower = query.lower()
        for city in cities:
            if city in q_lower:
                return city.capitalize()
        return "Unknown"

    def _rule_based_predict(self, query: str) -> Tuple[str, float]:
        q = query.lower()
        if any(w in q for w in ["signal","outage","4g","5g","no service","network down"]):
            return "CheckNetworkStatus", 0.85
        if any(w in q for w in ["policy","regulation","trai","exact terms","section"]):
            return "GetPolicy", 0.85
        if any(w in q for w in ["escalate","complaint","supervisor","fraud","hacked"]):
            return "CreateTicket", 0.85
        return "SearchKB", 0.90

    def _rule_based_calls(self, query: str, history: List[Dict]) -> List[Tuple[str, Dict]]:
        """Mirrors the original tool_policy() from inference_pipeline.py."""
        q = query.lower()
        category = self._infer_category(q)
        calls = [("SearchKB", {"query": query, "category_filter": category, "top_k": 5})]

        network_kw = ["signal","4g","5g","network","outage","down","not working","slow internet"]
        if any(kw in q for kw in network_kw):
            region = self._extract_region(query)
            svc    = "5G" if "5g" in q else "4G" if "4g" in q else "all"
            calls.append(("CheckNetworkStatus", {"region": region, "service_type": svc}))

        policy_kw = ["policy","rule","regulation","how long","deadline","eligible","trai"]
        if any(kw in q for kw in policy_kw):
            calls.append(("GetPolicy", {"_deferred": True}))

        return calls


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    action="store_true", help="Train the classifier")
    parser.add_argument("--test",     action="store_true", help="Test on sample queries")
    parser.add_argument("--epochs",   type=int, default=5)
    args = parser.parse_args()

    if args.train:
        results = train_tool_policy(epochs=args.epochs)
        print(f"\nTraining complete. Val accuracy: {results['eval_accuracy']:.4f}")

    if args.test or not args.train:
        policy = TrainedToolPolicy()
        test_queries = [
            "My 4G is not working in Mumbai",
            "What does the TRAI policy say about billing disputes?",
            "I was charged Rs. 8000 unauthorized — I want to escalate",
            "How do I dispute a charge on my bill?",
            "Is there an outage in Bangalore right now?",
            "Show me the exact terms for data rollover",
            "This issue has been unresolved for 2 weeks, I need a supervisor",
        ]
        print("\nTool Policy Predictions:")
        print("-" * 50)
        for q in test_queries:
            tool, conf = policy.predict(q)
            print(f"  {tool:<22} ({conf:.2f})  {q}")

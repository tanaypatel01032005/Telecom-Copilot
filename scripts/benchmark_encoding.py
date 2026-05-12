import time
import torch
from sentence_transformers import SentenceTransformer

model_path = "checkpoints/retriever"
print(f"Loading model from {model_path}...", flush=True)
model = SentenceTransformer(model_path)

text = "This is a test passage for telecom copilot search indexer."
print("Encoding 1 passage...", flush=True)
t0 = time.time()
emb = model.encode([text])
print(f"Time taken: {time.time() - t0:.4f}s", flush=True)

print("Encoding 32 passages...", flush=True)
texts = [text] * 32
t0 = time.time()
embs = model.encode(texts)
print(f"Time taken: {time.time() - t0:.4f}s", flush=True)

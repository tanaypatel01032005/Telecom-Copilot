import os
from huggingface_hub import InferenceClient

token = os.getenv("HF_TOKEN")
model = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Connecting to {model}...", flush=True)
client = InferenceClient(model, token=token)

try:
    print("Sending test query...", flush=True)
    resp = client.chat_completion(messages=[{"role": "user", "content": "hello"}], max_tokens=10)
    print(f"Response: {resp.choices[0].message.content}", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)

# run_demo.py
import torch
import transformers
from core import get_tokenizer, SimpleScheduler, GPUMonitor

def load_gpt2():
    model_name = "gpt2"
    tokenizer = get_tokenizer(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, [model]

def main():
    tokenizer, blocks = load_gpt2()

    # 1️⃣  Monitor (optional)
    monitor = GPUMonitor()
    print("GPU(s) détectés :", [g["name"] for g in monitor.gpus])

    # 2️⃣  Scheduler
    scheduler = SimpleScheduler(blocks, callbacks={
        "on_start": lambda idx, x: print(f"START on {x.device}"),
        "on_end":   lambda idx, x: print(f"END on {x.device}"),
    })

    # 3️⃣  Prompt
    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    # 4️⃣  Inference
    logits = scheduler.forward(inputs)
    print("Logits shape :", logits.shape)

    # 5️⃣  Prediction
    pred = scheduler.predict(inputs)
    print("Token prédit :", tokenizer.decode(pred.tolist()))

if __name__

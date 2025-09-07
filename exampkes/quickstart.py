# examples/quickstart.py
import torch
import transformers
from core import get_tokenizer, SimpleScheduler, GPUMonitor

def load_gpt2():
    tokenizer = get_tokenizer("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, [model]

def main():
    tokenizer, blocks = load_gpt2()

    # 1️⃣  Monitor (optional)
    monitor = GPUMonitor()
    print("GPUs détectés :", [g["name"] for g in monitor.gpus])

    # 2️⃣  Scheduler
    scheduler = SimpleScheduler(blocks)

    # 3️⃣  Prompt
    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    # 4️⃣  Inference
    logits = scheduler.forward(inputs)
    print("Logits shape :", logits.shape)

    # 5️⃣  Prediction
    pred = scheduler.predict(inputs)
    print("Token prédit :", tokenizer.decode(pred.tolist()))

if __name__ == "__main__":
    main()

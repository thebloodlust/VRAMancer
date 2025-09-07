# run_demo.py
import torch
import transformers
from core.monitor import GPUMonitor
from core.scheduler import SimpleScheduler

def load_gpt2():
    model_name = "gpt2"   # Vous pouvez changer pour un autre modèle
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # Charger le modèle entier, puis l’envelopper dans une liste de blocs
    # (ici on ne le découpe pas, chaque bloc est un modèle complet)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, [model]   # un bloc unique

def main():
    tokenizer, blocks = load_gpt2()

    # Monitor (facultatif)
    monitor = GPUMonitor()
    print("GPU(s) détectés :", [g["name"] for g in monitor.gpus])

    # Scheduler
    scheduler = SimpleScheduler(blocks, verbose=True)

    # Prompt de test
    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    # Exécution
    logits = scheduler.forward(inputs)
    print("Logits shape :", logits.shape)

    # Prédiction simple
    pred = scheduler.predict(inputs)
    print("Token prédit :", tokenizer.decode(pred.tolist()))

if __name__ == "__main__":
    main()

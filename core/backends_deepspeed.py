# Backend DeepSpeed (stub)
try:
    import deepspeed
except ImportError:
    deepspeed = None

def is_deepspeed_available():
    return deepspeed is not None

def run_deepspeed_inference(model, inputs):
    if not is_deepspeed_available():
        raise ImportError("DeepSpeed n'est pas installé")
    # Stub : à compléter avec la logique DeepSpeed
    print("[DeepSpeed] Inference en cours...")
    return model(inputs)

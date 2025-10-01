# Backend TensorRT (stub)
try:
    import tensorrt as trt
except ImportError:
    trt = None

def is_tensorrt_available():
    return trt is not None

def run_tensorrt_inference(model, inputs):
    if not is_tensorrt_available():
        raise ImportError("TensorRT n'est pas installé")
    # Stub : à compléter avec la logique TensorRT
    print("[TensorRT] Inference en cours...")
    return model(inputs)

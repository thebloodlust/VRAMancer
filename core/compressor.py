class Compressor:
    def __init__(self, strategy="adaptive", verbose=True):
        self.strategy = strategy
        self.verbose = verbose

    def compress(self, layer_name, original_size_mb):
        """
        Simule une compression adaptative selon la stratégie.
        """
        if self.strategy == "aggressive":
            factor = 0.5
        elif self.strategy == "light":
            factor = 0.85
        else:  # adaptive
            factor = 0.75 if "layer" in layer_name else 0.9

        new_size = round(original_size_mb * factor, 2)
        if self.verbose:
            print(f"🗜️ Compression {layer_name} → {new_size}MB (stratégie: {self.strategy})")
        return new_size

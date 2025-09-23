def route(self, block, input_tensor, index=0, importance="normal", estimated_size_mb=100):
    backend = self.engine.backend
    ram_available, _ = self.engine.get_ram_status()

    # 🔁 Poids élevé → éviter RAM si < seuil
    if estimated_size_mb > 1000 and ram_available < 2 * 1024**3:
        if self._nvme_available():
            print(f"📦 Bloc {index} → NVMe (poids élevé)")
            block = load_block_from_disk(f"blocks/block_{index}.pt")
            return block(input_tensor)

    # 🔁 Importance critique → éviter réseau
    if importance == "critical" and backend in ["cuda", "rocm", "mps"]:
        device = self.engine._get_device(index)
        print(f"📦 Bloc {index} → {device} (critique)")
        return block.to(device)(input_tensor)

    # 🔁 Importance faible → autoriser réseau
    if importance == "low":
        print(f"📦 Bloc {index} → Réseau (faible priorité)")
        remote = RemoteBlock("192.168.1.42", 9000)
        return remote.forward(input_tensor)

    # 🔁 Fallback RAM ou CPU
    if ram_available > 2 * 1024**3:
        print(f"📦 Bloc {index} → CPU (RAM disponible)")
        return block.to("cpu")(input_tensor)

    # Dernier recours
    print(f"📦 Bloc {index} → Réseau (fallback)")
    remote = RemoteBlock("192.168.1.42", 9000)
    return remote.forward(input_tensor)

from transformers import AutoModel, AutoConfig
import torch
import json

class ModelSplitter:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.layers = self._extract_layers()

    def _extract_layers(self):
        """
        Retourne une liste des couches du modèle avec leur nom, taille estimée et type.
        """
        layers = []
        for name, param in self.model.named_parameters():
            size_mb = param.numel() * param.element_size() / (1024 ** 2)
            layer_type = self._classify_layer(name)
            layers.append({
                "name": name,
                "size_mb": round(size_mb, 2),
                "requires_grad": param.requires_grad,
                "type": layer_type
            })
        return layers

    def _classify_layer(self, name):
        if any(key in name for key in ["attention", "embeddings", "output"]):
            return "core"
        return "secondary"

    def estimate_total_memory(self):
        """
        Retourne la mémoire totale estimée du modèle.
        """
        total = sum(layer["size_mb"] for layer in self.layers)
        return round(total, 2)

    def split_by_gpu(self, gpus):
        """
        Répartit les couches entre GPU selon leur type et VRAM dispo.
        """
        allocation = {gpu["id"]: [] for gpu in gpus}
        for layer in self.layers:
            target_gpu = 0 if layer["type"] == "core" else self._find_secondary_gpu(gpus)
            allocation[target_gpu].append(layer)
        return allocation

    def _find_secondary_gpu(self, gpus):
        secondary_gpus = [gpu for gpu in gpus if gpu["id"] != 0]
        sorted_gpus = sorted(secondary_gpus, key=lambda g: g["total_vram_mb"], reverse=True)
        return sorted_gpus[0]["id"] if sorted_gpus else 0

    def export_allocation(self, allocation, filename="layer_allocation.json"):
        with open(filename, "w") as f:
            json.dump(allocation, f, indent=2)
        print(f"✅ Allocation exportée dans {filename}")

    def visualize_layers(self):
        print("📊 Couches du modèle :")
        for layer in self.layers:
            print(f" - {layer['name']} | {layer['size_mb']}MB | {layer['type']}")

from transformers import AutoModel, AutoConfig
import torch

class ModelSplitter:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.layers = self._extract_layers()

    def _extract_layers(self):
        """
        Retourne une liste des couches du modèle avec leur nom et taille estimée.
        """
        layers = []
        for name, param in self.model.named_parameters():
            size_mb = param.numel() * param.element_size() / (1024 ** 2)
            layers.append({
                "name": name,
                "size_mb": round(size_mb, 2),
                "requires_grad": param.requires_grad
            })
        return layers

    def classify_layers(self):
        """
        Classe les couches en 'core' ou 'secondary' selon leur nom.
        """
        classified = []
        for layer in self.layers:
            if any(key in layer["name"] for key in ["attention", "embeddings", "output"]):
                layer["type"] = "core"
            else:
                layer["type"] = "secondary"
            classified.append(layer)
        return classified

    def split_by_gpu(self, gpus):
        """
        Répartit les couches entre GPU selon leur type et VRAM dispo.
        """
        classified = self.classify_layers()
        allocation = {gpu["id"]: [] for gpu in gpus}

        for layer in classified:
            target_gpu = 0 if layer["type"] == "core" else self._find_secondary_gpu(gpus)
            allocation[target_gpu].append(layer)

        return allocation

    def _find_secondary_gpu(self, gpus):
        """
        Choisit un GPU secondaire (≠ 0) avec le plus de VRAM.
        """
        secondary_gpus = [gpu for gpu in gpus if gpu["id"] != 0]
        sorted_gpus = sorted(secondary_gpus, key=lambda g: g["total_vram_mb"], reverse=True)
        return sorted_gpus[0]["id"] if sorted_gpus else 0

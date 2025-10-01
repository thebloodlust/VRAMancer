"""
Stub backend Ollama pour VRAMancer
Gestion d’erreur, docstring, hooks pour intégration réelle.
"""
class OllamaBackend:
    """
    Backend Ollama (LLM local) pour VRAMancer.
    Méthodes : load_model, infer, unload, get_status
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.status = "init"

    def load_model(self):
        """Charge le modèle Ollama."""
        try:
            # TODO: Intégration réelle Ollama
            self.model = "stub_model"
            self.status = "loaded"
        except Exception as e:
            self.status = f"error: {e}"
            raise

    def infer(self, prompt: str) -> str:
        """Effectue une inférence sur le modèle Ollama."""
        if self.model is None:
            raise RuntimeError("Modèle non chargé")
        try:
            # TODO: Appel réel Ollama
            return f"Réponse Ollama (stub) pour : {prompt}"
        except Exception as e:
            self.status = f"error: {e}"
            return "[Erreur Ollama]"

    def unload(self):
        """Décharge le modèle Ollama."""
        self.model = None
        self.status = "unloaded"

    def get_status(self) -> str:
        """Renvoie le statut du backend."""
        return self.status
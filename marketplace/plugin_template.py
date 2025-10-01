# Template plugin VRAMancer
class VRAMancerPlugin:
    def __init__(self):
        self.name = "MonPlugin"
        self.version = "1.0"
    def activate(self):
        print(f"Plugin {self.name} activé !")
    def deactivate(self):
        print(f"Plugin {self.name} désactivé !")
# Exemple d’intégration
if __name__ == "__main__":
    plugin = VRAMancerPlugin()
    plugin.activate()
    plugin.deactivate()

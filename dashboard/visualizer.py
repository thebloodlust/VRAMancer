# ... (le code que vous aviez déjà) ...

class GpuMonitorGUI:
    # … votre constructeur existant …

    # --------------------------------------------------------------------
    # Méthode d’appel depuis le helper `update_dashboard()`
    # --------------------------------------------------------------------
    def update(self, memory_state: dict[int, dict[str, int]]) -> None:
        """
        `memory_state` a la même forme que celle retournée par
        `MemoryBalancer.get_memory_state()`.
        On met à jour les champs “total / used” et on redessine le tracé.
        """
        # Mise à jour des variables internes
        for gpu_id, state in memory_state.items():
            if gpu_id in self.gpu_state:
                self.gpu_state[gpu_id]["used"]   = state["used"]
                self.gpu_state[gpu_id]["total"]  = state["total"]

        # On déclenche un rafraîchissement dans le thread UI
        self.after(0, self._refresh_canvas)

    # Méthode interne qui redessine le canvas (identique à ce que vous aviez)
    def _refresh_canvas(self) -> None:
        self.canvas.delete("all")
        # … le même code que dans votre `__init__` qui trace les rectangles …
        # Vous pouvez copier‑coller la partie du code qui se trouve dans
        # `self._draw_gpu_rectangles()` que vous avez déjà dans votre widget.

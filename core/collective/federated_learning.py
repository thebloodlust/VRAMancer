"""
Federated Learning natif :
- Agrégation sécurisée, privacy-preserving
"""
class FederatedLearner:
    def __init__(self, peers):
        self.peers = peers
        self.clip_norm = 10.0  # baseline DP-like clipping
        self.noise_sigma = 0.0  # activer >0 pour ajouter bruit gaussien

    def aggregate(self, updates):
        print("[Federated] Agrégation des modèles")
        return sum(updates) / len(updates)

    def aggregate_weighted(self, values, weights):
        if not values:
            raise ValueError("no values")
        if len(values) != len(weights):
            raise ValueError("length mismatch")
        total_w = sum(weights)
        if total_w == 0:
            raise ValueError("zero weight")
        # clipping + bruit optionnel
        acc = 0.0
        for v, w in zip(values, weights):
            fv = float(v)
            if fv > self.clip_norm:
                fv = self.clip_norm
            elif fv < -self.clip_norm:
                fv = -self.clip_norm
            acc += fv * float(w)
        import random, math
        if self.noise_sigma > 0:
            # Box-Muller simple
            u1, u2 = random.random(), random.random()
            z = math.sqrt(-2*math.log(u1)) * math.cos(2*math.pi*u2)
            acc += z * self.noise_sigma
        return acc / total_w

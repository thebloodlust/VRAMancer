import logging
logging.basicConfig(level=logging.INFO)

from core.network.trust_ring import TRUST_MANAGER

if __name__ == "__main__":
    print("\n--- Création de Cercles de Confiance (Trust Rings) ---")
    
    # Créer un cercle privé
    ring_info = TRUST_MANAGER.create_ring(
        name="Cabinet d'Avocats Alpha"
    )
    
    print("\n[SUCCÈS] Nouveau groupe restreint créé :")
    print(f"Nom du groupe : {ring_info['name']}")
    print(f"ID du Ring    : {ring_info['ring_id']}")
    print(f"Clé Secrète   : {ring_info['token']}")
    
    print("\nPour connecter un noeud (ex: PC portable) à ce groupe spécifique, l'URL WebSocket sera :")
    print(f"ws://VOTRE_IP:5060/?ring_id={ring_info['ring_id']}&token={ring_info['token']}")
    print("\nTout hôte ne connaissant pas ce couple ID/Token sera rejeté par l'Orchestrateur.")

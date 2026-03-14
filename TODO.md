# VRAMancer Todo

- Wake On Inference
- WebGPU
- Tests Multi OS PC et MAC
- Déterminer/Forcer l'ordre des GPUs (L0 vs L1) : prioriser la RTX 5070 Ti en maître pour profiter de ses cœurs Tensor NVFP4 (architecture Blackwell) et booster les performances de calcul.
- Implémenter un équilibrage de charge asymétrique (Load Imbalance) : répartir le calcul proportionnellement à la puissance (ex: 65% pour la 5070 Ti, 35% pour la 3090) plutôt qu'à 50/50. Objectif : réduire le temps d'attente entre les cartes lors du Prefill, et faire s'envoler les tok/s.

## Swarm Security & Scalability
- **Groupes Privés (Cercles de Confiance) :** Implémenter un système d'invitation par jetons (Token-based Room/Group) pour créer des "mini-Swarm" privés. Un Master ne doit accepter que les nœuds (navigateurs/clients) disposant de la clé cryptographique du groupe, rejetant instantanément les inconnus ou les connexions publiques sauvages.
- **Redondance K-Répétition (Fault Tolerance) :** Assurer la sauvegarde de l'état (Checkpointing) et dupliquer les "chunks" du modèle sur plusieurs nœuds pour que la déconnexion brutale d'un PC portable pendant la génération ne plante pas la phrase de l'IA.

## Mobile Edge AI & NPU Integration
- **Support WebGPU/WebNN Mobile :** Intégrer les smartphones au Swarm. Les puces récentes (Apple Neural Engine, Snapdragon NPU, Google Tensor) ont des capacités matricielles exceptionnelles.
- **Profilage Asymétrique Extrême :** Le `layer_profiler` doit pouvoir assigner dynamiquement des petites charges aux téléphones (ex: 1 couche d'attention) et des grosses charges aux serveurs (ex: 20 couches), selon leur bande passante (WiFi/5G) et leur VRAM mobile (UMa).
- **Wake-Lock / Battery Aware :** Empêcher la mise en veille du téléphone pendant le calcul et stopper l'effort si la batterie passe sous les 20% ou si le téléphone n'est pas branché sur secteur.

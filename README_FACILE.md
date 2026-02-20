# 🚀 VRAMancer - Guide Ultra Facile (Plug & Play)

Bienvenue sur VRAMancer ! Ce guide est fait pour vous si vous voulez juste lancer l'IA sans vous prendre la tête avec du code.

## 💻 Vos Machines (Votre Cluster)

Vous avez un matériel incroyable. Voici comment VRAMancer va l'utiliser :

1. **Le Monstre (Serveur Principal)** : EPYC 7402 + 256 Go RAM + RTX 3090 (24Go) + RTX 5070 Ti (16Go).
   - *Rôle* : C'est le cerveau. Il charge les plus gros modèles (comme Llama 3 70B) et utilise le C++ ultra-rapide pour faire communiquer les deux cartes graphiques.
2. **Le Portable (Renfort)** : Intel 12ème Gen + RTX 4060 (8Go).
   - *Rôle* : Il se connecte en Wi-Fi ou câble au serveur et prête ses 8 Go de VRAM quand le serveur est plein.
3. **Le Mac Mini M4 (Renfort Apple)** : Puce M4.
   - *Rôle* : Il utilise sa mémoire unifiée ultra-rapide (MPS) pour calculer des petits bouts du modèle en renfort.

---

## 📥 1. Télécharger l'Exécutable (Pas besoin d'installer Python !)

Nous avons créé des exécutables "tout-en-un" (Standalone). Vous n'avez **rien à installer** (ni Python, ni PyTorch).

1. Allez dans l'onglet **Releases** du projet (ou dans le dossier `dist/` si vous l'avez compilé).
2. Téléchargez le fichier correspondant à votre machine :
   - Pour le Serveur EPYC (Linux) : `vramancer-linux`
   - Pour le Portable (Windows) : `vramancer.exe`
   - Pour le Mac Mini (macOS) : `vramancer-macos`

---

## 🚀 2. Lancer le Serveur Principal (Le Monstre EPYC)

Sur votre gros serveur Linux, ouvrez un terminal dans le dossier où vous avez téléchargé le fichier et tapez :

```bash
./vramancer-linux start --model "meta-llama/Llama-3-70b-instruct" --master
```

*C'est tout !* Le serveur va télécharger le modèle, le couper en deux (un bout sur la 3090, un bout sur la 5070 Ti), et attendre les connexions.

---

## 🔌 3. Connecter les Renforts (Plug & Play)

### Sur le Portable Windows (RTX 4060)
Double-cliquez sur `vramancer.exe` ou ouvrez l'invite de commande et tapez :
```cmd
vramancer.exe join --master-ip ADRESSE_IP_DU_SERVEUR
```

### Sur le Mac Mini M4
Ouvrez le terminal et tapez :
```bash
./vramancer-macos join --master-ip ADRESSE_IP_DU_SERVEUR
```

*Magie !* Le serveur EPYC va détecter automatiquement la 4060 et le Mac M4 sur le réseau et leur envoyer des calculs. Vous venez de créer un supercalculateur distribué !

---

## 💬 4. Discuter avec l'IA

Ouvrez votre navigateur web sur n'importe quel appareil de la maison et allez sur :
👉 **http://ADRESSE_IP_DU_SERVEUR:5000**

Vous verrez une interface de chat (comme ChatGPT) prête à l'emploi.

---

## ❓ Problèmes fréquents

- **"Je n'ai pas l'exécutable"** : Demandez au développeur de lancer `python build_standalone.py` pour les générer.
- **"Les PC ne se voient pas"** : Vérifiez que le pare-feu Windows du portable autorise `vramancer.exe`.
- **"C'est lent sur le réseau"** : Si possible, branchez le portable et le Mac en câble Ethernet (ou USB4) plutôt qu'en Wi-Fi.

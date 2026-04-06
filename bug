=== INSTRUCTIONS WINDOWS - Cross-node VTP bench ===

1) Ouvrir un CMD

2) Aller dans le SOUS-DOSSIER VRAMancer (celui qui contient core/) :
   cd C:\Users\jerem\Documents\GitHub\VRAMancer\VRAMancer
   
   IMPORTANT : tu dois voir le dossier "core" si tu fais "dir core"

3) git pull

4) Activer le venv :
   .venv\Scripts\activate
   
   Si le venv n'existe pas :
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   pip install flask transformers accelerate safetensors requests huggingface_hub sentencepiece protobuf

5) Lancer le worker VTP :

set VRM_DISABLE_RATE_LIMIT=1
set VRM_API_TOKEN=testtoken
set VRM_CLUSTER_SECRET=testtoken
set VRM_BACKEND_ALLOW_STUB=1
python -c "import sys; sys.path.insert(0,'.'); from core.cross_node import VTPWorkerServer; s = VTPWorkerServer('0.0.0.0', 18951); s.start(); input('Worker VTP ready on port 18951, press Enter to stop...')"

6) Ca doit afficher "Worker VTP ready on port 18951, press Enter to stop..."
   Confirme dans le chat quand c'est bon.

=== UPDATE: j'ai corrigé un bug, le serveur renvoyait rien ===
Ferme le worker (Entrée), puis:
   git pull
Puis relance la commande de l'étape 5 ci-dessus.

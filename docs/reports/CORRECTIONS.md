# VRAMancer — Corrections à effectuer

> Liste priorisée de toutes les corrections nécessaires, avec explication et contexte.  
> **P0** = bloquant déploiement · **P1** = avant production · **P2** = qualité · **P3** = cosmétique

---

## P0 — Corrections critiques (bloquent le déploiement)

### 1. Pickle RCE — Exécution de code arbitraire à distance

**Fichiers :** `core/network/remote_executor.py`, `core/hierarchical_memory.py`, `core/network/transmission.py`

**Problème :** `pickle.loads()` est utilisé pour désérialiser des données reçues du réseau ou lues depuis le cache NVMe. Un attaquant peut forger un payload pickle qui exécute du code arbitraire sur le serveur (Remote Code Execution). C'est la vulnérabilité Python la plus dangereuse qui existe.

**Correction :**
```python
# AVANT (dangereux)
data = pickle.loads(payload)

# APRÈS (sécurisé)
import safetensors.torch as st
data = st.load(payload)  # Pour les tenseurs

# Ou pour les métadonnées :
import msgpack
data = msgpack.unpackb(payload, raw=False)
```

**Impact :** Passer SafeTensors pour les tenseurs, MessagePack/JSON pour les métadonnées. Tester la compatibilité avec les clients existants.

---

### 2. select_backend() — Crash si vLLM absent

**Fichier :** `core/backends.py`

**Problème :** Si le backend choisi n'est ni `"huggingface"`, ni `"vllm"`, ni `"ollama"`, la fonction retourne `VLLMBackend` par défaut — qui crashe si vLLM n'est pas installé. Le stub `VRM_BACKEND_ALLOW_STUB` n'est pas vérifié dans ce chemin.

**Correction :**
```python
# AVANT
else:
    return VLLMBackend(model_name)

# APRÈS
else:
    if os.environ.get("VRM_BACKEND_ALLOW_STUB"):
        return StubBackend(model_name)
    return HuggingFaceBackend(model_name)  # fallback sûr
```

---

### 3. asyncio.run() non thread-safe dans le hot path

**Fichier :** `core/backends.py` (HuggingFaceBackend)

**Problème :** `asyncio.run()` crée une nouvelle event loop à chaque appel. Si appelé depuis un thread Flask qui a déjà une loop (ou depuis un autre `asyncio.run()`), ça lève `RuntimeError: This event loop is already running` ou crée des race conditions.

**Correction :**
```python
# AVANT
result = asyncio.run(self._async_generate(prompt))

# APRÈS
try:
    loop = asyncio.get_running_loop()
    future = asyncio.ensure_future(self._async_generate(prompt))
    result = loop.run_until_complete(future)
except RuntimeError:
    result = asyncio.run(self._async_generate(prompt))
```

Ou mieux : utiliser un `ThreadPoolExecutor` dédié avec sa propre loop persistante.

---

### 4. Auth API optionnelle

**Fichier :** `core/production_api.py`

**Problème :** L'authentification est bypassée pour les routes Swarm Ledger et certaines routes de gestion. Un nœud malicieux peut appeler `/api/models/load` ou `/api/generate` sans token.

**Correction :** Rendre `install_security(app)` obligatoire (pas conditionnel). Vérifier que TOUTES les routes passent par le `before_request` d'authentification. Les seules exceptions doivent être `/health`, `/live`, `/ready`.

---

### 5. Credentials loggées en clair

**Fichier :** `core/security/remote_access.py`

**Problème :** Les mots de passe admin/user et codes MFA sont affichés dans les logs au démarrage, y compris en production.

**Correction :**
```python
# AVANT
logger.warning(f"Using default admin password: {admin_pass}")

# APRÈS
logger.warning("Using default admin password — change VRM_REMOTE_ADMIN_PASS immediately")
```

---

### 6. Token hardcodé "default_insecure_token"

**Fichier :** `core/block_router.py`

**Problème :** Un token par défaut est utilisé si `VRM_API_TOKEN` n'est pas défini. Ce token est dans le code source, donc connu de tous.

**Correction :** Refuser de démarrer si `VRM_API_TOKEN` n'est pas défini en mode production (`VRM_PRODUCTION=1`). En mode dev, générer un token aléatoire au démarrage et l'afficher une seule fois.

---

### 7. trust_remote_code=True inconditionnel

**Fichier :** `core/backends.py`

**Problème :** `AutoModel.from_pretrained(..., trust_remote_code=True)` exécute du code Python arbitraire contenu dans le repo HuggingFace du modèle. Si un modèle malicieux est chargé, c'est une RCE.

**Correction :** Mettre `trust_remote_code=False` par défaut. Ajouter un env var `VRM_TRUST_REMOTE_CODE=1` pour les modèles qui en ont besoin (ex: Falcon, MPT).

---

## P1 — Corrections haute priorité (avant production)

### 8. 4 fichiers tronqués/incomplets

**Fichiers :** `core/continuous_batcher.py`, `core/cross_vendor_bridge.py`, `core/vram_lending.py`, `core/speculative_decoding.py`

**Problème :** Ces fichiers sont coupés en pleine implémentation (syntaxe invalide ou fonctions vides). Ils ne crashent pas au runtime car jamais importés directement, mais toute tentative d'utilisation échouera.

**Correction :** Compléter l'implémentation ou les remplacer par des stubs propres qui lèvent `NotImplementedError` avec un message explicatif.

---

### 9. stop_polling() ne fonctionne pas

**Fichier :** `core/monitor.py`

**Problème :** `GPUMonitor.stop_polling()` set un flag mais le thread daemon ne le vérifie pas correctement (race condition sur le `self._running` flag). Le thread continue parfois après l'arrêt.

**Correction :** Utiliser un `threading.Event` au lieu d'un booléen :
```python
self._stop_event = threading.Event()

# Dans le thread :
while not self._stop_event.is_set():
    self._stop_event.wait(timeout=self.interval)

# Pour arrêter :
def stop_polling(self):
    self._stop_event.set()
```

---

### 10. Pas de TLS sur les communications mesh

**Fichiers :** `core/network/fibre_fastpath.py`, `core/network/llm_transport.py`, `core/network/cluster_discovery.py`

**Problème :** Toutes les communications inter-nœuds (RDMA, TCP, UDP discovery) sont en clair. Un attaquant sur le réseau peut intercepter les tenseurs, les modifier, ou usurper un nœud.

**Correction :** Ajouter TLS mutuel (mTLS) sur les connexions TCP. Pour RDMA, utiliser IPsec ou le chiffrement IB natif. Pour la discovery, signer les annonces avec HMAC.

---

### 11. Métriques Prometheus exposées sans auth

**Fichier :** `core/metrics.py`

**Problème :** Le serveur Prometheus écoute sur `0.0.0.0:9108` par défaut. Tout le monde sur le réseau peut accéder aux métriques, qui révèlent la topologie GPU, les modèles chargés, et les patterns d'utilisation.

**Correction :**
```python
# Bind localhost par défaut
bind_addr = os.environ.get("VRM_METRICS_BIND", "127.0.0.1")
start_http_server(9108, addr=bind_addr)
```

---

### 12. supervision_api.py et actions.py sans authentification

**Fichier :** `core/network/supervision_api.py`, `core/network/actions.py`

**Problème :** Les endpoints de supervision (reboot nœud, failover, status) n'ont aucune vérification de token. N'importe qui peut redémarrer un nœud du cluster.

**Correction :** Intégrer ces routes dans le système `install_security()` existant ou ajouter le décorateur `@require_token` sur chaque route.

---

### 13. AITP/multicast sans authentification

**Fichiers :** `core/network/aitp_protocol.py`, `core/network/aitp_sensing.py`

**Problème :** Les paquets AITP et les annonces multicast ne sont pas signés. Un nœud malicieux peut usurper une identité, annoncer de fausses capacités (1000 TFLOPS), ou rediriger des tenseurs.

**Correction :** Ajouter une signature HMAC sur chaque paquet AITP. Distribuer le secret partagé via le mécanisme `VRM_CLUSTER_SECRET`.

---

### 14. Threads daemon jamais nettoyés

**Fichiers :** `core/stream_manager.py`, `core/hierarchical_memory.py`, `core/inference_pipeline.py`, `core/monitor.py`

**Problème :** De nombreux threads daemon sont lancés (monitoring, autosave, eviction) mais jamais stoppés proprement. Au shutdown, ces threads peuvent corrompre des données.

**Correction :** Implémenter un `shutdown()` method sur chaque module qui :
1. Set un `threading.Event` de stop  
2. `join(timeout=5)` chaque thread  
3. Log un warning si le thread ne s'arrête pas

---

### 15. Version désynchronisée

**Fichiers :** `core/__init__.py` (1.5.0), `core/api/routes_ops.py` (fallback 0.2.4), `vramancer/main.py`, `pyproject.toml`, `setup.cfg`

**Problème :** La version est `1.5.0` dans le code mais `0.2.4` dans la documentation et certains fallbacks. Les routes `/health` retournent parfois `0.2.4`.

**Correction :** Définir une seule source de vérité (`core/__init__.__version__`) et faire référencer tous les autres fichiers vers elle.

---

## P2 — Corrections qualité

### 16. CORS trop permissif

**Fichier :** `core/production_api.py`

**Problème :** `CORS(app, origins="*")` autorise tous les domaines à appeler l'API. En production, restreindre aux domaines autorisés.

**Correction :** `CORS(app, origins=os.environ.get("VRM_CORS_ORIGINS", "http://localhost:*"))`

---

### 17. Bare except: pass dans le réseau

**Fichiers :** `core/network/aitp_sensing.py`, `core/network/cluster_discovery.py`, et ~10 autres

**Problème :** `except: pass` avale toutes les exceptions y compris `KeyboardInterrupt` et `SystemExit`. Impossible de débugguer les erreurs réseau.

**Correction :** Remplacer par `except Exception as e: logger.debug(f"...: {e}")` minimum.

---

### 18. print() en production

**Fichiers :** `core/backends.py`, `core/inference_pipeline.py`

**Problème :** Des `print()` sont utilisés au lieu du logger, notamment pour les erreurs d'inférence. Ces messages ne sont ni horodatés, ni routables vers les outils de monitoring.

**Correction :** Remplacer tous les `print()` par `logger.error()` ou `logger.warning()`.

---

### 19. HTML inline dans dashboard_web.py

**Fichier :** `dashboard/dashboard_web.py`

**Problème :** Le template HTML Three.js est en string dans le fichier Python (~500 lignes de HTML/JS inline). Difficile à maintenir, pas de linting HTML.

**Correction :** Extraire vers `dashboard/templates/dashboard.html` et utiliser `render_template()`.

---

### 20. Résolution device répétée dans le hot path

**Fichier :** `core/backends.py`

**Problème :** `next(self.model.parameters()).device` est appelé 4+ fois par forward pass. C'est un parcours de l'iterator des paramètres à chaque fois.

**Correction :** Cacher le device au `load_model()` :
```python
self._device = next(self.model.parameters()).device
```

---

## P3 — Corrections cosmétiques

### 21. Commentaires FR/EN mélangés
Uniformiser la langue des commentaires (EN recommandé pour un projet open-source).

### 22. Emoji dans les logs
Retirer les "🧠", "🔮", "⚡" des messages de log — non professionnel et casse certains terminaux.

### 23. Code mort à supprimer
- `core/backends_deepspeed.py` — jamais importé par `select_backend()`
- `core/backends_tensorrt.py` — jamais importé
- `core/orchestrator/adaptive_routing.py` — import circulaire, inutilisé

### 24. Fichiers .py.bak
- `vramancer/main.py.bak` — fichier backup à supprimer du repo

---

## Synthèse

| Priorité | Nombre | Type | Effort estimé |
|----------|--------|------|---------------|
| **P0** | 7 | Sécurité + crash | 2-3 jours |
| **P1** | 7 | Stabilité + sécurité réseau | 5-7 jours |
| **P2** | 5 | Qualité + maintenabilité | 2-3 jours |
| **P3** | 4 | Cosmétique | 1 jour |
| **Total** | **23** | | **~2 semaines** |

**Ordre d'exécution recommandé :**
1. P0 #1 (pickle) + P0 #5 (credentials) + P0 #6 (token) — 1 jour, impact sécurité maximal
2. P0 #2 (select_backend) + P0 #3 (asyncio) — 1 jour, stabilité
3. P0 #4 (auth) + P0 #7 (trust_remote_code) — 1 jour, sécurité
4. P1 #8 (fichiers tronqués) — 3-5 jours, fonctionnalité
5. P1 #10-13 (réseau) — 2-3 jours, sécurité périmètre
6. P2 puis P3 en continu

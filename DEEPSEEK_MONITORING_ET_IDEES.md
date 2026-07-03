# DeepSeek → Opus : Monitoring & dernières idées

> État des lieux du monitoring existant + dashboard one-liner + dernières idées.
> Date : 2026-06-15.

---

# PARTIE 1 — État des lieux monitoring

## Ce qui existe DÉJÀ (solide)

| Composant | Fichier | Statut |
|---|---|---|
| **Prometheus config** | `monitoring/prometheus.yml` | ✅ Prêt |
| **Grafana dashboard** | `monitoring/grafana_dashboard.json` (616 lignes) | ✅ Prêt |
| **22 règles d'alerte** | `monitoring/alerting_rules.yml` | ✅ Prêtes |
| **Alertmanager** | `monitoring/alertmanager.yml` | ✅ Prêt |
| **K8s deployment example** | `monitoring/k8s-deployment-example.yaml` | ✅ Prêt |
| **57 métriques Prometheus** | `core/metrics.py` (Counter, Gauge, Histogram) | ✅ Exposées |
| **Health checks (15 fonctions)** | `core/health.py` | ✅ GPU, KV, transfert, lending, système |
| **Supervision API** | `core/network/supervision_api.py` | ✅ REST + WebSocket, HA sync HMAC |
| **Dashboard web (Flask)** | `dashboard/dashboard_web.py` | 🟡 Demo/local only |
| **Dashboard CLI** | `dashboard/cli_dashboard.py` | ✅ Terminal temps réel |
| **GPU polling** | `core/monitor.py` | ✅ Multi-accélérateur |
| **Grafana provisioning** | `monitoring/grafana_provisioning/` | ✅ Auto-config |

## Ce qui manque

| Gap | Impact |
|---|---|
| **Dashboard intégré à `vramancer serve`** — accessible sans config | ★★★ |
| **Vue multi-nœuds** — tous les GPUs du cluster sur une page | ★★★ |
| **Topologie visuelle** — graphe des GPUs, liens PCIe, bande passante | ★★☆ |
| **Historique** — rétention 24h/7j sans Prometheus externe | ★★☆ |
| **Alertes temps réel** — notif quand GPU OOM ou nœud down | ★★☆ |
| **Dashboard mobile** — PWA, checker ses GPUs depuis le téléphone | ★☆☆ |

---

# PARTIE 2 — Proposition : dashboard one-liner

## Le concept

```
$ vramancer serve Qwen2.5-14B
─────────────────────────────────────────────────
  API        → http://localhost:5030
  Dashboard  → http://localhost:5030/dash
  Métriques  → http://localhost:5030/metrics
  Health     → http://localhost:5030/health
─────────────────────────────────────────────────
  GPU0 (5070 Ti) : 12.4/16.0 GB | 72°C | FP4
  GPU1 (3090)    :  8.2/24.0 GB | 65°C | BF16
  Tok/s : 87.3  |  Queue : 3 req  |  Uptime : 4h 12m
─────────────────────────────────────────────────
```

## Ce qu'il montre (une seule page, sans scroll)

```
┌──────────────────────────────────────────────────────────────┐
│  VRAMancer                                       [⚙️] [🌙]  │
│  Qwen2.5-14B · 87 tok/s · Uptime 4h12m                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                  │
│  │ 5070 Ti (FP4)   │  │ 3090 (BF16)      │                  │
│  │ ████████░░ 12.4GB│  │ ████░░░░░░  8.2GB│                  │
│  │ 72°C · 140W     │  │ 65°C ·  95W      │                  │
│  │ DECODE · 3 req  │  │ PREFILL · idle   │                  │
│  └──────────────────┘  └──────────────────┘                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Throughput (dernière minute)                         │    │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 87.3 tok/s       │    │
│  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░ Queue : 3 req          │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  Dernières requêtes :                                        │
│  ✓ #42 "Write a CSV parser" 512→128 tok  2.3s  OK          │
│  ✓ #43 "Explain async/await" 256→64 tok   1.1s  OK          │
│  ⚠ #44 "Analyze this log"   8096→256 tok  8.2s  OK (long)  │
│                                                              │
│  [Prometheus metrics] [API docs] [Health check]              │
└──────────────────────────────────────────────────────────────┘
```

## Implémentation

- **Backend** : une route Flask supplémentaire `/dash` + `/api/dash/stats` (JSON)
- **Frontend** : une seule page HTML vanilla (pas de React/Vue, zéro dépendance)
- **Données** : appelle les 57 métriques Prometheus ET `health.py` en local
- **Rafraîchissement** : polling 2s ou Server-Sent Events (SSE)
- **Temps estimé** : 1-2 sessions (HTML/CSS pur, données déjà disponibles)

---

# PARTIE 3 — Dernières idées fraîches

## Idée M1 ★★★ — `vramancer doctor` : diagnostic automatique

Une commande qui vérifie TOUT et dit ce qui va pas :

```
$ vramancer doctor
─────────────────────────────────────────────────────
✅ CUDA 12.4 détecté
✅ GPU0 : RTX 5070 Ti (16 GB, Blackwell, FP4 OK)
✅ GPU1 : RTX 3090 (24 GB, Ampere)
⚠️  P2P : indisponible (GPUs consumer sans NVLink)
   → Les transferts passeront par le CPU (25 GB/s)
   → OK pour de l'inférence, lent pour du split de modèle
⚠️  ReBAR : activé sur GPU1 (32 GB) mais pas exploité
   → Exécutez 'vramancer rebar enable' pour benchmarker
✅ RAM : 128 GB (94 GB libres)
✅ NVMe : 1.8 TB libres pour cache modèle
✅ PyTorch 2.6 + transformers 4.50
⚠️  accelerate 1.2 : version ancienne, 1.5 disponible
   → pip install --upgrade accelerate
✅ Réseau : 1 Gbps (192.168.1.100)

RECOMMANDATION :
  Meilleure config = FP4 sur 5070 Ti, BF16 master sur 3090
  Commande : vramancer serve Qwen2.5-14B --quant fp4 --master gpu1
  Tok/s estimé : ~85-90

PRÊT. Lancez 'vramancer serve Qwen2.5-14B' pour commencer.
```

**Valeur** : plus jamais "pourquoi ça marche pas ?" sans réponse. Le diagnostic
détecte les problèmes AVANT que l'utilisateur les rencontre.

---

## Idée M2 ★★☆ — Playground web intégré

Un bac à sable dans le dashboard pour tester prompts et paramètres :

```
┌─────────────────────────────────────────────────────┐
│  [Prompt]                                           │
│  ┌─────────────────────────────────────────────┐    │
│  │ Write a Python function that parses a CSV   │    │
│  │ file and returns a dictionary...            │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  Max tokens : [128▾]  Temperature : [0.7▾]          │
│  Model : [Qwen2.5-14B ▾]                            │
│                                                     │
│  [Generate]  [Compare A/B]                          │
│                                                     │
│  ┌─ Résultat ─────────────────────────────────┐    │
│  │ ```python                                   │    │
│  │ import csv                                  │    │
│  │ def parse_csv(filepath):                    │    │
│  │     with open(filepath, 'r') as f:          │    │
│  │         reader = csv.DictReader(f)          │    │
│  │ ...                                         │    │
│  │ ```                                         │    │
│  │ 128 tokens · 2.3s · 55.6 tok/s              │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

**Valeur** : tester des modèles sans quitter le navigateur. Idéal pour comparer
rapidement deux quantisations ou deux modèles.

---

## Idée M3 ★★☆ — Historique local léger (SQLite)

Garder les 1000 dernières requêtes avec métadonnées :

```sql
-- Stockage local, léger
CREATE TABLE requests (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    model TEXT,
    prompt_hash TEXT,      -- hashé, pas en clair
    prompt_tokens INTEGER,
    generated_tokens INTEGER,
    duration_ms REAL,
    tok_s REAL,
    gpu0_vram_mb REAL,
    gpu1_vram_mb REAL,
    status TEXT             -- 'ok', 'oom', 'timeout', 'error'
);
```

**Utile pour** :
- Voir les tendances (tok/s moyen sur 24h, heure de pointe)
- Identifier les requêtes problématiques (OOM, timeout)
- Dashboard "dernières 24h" sans Prometheus externe

---

## Idée M4 ★★☆ — Alerte Telegram/Discord

```
⚠️ VRAMancer Alert — 2026-06-15 14:32
GPU0 (5070 Ti) : 15.2/16.0 GB (95%)
Queue : 8 requêtes en attente
Action : évacuation KV vers GPU1 en cours
```

**Simple** : un webhook. L'utilisateur configure `VRM_ALERT_WEBHOOK=https://...`
et reçoit les alertes définies dans `alerting_rules.yml`.

---

## Idée M5 ★☆☆ — Mode kiosque / affichage permanent

Un écran dédié (vieux iPad, Raspberry Pi avec écran) qui affiche le dashboard
en plein écran, rafraîchissement auto, mode sombre. Pour les homelabs.

---

## Idée M6 ★☆☆ — Benchmark continu hebdomadaire

Toutes les semaines (cron), VRAMancer lance un benchmark standardisé et log le
résultat. Permet de détecter :
- Dégradation de performance (driver update foireuse)
- Problème matériel progressif (VRAM ECC errors)
- Amélioration après upgrade

```bash
# cron hebdomadaire
0 3 * * 0 vramancer bench --quick >> ~/.vramancer/bench_history.jsonl
```

---

# PARTIE 4 — Priorités monitoring

| Priorité | Chantier | Effort | Impact |
|---|---|---|---|
| **1** | Dashboard one-liner (`/dash` dans `vramancer serve`) | 1-2 sessions | Énorme — première impression |
| **2** | `vramancer doctor` | 1 session | Support, onboarding |
| **3** | Playground web intégré | 2 sessions | UX, démo |
| **4** | Historique SQLite local | 1 session | Debug, tendances |
| **5** | Alertes webhook (Telegram/Discord) | 0.5 session | Monitoring passif |
| **6** | Mode kiosque | 0.5 session | Homelab |
| **7** | Benchmark continu | 0.5 session | Confiance long terme |

---

— DeepSeek

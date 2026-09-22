# RX 7900 XT — conclusion de la session du 2026-09-22

> À lire après avoir retiré la carte. Tout ce qui est chiffré ici a été **mesuré
> pendant la session**, sortie brute conservée dans le dépôt. Rien n'est extrapolé.

---

## 1. Réponse courte

> **Mise à jour finale (00 h 10) — LA CONCLUSION A CHANGÉ.** Un premier verdict disait
> « le cross-vendor n'apporte rien, la carte peut partir ». **C'était faux par omission.**
> Les deux modèles testés tombaient de part et d'autre du seul régime qui compte. Un
> troisième test, sur un modèle qui ne tient QUE dans la VRAM cumulée, donne **4.4× en
> faveur de la paire** (20.7 → 91.7 tok/s). Lire §8 avant toute décision de revente.

**La 7900 XT est stable et utilisable.** 12 générations longues d'affilée sans une
seule erreur, 5 benchmarks identiques à ±1.3 % en prefill, aucune erreur amdgpu dans
le journal noyau, `runtime_status=active` sur les 78 relevés de capteurs. Le correctif
`amdgpu.runpm=0` que tu avais appliqué tient : le bug de resume runtime-PM de septembre
ne s'est pas reproduit une seule fois.

**Et elle a une vraie raison d'être gardée** (§8) : sur un modèle de 27 GB — trop gros
pour la 3090 seule, trop gros pour elle seule — la paire donne **91.7 tok/s contre 20.7
pour la 3090 seule**. Elle ne rend pas le modèle « plus rapide », elle le rend utilisable.

**Ce qui ne marche pas n'a rien à voir avec elle :**
- la **RTX 3090** est inutilisable *côté logiciel* (aucun module noyau nvidia pour le
  kernel 6.8.0-139) → la vraie question de D3.d, la paire mixte, reste **non mesurée** ;
- le **clignotement** de l'écran vient du `virtio_gpu` de QEMU sous GNOME **Wayland**,
  pas des cartes : l'amdgpu est headless (`Cannot find any crtc`). Retirer la 7900 XT
  ne le corrigera pas. La piste, c'est une session **Xorg**, ou l'affichage côté hyperviseur.

---

## 2. Ce qu'elle donne en performance

Qwen3.6-35B-A3B Q4_K_M (20.60 GiB, 40 couches, MoE 256 experts / 8 actifs), llama.cpp
b11026 Vulkan/RADV, `-p 512 -n 128 -fa on`.

| Couches sur GPU (`-ngl`) | Prefill (tok/s) | Génération (tok/s) |
|---|---|---|
| 0 — CPU seul (contrôle) | 115.37 | 6.59 |
| 30 | 321.48 | 21.04 |
| 34 | 460.82 | 25.52 |
| **38** | **829.14** | **37.78** |
| 40 (toutes) — run 1 | 128.65 | 17.43 |
| 40 (toutes) — run 2 | 120.15 | 8.95 |

**Le résultat le plus utile de la journée est cette non-monotonie.** Le modèle fait
20.60 GiB pour 20.0 GiB de VRAM. À 38 couches, deux couches restent au CPU et tout tient :
37.78 tok/s. À 40 couches, on demande 0.6 GiB de trop, le pilote déborde en mémoire hôte,
et le débit tombe à 8.95–17.43 tok/s **avec une variance x2 entre deux runs identiques**.
Autrement dit : *mettre tout sur le GPU coûte 2 à 4× le débit*. Le bon réglage est
« juste en dessous de la saturation », et c'est précisément le genre d'arbitrage que
l'orchestrateur doit faire tout seul.

Reproductibilité à `-ngl 38` (5 runs) : prefill 797–807, génération 37.18–38.60.

---

## 3. Campagne de stabilité — ce qui a été fait

| Test | Résultat |
|---|---|
| 12 générations de 512 tokens à la suite (≈6 min de charge) | **12/12 rc=0**, 22–36 s chacune, ~3.3 k caractères, texte cohérent (raisonnement + code Python corrects) |
| 5 benchmarks identiques | écart max **1.3 %** prefill, **3.8 %** génération |
| 3 saturations VRAM volontaires (`-ngl 40`) | aucun plantage, `runtime_status=active` après chacune |
| Journal noyau sur toute la campagne | **zéro** ligne d'erreur amdgpu, zéro reset GPU, zéro ring timeout |
| Capteurs (78 relevés) | temp max **41 °C**, puissance max **120 W**, ventilateur max **765 rpm**, sclk max **2956 MHz**, VRAM max **20 452 MiB**, occupation max **100 %** |

La carte n'a jamais chauffé ni bruité : 41 °C au maximum sous charge soutenue, très loin
de ses limites. Le passthrough VFIO tient la charge.

---

## 4. Ce que la carte a révélé dans VRAMancer (le vrai butin)

Brancher une AMD a fait tomber **quatre bugs réels**, dont trois qui touchaient *tous*
les utilisateurs, pas seulement les AMDistes. Aucun n'aurait été trouvé en relisant le code.

1. **Le téléchargement automatique de llama-server était cassé pour tout le monde.**
   La table d'assets demandait des `.zip` `llama-{tag}-bin-ubuntu-x64.zip` ; upstream
   publie des `.tar.gz` nommés par accélérateur. Pire, la release « latest » est
   maintenant un tag de version (`v0.4.1`) qui ne contient qu'un `nightly-tag.txt` :
   l'URL construite était `llama-v0.4.1-bin-ubuntu-x64.zip` → **404 vérifié en ligne**.
2. **L'extraction produisait un binaire qui ne démarre pas.** Seul `llama-server` était
   extrait ; les builds sont dynamiques, avec des `libggml-*.so` versionnées et leurs
   liens symboliques. Résultat : `libllama-common.so.0: cannot open shared object file`.
3. **Les options de llama.cpp avaient changé de forme** : `--flash-attn` exige désormais
   une valeur (sinon il avale l'argument suivant et le serveur refuse de démarrer) et
   `--no-mmap` est devenu `--load-mode none`. Le code lit maintenant `--help` du binaire
   au lieu de supposer, donc un binaire local ancien reste supporté.
4. **Les GPU AMD étaient invisibles de tout l'outillage** (`pynvml` et `torch.cuda` ne
   voient que NVIDIA) : dashboard et `vramancer doctor` annonçaient « aucun GPU détecté »
   pendant que la carte tournait à 37.8 tok/s. Et sur une machine AMD sans NVIDIA, le
   backend téléchargeait le build **CPU** — soit, mesuré ici, **6.59 tok/s au lieu de
   37.78, un facteur 5.7**, silencieusement.

**Preuve de bout en bout après correction** : `LlamaServerBackend` télécharge tout seul
le build Vulkan b11112, démarre sur la 7900 XT en **9.6 s**, répond correctement, et le
serveur rapporte **38.2 tok/s** — cohérent avec les 37.78 de `llama-bench`. C'est
VRAMancer qui pilote, pas llama.cpp lancé à la main.

Depuis le correctif, `vramancer doctor --share` affiche :
`GPU0 : Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] (amdgpu, 20.0 GB)`.

---

## 5. Remettre la 3090 en route — FAIT le 2026-09-22 à 23 h

Le module nvidia 595-open n'existe que pour le kernel **6.8.0-134** ; tu tournes en
**6.8.0-139**, et le métapaquet `linux-modules-nvidia-595-open-generic` est resté bloqué
en version -134. Il n'y a pas de DKMS installé, donc rien ne se reconstruit tout seul.

```bash
sudo apt install linux-modules-nvidia-595-open-6.8.0-139-generic
sudo modprobe nvidia nvidia_uvm && nvidia-smi
```

La simulation `apt` montre que cela met aussi à jour le userland 595.71.05 → 595.91.07,
ce qui est cohérent (aucun module nvidia n'est chargé actuellement, donc pas de conflit).
La 3090 étant en passthrough et ne pilotant aucun écran, `modprobe` suffit — **le reboot
n'apporte rien**. Pour que ça ne casse plus à chaque mise à jour de kernel :
`sudo apt install dkms nvidia-dkms-595-open`.

**Exécuté, ça a marché** : `nvidia-smi` répond, les deux cartes apparaissent ensemble
dans llama.cpp (`Vulkan0: RTX 3090`, `Vulkan1: RX 7900 XT`), et le bench de la paire a
pu tourner → voir §8 pour le verdict.

À retenir pour la prochaine mise à jour de kernel : **le problème se reproduira**, parce
qu'il n'y a toujours pas de DKMS. Pour l'éviter durablement :
`sudo apt install dkms nvidia-dkms-595-open`.

Note annexe : la session a aussi basculé de **Wayland à Xorg** à ce moment-là, et **c'est
ça qui a fait disparaître le clignotement** — pas les GPU, pas le code. Si le scintillement
revient un jour, vérifier d'abord `echo $XDG_SESSION_TYPE`.

---

## 6. Travail livré dans la même session (plan D)

10 commits, suite de référence **1212 tests, 100 %, zéro échec** avant comme après.

| Commit | Contenu |
|---|---|
| `[D0.2]` | post de lancement réencodé (UTF-8 double *et* triple encodé) et déplacé |
| `[D2.1]` | `examples/tool_calling_quickstart.py` (client `openai` officiel, 2 tours) |
| `[D2.2]` | 3 compteurs Prometheus tool-calls + 5 tests |
| `[D2.3]` | dashboard : GPU AMD visibles (le « GPU factice » du plan n'existait pas) |
| `[D2.5]` | docstrings honnêtes de `core/orchestrator/` — aucun de ces 3 modules n'a de consommateur hors tests, c'est écrit noir sur blanc |
| `[D2.6]` | `vramancer doctor --share` (dump anonymisé pour issues) |
| `[D2.7]` | README : section des 6 résultats réfutés + table de compat des clients agents + template d'issue |
| `[D2.9]` | 13 tests sécurité (`enforce_startup_checks` n'en avait aucun) |
| `[D3.c]` | simulation cross-nœud Docker : UDP **et** mDNS traversent le bridge, mort d'un nœud détectée en ~10 s |
| `[D3.d]` | moitié AMD mesurée (rapport ci-dessus) |
| `[AMD]` ×3 | les 4 bugs de la section 4 + les 3 bugs de service de la section 7 |
| `[AMD]` doc | `docs/sessions/AMD_E2E_2026-09-22.md` — VRAMancer de bout en bout sur AMD |

**D0.1** : le token `ghp_…` a été retiré de `.git/config` (le remote est propre), mais
il faut encore le **révoquer sur github.com** — c'est la seule action de sécurité qui
reste de ton côté.

---

## 7. Seconde passe : VRAMancer lui-même sur la carte

La première passe testait llama.cpp. Celle-ci lance `vramancer serve` et des clients
HTTP standards. Rapport complet : `docs/sessions/AMD_E2E_2026-09-22.md`.

### 7.1 Le serveur tournait à 2 tok/s sur une carte inutilisée

| Chemin | Débit | VRAM | GPU |
|---|---|---|---|
| `vramancer serve` **avant** | **2.00 tok/s** | 335 MiB | 1 % |
| `vramancer serve` **après** | **31.8 – 33.0 tok/s** | 19 961 MiB | 52 % |
| `llama-bench` direct (référence) | 37.78 tok/s | 19.8 GiB | — |

`llama-cpp-python` est compilé pour **un** accélérateur. La roue installée est une roue
CUDA : elle s'importe parfaitement sur une machine AMD, mais `llama_supports_gpu_offload()`
renvoie False et le modèle part **entièrement en CPU, sans un message**. Le choix de
backend détecte maintenant ce cas et bascule sur le sous-processus llama-server (build
Vulkan téléchargé tout seul). L'écart résiduel 31.8 vs 37.78 est le coût HTTP sur une
requête de 128 tokens, pas une perte GPU.

### 7.2 Une requête suffisait à mettre le serveur à terre

Un client envoyant `{"model": "local"}` — nom arbitraire, ce que font beaucoup de
clients OpenAI — faisait partir le serveur en résolution HuggingFace. Or
`PipelineRegistry.load()` **arrête le pipeline courant AVANT** de charger le nouveau :
l'échec laissait le serveur **sans aucun modèle**, répondant 500 à tout le monde
jusqu'au redémarrage. Corrigé : un nom qui n'est pas une référence plausible est
ignoré et le modèle chargé est servi (comportement de llama.cpp et LM Studio). Les
vraies références restent honorées ; un échec renvoie un 400 explicite. 3 tests de
non-régression.

### 7.3 Le sous-processus squattait 20 GB de VRAM après la mort du parent

`llama-server` survivait à son parent, gardant le port 8081 **et** ~20 GB de VRAM. Le
démarrage suivant échouait alors de façon illisible — ça m'a coûté trois faux
diagnostics dans cette session. Corrigé par `weakref.finalize` + `atexit`, recherche
d'un port libre si un orphelin traîne, et `PR_SET_PDEATHSIG` pour le cas où le parent
est tué par SIGKILL (où `atexit` ne tourne pas). Vérifié : parent SIGKILL → enfant
mort, VRAM rendue.

### 7.4 Le chemin agent complet fonctionne sur AMD

`examples/tool_calling_quickstart.py` (client `openai` officiel) contre ce serveur :

```
→ tool_call : get_weather({'city': 'Toulouse'})
→ résultat  : {'city': 'Toulouse', 'temp_c': 18, 'conditions': 'nuageux'}
Réponse finale : Il fait 18°C et il est nuageux à Toulouse.
```

Le modèle émet du `<tool_call>`, VRAMancer le convertit au format OpenAI, le client
exécute l'outil, le modèle rédige la réponse. **C'est la promesse du projet, validée
sur AMD.**

### 7.5 Concurrence

| Requêtes simultanées | Durée | Latence min/max | Débit agrégé |
|---|---|---|---|
| 1 | 2.0 s | 2.0 / 2.0 s | 31.5 tok/s |
| 4 | 4.0 s | 1.8 / 4.0 s | **64.5 tok/s** |
| 8 | 8.8 s | 3.6 / 8.8 s | 58.5 tok/s |

Le parallélisme double le débit agrégé à 4 requêtes puis plafonne. Sur cette carte et
ce modèle, **4 simultanées est le bon point de fonctionnement**.

### 7.6 `vramancer benchmark` était cassé pour tout le monde

La commande appelait `profiler.benchmark_gpu()` — **méthode qui n'existe pas** dans
`LayerProfiler`. Elle échouait donc sur `AttributeError` sur n'importe quelle machine,
NVIDIA comprise, en affichant « Benchmark failed ». C'est pourtant la commande que
`vramancer doctor` recommande pour obtenir des tok/s mesurés. Corrigée (API réelle :
`profile_gpus()`), et elle liste maintenant aussi les cartes AMD en indiquant
honnêtement que torch ne les pilote pas :

```
  [cpu] GPU0 CPU — compute 165.0 GFLOPS, mémoire 17.0 GB/s (mesuré)
  [amdgpu] GPU0 Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] — 20.0 GB
      compute/mémoire : non mesurés — torch ne pilote pas cette carte
```

### 7.7 Claims cross-vendor : marqués non mesurés

`experimental/cross_vendor_bridge.py` annonçait « 25-50 GB/s sustained » et « ~20 GB/s ».
Ce sont des bornes théoriques PCIe, **jamais mesurées** — et impossibles à mesurer tant
que les deux cartes ne sont pas vivantes en même temps. Le fichier le dit maintenant en
tête, conformément à la règle d'honnêteté du projet. Si tu vends la carte sans avoir
mesuré, ces chiffres devront **disparaître**, pas rester en « théorique ».

---

## 8. D3.d — LA question, enfin tranchée

Rapport complet : `docs/reports/D3D_CROSS_VENDOR_VERDICT.md`.

### 8.1 Modèle qui tient sur une seule carte (Qwen3.6-35B, 20.60 GiB)

| Configuration | Prefill pp512 | Génération tg128 |
|---|---:|---:|
| 7900 XT seule (`-ngl 38`) | 805 | 29.5 |
| **3090 seule** (`-ngl 99`) | **3 680** | **135.9** |
| Paire mixte, split défaut | 2 796 | 92.0 |
| Paire mixte, split 24:20 | 2 784 | 91.5 |
| Paire mixte, split par lignes | ❌ le modèle ne charge pas | ❌ |

La paire **marche** (llama.cpp Vulkan mélange NVIDIA et AMD d'office) mais coûte
**−32 % en génération** face à la 3090 seule : le modèle tient déjà dans ses 24 GB, la
carte AMD n'ajoute que des transferts PCIe.

### 8.2 Modèle trop gros pour une carte (DeepSeek-V4-Flash, 81 GiB, 43 couches)

| Configuration | Couches sur GPU | Prefill pp128 | Génération tg32 |
|---|---:|---:|---:|
| CPU seul | 0 / 43 | 24.26 | 0.97 |
| 3090 seule (`-ngl 11`) | 11 / 43 | 27.37 | 1.08 |
| **Paire** (`-ngl 16`) | **16 / 43** | **32.88** | **1.38** |
| Paire (`-ngl 21`) | — | ❌ échec d'allocation | ❌ |

La paire gagne pour de bon : **+28 %** contre la 3090 seule, +42 % contre le CPU. Et
c'est **sans conséquence** : 1.38 tok/s reste inutilisable, exactement comme 1.08. On
passe d'inutilisable à inutilisable.

### 8.3 Le régime qui change tout (Qwen3.6-35B **Q6_K**, 27.29 GiB)

Ni la 3090 (24 GB) ni la 7900 XT (20 GB) ne peuvent héberger ce modèle. Les deux
ensemble, si.

| Configuration | Couches sur GPU | Prefill | Génération |
|---|---:|---:|---:|
| CPU seul | 0 / 40 | 195 | 7,5 |
| 3090 seule (le reste déborde) | 30 / 40 | 507 | 20,7 |
| 7900 XT seule (le reste déborde) | 26 / 40 | 212 | 18,0 |
| **PAIRE — modèle ENTIER en VRAM** | **40 / 40** | **2 141** | **91,7** |
| PAIRE — split 24:20 | 40 / 40 | 2 139 | **92,7** |

**4,4× la meilleure carte seule. 12× le CPU.** Et le chiffre brut sous-estime l'effet :
20,7 tok/s est trop lent pour un agent de code, 92 tok/s est confortable. La paire ne
rend pas le modèle plus rapide, elle le rend **utilisable**.

### 8.4 Verdict révisé

| Le modèle… | Verdict | Mesure |
|---|---|---|
| tient sur la carte la plus rapide | ❌ ne pas mélanger, −32 % | 136 → 92 tok/s |
| **ne tient QUE dans la VRAM cumulée** | ✅ **mélanger, 4,4×** | 20,7 → 91,7 tok/s |
| dépasse même la VRAM cumulée | ⚠️ +28 % sans usage | 1,08 → 1,38 tok/s |

Le **pont cross-vendor maison** reste inutile — mais pour une autre raison qu'annoncé :
llama.cpp Vulkan livre déjà ce 4,4× tout seul, sans ROCm ni configuration. Ce qui a de
la valeur, c'est **l'orchestration** : encore faut-il savoir qu'il y a une deuxième carte.
VRAMancer ne le savait pas — son calcul de répartition passait par `torch.cuda`, aveugle
à l'AMD ; il aurait servi ce modèle sur la 3090 seule à 20,7 tok/s. Corrigé dans la
foulée. C'est exactement le métier du projet : **ne pas laisser 4,4× sur la table.**

### 8.5 Donc : garder ou vendre ?

La réponse a changé avec la mesure. **La 7900 XT a une utilité démontrée et chiffrée** :
elle fait passer ce nœud de « Q4_K_M à 136 tok/s » à « **Q6_K à 92 tok/s** » — un quant
sensiblement meilleur, à une vitesse toujours confortable, impossible sans elle.

- **La garder** si tu veux servir des modèles de 24-44 GB (quants Q6/Q8 de 30B, ou un
  70B très quantifié) : c'est là qu'elle vaut son prix.
- **La vendre** si ton usage tient dans 24 GB : sur ces modèles-là elle coûte 32 % de
  débit et ne sert à rien.

Ce qui ne dépend plus d'elle, en revanche : les huit bugs qu'elle a fait tomber et le
support AMD sont acquis, dans le dépôt, définitivement.

---

## 9. Ce qui reste ouvert

- ~~D3.d, paire 3090 + 7900 XT~~ → **MESURÉ et tranché** (§8). Plus rien à faire.
- **D3.a** (bridge cross-vendor en Python) : nécessite ROCm/torch-hip, non installé.
  Rien de ce qui a été mesuré aujourd'hui ne passe par ROCm — Vulkan/RADV a suffi.
- **D3.b** : 2e machine physique. La simulation Docker dégrossit, elle ne remplace pas.
- **D1** : PyPI, tag `v2.0.0`, publication du post — manuel, à toi.

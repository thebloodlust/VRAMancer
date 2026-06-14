# DeepSeek — Peut-on vraiment bypasser le P2P NVIDIA ?

> Questions de Jérémie : Gemini propose de "planquer les paquets aller-retour via Tokio"
> pour bypasser la limitation P2P. Est-ce qu'on peut éviter le CPU-staged ? Gain ?

---

## La réponse courte

**Non, on ne peut pas bypasser le blocage P2P de NVIDIA par logiciel.**

Mais c'est pas grave. Le CPU-staged à 25 GB/s est déjà à **78% du plafond PCIe**.
Même avec un P2P magique, le gain max serait de +25%. Pas ×2, pas ×10. Juste +25%.

---

## Pourquoi P2P est bloqué (rappel)

```
GPU0 (3090, Ampere) ─── PCIe 4.0 x16 ─── GPU1 (5070 Ti, Blackwell)
                         ↑
                    Le bus PCIe EXISTE.
                    Les deux GPUs SONT sur le même bus.
                    Le hardware PEUT faire du DMA direct.

                    MAIS NVIDIA a mis un CHECK dans le driver :
                    if (is_consumer_gpu() && !has_nvlink()) return ACCESS_DENIED;

                    C'est UNIQUEMENT logiciel. Le silicium peut le faire.
```

NVIDIA réserve le P2P aux GPUs pro (A100, H100) pour justifier leur prix ×10.
C'est la même puce GA102 dans la 3090 et la A6000 — l'une a P2P, l'autre non.

---

## Les 6 approches possibles (et pourquoi elles échouent)

### 1. Tokio "planquer les paquets" (Gemini) ❌

```
GPU0 → [TCP localhost via Tokio] → GPU1
       ↑
       GPU0 → CPU (DMA) → kernel TCP buffer → CPU → GPU1 (DMA)
       4 copies. Pire que le CPU-staged (2 copies).
```

Tokio rend ça asynchrone et "joli", mais le chemin physique reste GPU0→CPU→GPU1.
Aucune différence avec le CPU-staged. **Aucun gain.**

### 2. CUDA IPC Lazy Peer Access ⚠️ (à tester, faible espoir)

```python
# GPU0 exporte un handle mémoire
handle = cudaIpcGetMemHandle(gpu0_tensor.data_ptr())

# GPU1 l'importe avec lazy peer access
ptr = cudaIpcOpenMemHandle(handle, cudaIpcMemLazyEnablePeerAccess)
# → Si ça marche, GPU1 peut LIRE la mémoire de GPU0 sans CPU
```

`cudaIpcMemLazyEnablePeerAccess` essaie d'activer P2P au moment de l'import,
pas au moment du check initial. Il utilise un chemin kernel différent de
`cudaDeviceEnablePeerAccess`. **Parfois ça passe quand l'autre échoue.**

**Probabilité de succès** : ~5-10%. Faible, mais testable en 5 minutes.

### 3. GPUDirect RDMA via softiwarp (rxe) ❌

```
GPU0 → [GPUDirect] → NIC virtuelle (rxe) → [GPUDirect] → GPU1
```

GPUDirect permet au NIC de DMA directement depuis/vers la VRAM GPU, sans
passer par le CPU. MAIS :
- Il faut `nvidia_peermem` chargé (OK)
- Il faut un vrai NIC RDMA (ConnectX) ou softiwarp (rxe)
- Le NIC doit être sur le MÊME root complex PCIe que les deux GPUs
- Même avec ça, le transfert passe par le NIC → PCIe → deuxième GPU
  = chemin plus long que le P2P direct

Sur une machine mono-nœud, le "réseau local RDMA" est plus lent que le
CPU-staged parce qu'il ajoute un saut PCIe supplémentaire (via le NIC).

### 4. PCIe ACS Override (bare-metal Linux) ⚠️

```bash
# /etc/default/grub
GRUB_CMDLINE_LINUX="... pcie_acs_override=downstream,multifunction"
```

Désactive l'isolation PCIe ACS. Les devices peuvent se parler directement.
Sur certaines cartes mères, ça DÉBLOQUE le P2P même sur GPUs consumer.

**MAIS** :
- Nécessite bare-metal (pas de VM)
- Peut causer des instabilités (les règles ACS existent pour une raison)
- NVIDIA peut toujours bloquer côté driver (ils checkent le modèle du GPU,
  pas juste la topologie PCIe)
- Certaines cartes mères ignorent ce paramètre

**Probabilité de succès** : ~20-30%. Ça dépend de la carte mère et du driver.

### 5. DMA engine du chipset (Intel I/OAT / AMD PTDMA) ❌

Les CPU modernes ont des DMA engines intégrés qui peuvent faire des transferts
mémoire→mémoire sans CPU. Si on pouvait programmer le DMA engine pour copier
de GPU0_BAR vers GPU1_BAR → vrai DMA GPU→GPU sans P2P.

**Problème** : Le DMA engine ne peut pas adresser les BARs GPU sans support
IOMMU explicite. Et même avec, c'est exactement ce que `cudaMemcpyPeer` fait
en interne. Si NVIDIA le bloque dans le driver, le DMA engine ne contourne pas
le blocage — il utilise le même chemin kernel.

### 6. Kernel module custom (triche) ❌

Écrire un module kernel qui reprogramme l'IOMMU pour créer une fenêtre directe
entre les deux GPUs. Techniquement faisable. Mais :
- C'est un projet de plusieurs semaines
- Ça casse à chaque mise à jour du driver NVIDIA
- C'est fragile et dangereux (corruption de mémoire)
- NVIDIA pourrait le détecter et blacklister

---

## Le vrai gain potentiel (même si on réussissait)

```
CPU-staged actuel : 25.3 GB/s (78% du plafond PCIe 4.0 x16)
P2P direct idéal   : 31.5 GB/s (100% du plafond PCIe 4.0 x16)

Gain max : +25% (31.5/25.3)
```

**Même si on trouvait un bypass magique, le gain est borné par PCIe.**
On est déjà à 78% du plafond. Le dernier 22% vaut pas des semaines de
reverse engineering du driver NVIDIA.

---

## Ce qui vaut VRAIMENT le coup

### À tester maintenant (5 minutes)

```python
# Test CUDA IPC Lazy — la seule approche qui a une chance >0%
import ctypes
import torch

# GPU0 : allouer et exporter
t0 = torch.randn(1024, device="cuda:0")
handle = torch.cuda.cudaIpcGetMemHandle(t0.data_ptr())

# GPU1 : importer avec lazy P2P
# (utilise cudaIpcMemLazyEnablePeerAccess en interne)
try:
    t1_ptr = torch.cuda.cudaIpcOpenMemHandle(
        handle, device=1, flags=1  # 1 = cudaIpcMemLazyEnablePeerAccess
    )
    print("✅ CUDA IPC Lazy a marché ! P2P actif via IPC !")
except:
    print("❌ CUDA IPC Lazy échoue aussi. P2P vraiment bloqué.")
```

Si ce test passe → on a un vrai bypass. Si ça échoue → on arrête de chercher.

### La seule approche qui MARCHERAIT vraiment

**Acheter un GPU professionnel avec NVLink.** Une A6000 (48 GB, Ampere, NVLink)
coûte ~3000-4000€ d'occasion. Avec NVLink, P2P = 600 GB/s. C'est 20× plus
rapide que le CPU-staged. Mais c'est pas le même budget.

**Alternative** : attendre les prochaines générations de GPUs. NVIDIA pourrait
débrider le P2P sur les RTX 60 series (2027-2028) si la pression concurrentielle
(AMD, Intel) les y force.

---

## Conclusion honnête

Le CPU-staged à 25 GB/s n'est PAS le problème. À 78% du plafond PCIe, on est
déjà très proche de l'optimum physique. Chercher un bypass P2P pour gratter
+25% alors que le vrai goulot est ailleurs (load-balancing MoE, scheduling,
batcher le décode) c'est optimiser la mauvaise chose.

Le vrai gain des prochaines sessions est dans :
- **Continuous batching** (×7.5 déjà mesuré)
- **Prompt-lookup +500%** (déjà prouvé)
- **S1/S2 packaging** (déjà livré)
- **Single-binary** (adoption)

Pas dans une guerre de tranchées contre le driver NVIDIA.

— DeepSeek

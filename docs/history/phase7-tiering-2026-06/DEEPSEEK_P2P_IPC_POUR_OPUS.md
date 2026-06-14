# DeepSeek → Opus : Bypass P2P, Fake ID, Test IPC

> Regroupe : analyse bypass P2P, question fake ID GPU, test CUDA IPC Lazy.
> Date : 2026-06-14.

---

## 1. Fake ID GPU : NON. Voici pourquoi.

Jérémie demande : peut-on faire croire au driver NVIDIA que les cartes consumer
sont des cartes entreprise pour débloquer le P2P ?

**Techniquement possible ?** Oui. Le driver vérifie le `pci_device_id` dans la
config space PCIe du GPU. On pourrait intercepter cette lecture.

**Approches théoriques :**

| Approche | Faisabilité | Risque |
|---|---|---|
| **Patcher le driver NVIDIA** (modifier le binaire `nvidia.ko` pour skipper le check `is_consumer_gpu()`) | ~2-3 jours de reverse engineering | Casse à chaque mise à jour. Signé par NVIDIA → peut refuser de charger. |
| **Module kernel intercepteur** (s'intercale entre le driver NVIDIA et le PCIe, modifie les lectures de config space) | ~2-4 semaines | Extrêmement fragile. Casse au moindre changement kernel. |
| **vfio-pci + QEMU** (passer les GPUs en PCIe passthrough, QEMU modifie les device IDs présentés au driver invité) | ~1 semaine | Double virtualisation = perte de performance. Complexité énorme. |
| **Flasher le firmware GPU** (modifier le BIOS de la carte pour changer le device ID) | BRIQUE LE GPU | Irréversible si raté. Anti-tampering NVIDIA. |

**Pourquoi on ne fait PAS ça :**

1. **Gain max = +25%** (passer de 25.3 à 31.5 GB/s). Pas ×2, pas ×10. Le jeu n'en vaut pas la chandelle.
2. **Risque de brick** : flasher un firmware GPU = potentiellement détruire une carte à 500-1500€.
3. **Maintenance impossible** : chaque mise à jour driver/kernel casse le hack.
4. **Légalité douteuse** : contourner une limitation délibérée du driver → violation des termes de licence NVIDIA.
5. **Le vrai gain est ailleurs** : continuous batching ×7.5, prompt-lookup +500%, packaging.

**Verdict : on ne touche PAS à ça.** C'est le genre d'idée qui sonne bien dans une
conversation mais qui devient un cauchemar d'ingénierie. Le +25% de bande passante
ne justifie pas des semaines de reverse engineering fragile.

---

## 2. Ce qu'on peut VRAIMENT tester : CUDA IPC Lazy

Une seule approche crédible et sans risque : `cudaIpcMemLazyEnablePeerAccess`.

### Pourquoi ça pourrait marcher

`cudaDeviceEnablePeerAccess` est vérifié par `can_device_access_peer`. Si ce
dernier retourne False (ce qui est notre cas), le code suppose que P2P est
impossible. MAIS `cudaIpcOpenMemHandle` avec le flag `cudaIpcMemLazyEnablePeerAccess`
utilise un **chemin kernel différent** qui tente d'activer P2P au moment de
l'import du handle, pas au moment du check initial.

Il y a des cas documentés où `can_access_peer=False` mais `IpcOpenMemHandle+Lazy`
réussit quand même — notamment sur des GPUs consumer où le check initial est
conservatif mais le mapping lazy passe parce qu'il utilise les mécanismes de
mémoire partagée CUDA plutôt que le chemin P2P classique.

### Le test (5 minutes)

```python
#!/usr/bin/env python3
"""Test CUDA IPC Lazy — bypass P2P sur GPUs consumer ?"""
import torch
import ctypes
import os

def test_ipc_lazy_p2p(src_gpu=0, dst_gpu=1):
    """
    Teste si cudaIpcMemLazyEnablePeerAccess permet le P2P
    même quand can_device_access_peer retourne False.
    """
    print(f"=== Test CUDA IPC Lazy P2P : GPU{src_gpu} → GPU{dst_gpu} ===")
    
    # 1. Vérifier l'état P2P actuel
    can = torch.cuda.can_device_access_peer(src_gpu, dst_gpu)
    print(f"can_device_access_peer({src_gpu},{dst_gpu}) = {can}")
    
    if can:
        print("✅ P2P déjà disponible. Rien à bypasser.")
        return True
    
    print("P2P bloqué. Tentative bypass via CUDA IPC Lazy...")
    
    # 2. Allouer un tenseur sur GPU source
    with torch.cuda.device(src_gpu):
        src = torch.randn(1024, 1024, device=f"cuda:{src_gpu}")
        src.fill_(42.0)  # valeur reconnaissable
        src_ptr = src.data_ptr()
    
    # 3. Exporter le handle IPC (devrait marcher même sans P2P)
    try:
        handle = torch.cuda.cudaIpcGetMemHandle(src_ptr)
        print(f"✅ IPC handle exporté depuis GPU{src_gpu}")
    except Exception as e:
        print(f"❌ IPC export échoué : {e}")
        return False
    
    # 4. Importer sur GPU destination avec LazyEnablePeerAccess
    # Le flag cudaIpcMemLazyEnablePeerAccess = 0x1
    try:
        with torch.cuda.device(dst_gpu):
            # cudaIpcOpenMemHandle avec flag lazy
            # Utilise ctypes pour passer le flag (PyTorch ne l'expose pas directement)
            cuda = ctypes.CDLL("libcuda.so.1")
            
            # Types
            cuda.cuIpcOpenMemHandle.restype = ctypes.c_int
            cuda.cuIpcOpenMemHandle.argtypes = [
                ctypes.POINTER(ctypes.c_ulonglong),  # pdptr (out)
                ctypes.c_char * 64,                   # handle
                ctypes.c_uint,                        # flags
            ]
            
            dptr = ctypes.c_ulonglong(0)
            flags = 0x1  # cudaIpcMemLazyEnablePeerAccess
            
            ret = cuda.cuIpcOpenMemHandle(
                ctypes.byref(dptr),
                handle,
                flags
            )
            
            if ret == 0 and dptr.value != 0:
                print(f"✅ CUDA IPC Lazy a marché ! Pointeur GPU{dst_gpu} = 0x{dptr.value:x}")
                
                # 5. Vérifier qu'on peut lire depuis GPU1
                # Créer un tenseur depuis le pointeur importé
                import ctypes
                dtype = src.dtype
                shape = src.shape
                
                # Test simple : copier via cudaMemcpy (DtoD devrait marcher si lazy P2P OK)
                dst = torch.empty_like(src, device=f"cuda:{dst_gpu}")
                
                # Copie DtoD via le pointeur IPC
                cuda.cuMemcpyDtoD.restype = ctypes.c_int
                cuda.cuMemcpyDtoD.argtypes = [
                    ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_size_t
                ]
                ret2 = cuda.cuMemcpyDtoD(
                    ctypes.c_ulonglong(dst.data_ptr()),
                    dptr,
                    src.numel() * src.element_size()
                )
                
                if ret2 == 0:
                    # Vérifier la valeur
                    if dst[0, 0].item() == 42.0:
                        print("✅ DONNÉES VÉRIFIÉES ! P2P fonctionnel via CUDA IPC Lazy !")
                        print("   → On peut transférer GPU0→GPU1 sans CPU !")
                        
                        # Nettoyer
                        cuda.cuIpcCloseMemHandle(ctypes.c_ulonglong(dst.data_ptr()))
                        return True
                    else:
                        print(f"⚠️  Copie OK mais données corrompues : {dst[0,0].item()} ≠ 42.0")
                else:
                    print(f"⚠️  cuMemcpyDtoD échoué (ret={ret2}). P2P même en lazy pas dispo.")
                
                cuda.cuIpcCloseMemHandle(dptr)
            else:
                print(f"❌ CUDA IPC Lazy échoué (ret={ret})")
                
    except Exception as e:
        print(f"❌ Exception : {e}")
        import traceback
        traceback.print_exc()
    
    return False


if __name__ == "__main__":
    if torch.cuda.device_count() < 2:
        print("Besoin de 2 GPUs pour le test.")
        exit(1)
    
    result = test_ipc_lazy_p2p(0, 1)
    
    if result:
        print("\n🎉 BYPASS P2P FONCTIONNEL !")
        print("Prochaine étape : benchmark bande passante IPC vs CPU-staged")
    else:
        print("\nℹ️  P2P définitivement indisponible sur ce matériel.")
        print("Le CPU-staged à 25 GB/s reste l'optimum.")
```

### Ce que le test prouve

| Résultat | Signification | Action |
|---|---|---|
| ✅ Données vérifiées | P2P fonctionne via IPC Lazy | Benchmark BW, remplacer CPU-staged par IPC |
| ⚠️ Import OK mais copie échoue | Handle exportable mais P2P pas activable | P2P vraiment bloqué, abandon |
| ❌ Import échoue | Même le handle IPC ne passe pas | P2P définitivement bloqué |

---

## 3. Si le test IPC réussit — qu'est-ce qu'on en fait ?

```python
# Remplacement du CPU-staged par IPC dans GpuPipeline
# Actuel : GPU0 → pinned CPU → GPU1 (2 sauts PCIe, 25 GB/s)
# Nouveau : GPU0 → GPU1 direct via IPC (1 saut PCIe, ~31 GB/s)

class IpcPipeline:
    """Transfert GPU→GPU via CUDA IPC Lazy (bypass P2P NVIDIA)."""
    
    def __init__(self, src_gpu=0, dst_gpu=1):
        self.src = src_gpu
        self.dst = dst_gpu
        # Une seule allocation partagée, réutilisée
        self._setup_ipc()
    
    def _setup_ipc(self):
        """Configure la mémoire partagée IPC une fois."""
        # Allocation GPU source
        with torch.cuda.device(self.src):
            self._src_buf = torch.empty(
                MAX_TRANSFER_BYTES, dtype=torch.uint8, device=f"cuda:{self.src}"
            )
            handle = torch.cuda.cudaIpcGetMemHandle(self._src_buf.data_ptr())
        
        # Import sur GPU destination avec Lazy P2P
        self._dst_ptr = cudaIpcOpenMemHandle(handle, flags=LAZY_P2P)
    
    def transfer(self, src_ptr, size_bytes):
        """Transfert GPU0→GPU1 : copie vers buffer partagé → lecture directe."""
        # GPU0 : copier vers le buffer IPC (local, rapide)
        cuMemcpyDtoD(self._src_buf.data_ptr(), src_ptr, size_bytes)
        # GPU1 : le buffer IPC est déjà mappé → lecture directe
        # Le DMA PCIe fait GPU0_VRAM → GPU1_VRAM sans passer par le CPU
```

---

## 4. Bilan : ce que je recommande à Opus

1. **Lancer le test CUDA IPC Lazy** (5 minutes). Juste le script ci-dessus.
2. **Si ✅** → benchmark BW IPC vs CPU-staged. Si >25 GB/s → remplacer.
3. **Si ❌** → on ferme définitivement la question P2P. Le CPU-staged à 25 GB/s
   est l'optimum. On arrête de chercher des bypass.
4. **Ne PAS explorer** le fake ID GPU. Risque/bénéfice catastrophique.
5. **Ne PAS explorer** le kernel module custom. Même raison.

— DeepSeek

#!/usr/bin/env python3
"""Test autoritaire : existe-t-il un chemin P2P direct GPU0<->GPU1 sur ce matériel ?

DeepSeek propose cudaIpcMemLazyEnablePeerAccess comme « bypass ». Réalité technique :
le flag Lazy ne fait que DIFFÉRER l'appel cuCtxEnablePeerAccess à l'import du handle —
il ne crée PAS un chemin matériel absent. Si le P2P est bloqué pour raison matérielle
(générations différentes + IOMMU), l'enable échoue, lazy ou pas.

Donc on teste la VRAIE question, à la source (driver API) : tenter
cuCtxEnablePeerAccess et lire le code d'erreur EXACT. Pas d'assertion, une mesure.

  ret == 0                              -> P2P direct POSSIBLE -> on benchmarke
  ret == 217 (PEER_ACCESS_UNSUPPORTED)  -> matériel ne supporte pas -> fermé
  autre                                  -> reporter le code

Usage: python benchmarks/test_p2p_ipc_lazy.py
"""
import ctypes
import torch

# Codes CUresult utiles
CUDA_ERR = {
    0: "CUDA_SUCCESS",
    100: "CUDA_ERROR_NO_DEVICE",
    101: "CUDA_ERROR_INVALID_DEVICE",
    208: "CUDA_ERROR_INVALID_VALUE(ctx)",
    217: "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
    704: "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
    705: "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
}


def main():
    if torch.cuda.device_count() < 2:
        print("Besoin de 2 GPU."); return
    # 1) état rapporté par CUDA runtime (via torch)
    print(f"can_device_access_peer(0,1) = {torch.cuda.can_device_access_peer(0,1)}")
    print(f"can_device_access_peer(1,0) = {torch.cuda.can_device_access_peer(1,0)}")

    # Forcer l'init des contextes primaires torch sur les 2 GPU
    a = torch.zeros(1, device="cuda:0"); b = torch.zeros(1, device="cuda:1")
    torch.cuda.synchronize()

    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuInit(0)

    # Récupérer les devices
    dev0 = ctypes.c_int(0); dev1 = ctypes.c_int(0)
    cu.cuDeviceGet(ctypes.byref(dev0), 0)
    cu.cuDeviceGet(ctypes.byref(dev1), 1)

    # cuDeviceCanAccessPeer direct (driver API, source de vérité)
    can01 = ctypes.c_int(-1); can10 = ctypes.c_int(-1)
    cu.cuDeviceCanAccessPeer(ctypes.byref(can01), dev0, dev1)
    cu.cuDeviceCanAccessPeer(ctypes.byref(can10), dev1, dev0)
    print(f"[driver] cuDeviceCanAccessPeer 0->1 = {can01.value} | 1->0 = {can10.value}")

    # Retenir les contextes primaires
    ctx0 = ctypes.c_void_p(); ctx1 = ctypes.c_void_p()
    cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx0), dev0)
    cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx1), dev1)

    # Se placer dans le contexte de GPU0 et tenter d'activer le peer access vers GPU1
    cu.cuCtxSetCurrent(ctx0)
    ret = cu.cuCtxEnablePeerAccess(ctx1, 0)
    name = CUDA_ERR.get(ret, f"code {ret}")
    print(f"\n[TEST] cuCtxEnablePeerAccess(ctx_GPU1) depuis GPU0 -> ret={ret} ({name})")

    if ret in (0, 704):
        print("✅ P2P DIRECT POSSIBLE sur ce matériel ! -> on peut benchmarker IPC/peer vs CPU-staged.")
    elif ret == 217:
        print("❌ PEER_ACCESS_UNSUPPORTED : le matériel n'a aucun chemin P2P direct.")
        print("   -> cudaIpcMemLazyEnablePeerAccess ne peut PAS aider (il appelle ce même enable).")
        print("   -> CPU-staged est obligatoire. Question P2P définitivement fermée.")
    else:
        print(f"⚠️  Code inattendu {ret} ({name}) — à investiguer si on veut creuser.")

    # cleanup best-effort
    try:
        cu.cuDevicePrimaryCtxRelease(dev0); cu.cuDevicePrimaryCtxRelease(dev1)
    except Exception:
        pass


if __name__ == "__main__":
    main()

"""Runtime ROCm 7 pour les builds llama.cpp HIP (fork PrismML sur carte AMD).

Mesuré le 2026-09-24 sur RX 7900 XT, Bonsai 2 27B PQ2_0 : 0.9 tok/s avec le build Vulkan
du fork, **53.5 tok/s avec son build ROCm**. Le build ROCm a besoin des bibliothèques
ROCm 7 (libamdhip64.so.7, rocBLAS, hipBLAS). Si le système ne les a pas, on les installe
SANS sudo depuis les paquets pip officiels d'AMD (« TheRock », ~4 Go) dans
~/.cache/vramancer/rocm.

Reste un prérequis système qu'on ne peut pas contourner : l'accès à /dev/kfd
(groupe `render`). Sans lui, ROCm ne voit aucune carte.
"""
from __future__ import annotations

import glob
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

ROCM_DIR = Path.home() / ".cache" / "vramancer" / "rocm"
THEROCK_INDEX = "https://rocm.nightlies.amd.com/v2/{family}/"
KFD_FIX = "sudo usermod -aG render $USER  (puis se reconnecter)"


def gfx_versions() -> List[int]:
    """Versions gfx des GPU vus par le pilote KFD (ex. 110000 = gfx1100)."""
    out = []
    for f in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties"):
        try:
            for line in open(f):
                if line.startswith("gfx_target_version"):
                    v = int(line.split()[1])
                    if v:
                        out.append(v)
        except (OSError, ValueError):
            pass
    return out


def therock_family(version: int) -> Optional[str]:
    """Famille des paquets pip AMD : RDNA3 (gfx110x) et RDNA4 (gfx120x) seulement."""
    major, minor = version // 10000, (version // 100) % 100
    if (major, minor) in ((11, 0), (12, 0)):
        return f"gfx{major}{minor}X-all"
    return None


def kfd_accessible() -> bool:
    return os.access("/dev/kfd", os.R_OK | os.W_OK)


def rocm_libdirs() -> List[str]:
    """Dossiers contenant libamdhip64.so.7 et ses compagnons (système d'abord)."""
    for d in ("/opt/rocm/lib", "/usr/lib/x86_64-linux-gnu"):
        if glob.glob(f"{d}/libamdhip64.so.7*") and glob.glob(f"{d}/librocblas.so*"):
            return [d]
    core = glob.glob(str(ROCM_DIR / "_rocm_sdk_core" / "lib"))
    libs = glob.glob(str(ROCM_DIR / "_rocm_sdk_libraries_*" / "lib"))
    if core and libs and glob.glob(f"{core[0]}/libamdhip64.so.7*"):
        return core + libs
    return []


def ensure_rocm_runtime() -> List[str]:
    """Dossiers du runtime ROCm 7, installé à la demande ; [] si impossible."""
    dirs = rocm_libdirs()
    if dirs:
        return dirs
    fams = {therock_family(v) for v in gfx_versions()} - {None}
    if len(fams) != 1:
        log.warning("Runtime ROCm : architecture AMD non prise en charge ou mélangée (%s)",
                    [hex(v) for v in gfx_versions()])
        return []
    fam = fams.pop()
    log.warning("Installation du runtime ROCm 7 (%s, ~4 Go, une seule fois) dans %s…",
                fam, ROCM_DIR)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", str(ROCM_DIR),
                        "--index-url", THEROCK_INDEX.format(family=fam), "rocm[libraries]"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("Échec de l'installation ROCm : %s", r.stderr[-300:])
        return []
    return rocm_libdirs()


def rocm_usable() -> bool:
    """Carte AMD RDNA3/RDNA4 utilisable par ROCm (accès /dev/kfd compris)."""
    if not any(therock_family(v) for v in gfx_versions()):
        return False
    if not kfd_accessible():
        log.warning("ROCm : pas d'accès à /dev/kfd — %s. En attendant, build Vulkan "
                    "(bien plus lent pour les modèles ternaires : 0.9 contre 53.5 tok/s).",
                    KFD_FIX)
        return False
    return True


__all__ = ["ensure_rocm_runtime", "rocm_libdirs", "rocm_usable", "therock_family",
           "gfx_versions", "kfd_accessible", "ROCM_DIR"]

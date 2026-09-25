"""`vramancer plan` — placer les poids d'un GGUF sur la hiérarchie mémoire, par la mesure.

Idée d'origine de VRAMancer : classer les mémoires de la plus rapide à la plus lente
(GPU principal, GPU secondaire, RAM, …) et y ranger les poids par ordre d'utilité.
Ce qui a été établi en mesurant (2026-09-22/23, RTX 3090 + RX 7900 XT + EPYC 7402) :

1. **Amener le calcul aux données.** Chaque étage calcule sur ce qu'il détient ; seules
   les activations circulent. (Recopier des poids vers le GPU rapide à chaque token avait
   été réfuté : 61-73 % du débit de référence.)
2. **Le critère d'utilité est l'intensité de lecture**, pas le numéro de couche : ce qui
   est lu à chaque token (attention, normes, expert partagé, sortie) va sur le GPU le plus
   rapide ; les experts routés d'un MoE (~3 % lus par token) peuvent descendre d'étage.
   Mesuré : MoE 27 GB sur une 3090 seule, « experts de 10 couches en RAM » = 39.8 tok/s
   contre 17.1 pour « 10 couches entières en RAM ». DeepSeek-V4-Flash 81 GiB :
   3090 (tout le chaud + 9 couches d'experts) · 7900 XT (10) · RAM (24) = 12.07 tok/s.
3. **On ne devine pas les coûts, on les mesure sur le modèle visé.** Une calibration de la
   machine sur de petits modèles prédit les gros avec 46 % d'erreur médiane (coûts fixes
   non proportionnels, couches MoE bien plus chères que des couches denses). En revanche,
   le temps par token est LINÉAIRE dans la répartition : 2 ou 3 mesures sur le modèle
   lui-même suffisent à prédire toutes les autres à 1-2 % près.

Deux régimes :
- **le modèle tient dans la VRAM cumulée** → répartition des couches entre GPU :
  t = Σ f_d · T_d. Une mesure par GPU (répartitions différentes) donne les T_d ; on
  remplit ensuite le GPU le moins cher jusqu'à sa capacité, etc. ;
- **MoE plus gros que la VRAM cumulée** → tout le « chaud » sur le GPU principal, puis
  les experts couche par couche sur les étages : t = base + Σ_étage n_étage · c_étage.
  Trois mesures donnent les coûts par couche d'experts de chaque étage ; un étage plus
  cher que la RAM n'est pas utilisé (c'est le planificateur qui dit si le 2e GPU vaut
  le coup, pour CE modèle).

Garantie « ne jamais planter » : capacités calculées avec une marge, et toute mesure qui
échoue par manque de mémoire fait reculer d'un cran au lieu d'interrompre le plan.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger("vramancer.planner")

PLAN_CACHE = Path.home() / ".cache" / "vramancer" / "plans.json"
GIB = 2 ** 30
EXP_RE = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight")
_PP_RE = re.compile(r"\|\s*pp\d+[^|]*\|\s*([\d.]+)\s*±")
_TG_RE = re.compile(r"\|\s*tg\d+[^|]*\|\s*([\d.]+)\s*±")


# ── Profil du modèle (en-tête GGUF) ──────────────────────────────────────────

@dataclass
class ModelProfile:
    path: str
    shards: List[str]
    n_layers: int
    total_bytes: int
    expert_bytes_per_layer: Dict[int, int]          # vide pour un modèle dense
    expert_count: int = 0
    expert_used: int = 0

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 0 and bool(self.expert_bytes_per_layer)

    @property
    def hot_bytes(self) -> int:
        """Tout ce qui n'est pas expert routé : à garder sur le GPU principal."""
        return self.total_bytes - sum(self.expert_bytes_per_layer.values())


def _shards(path: str) -> List[str]:
    # Sur la chaîne telle quelle : Path() convertirait les « / » en « \ » sous Windows.
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", str(path))
    if not m:
        return [str(path)]
    n = int(m.group(3))
    return [f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]


def profile_model(path: str) -> ModelProfile:
    """Lit les tailles de tenseurs (tous les fragments d'un GGUF découpé)."""
    # Lecteur d'en-tête maison : le paquet `gguf` lève ValueError sur les types qu'il ne
    # connaît pas (ternaires PrismML) et met 10-15 s à indexer les tenseurs.
    from core.llama_server_backend import _gguf_header
    shards = _shards(path)
    total, per_layer = 0, {}
    n_layers = ne = nu = 0
    for i, sh in enumerate(shards):
        h = _gguf_header(sh, tensors=True)
        if h is None:
            raise ValueError(f"en-tête GGUF illisible : {sh}")
        if i == 0:
            kv = h["kv"]
            arch = kv["general.architecture"]
            n_layers = int(kv[f"{arch}.block_count"])
            ne = int(kv.get(f"{arch}.expert_count", 0))
            nu = int(kv.get(f"{arch}.expert_used_count", 0))
        for name, _t, b in h["tensors"]:
            total += b
            m = EXP_RE.match(name)
            if m:
                layer = int(m.group(1))
                per_layer[layer] = per_layer.get(layer, 0) + b
    return ModelProfile(path=path, shards=shards, n_layers=n_layers, total_bytes=total,
                        expert_bytes_per_layer=per_layer, expert_count=ne, expert_used=nu)


# ── Mesure ───────────────────────────────────────────────────────────────────

@dataclass
class Measure:
    pp: Optional[float]
    tg: Optional[float]

    @property
    def ok(self) -> bool:
        return self.tg is not None


def bench(binary, model: str, args: List[str], env: dict, depth: int = 512,
          n_gen: int = 32, reps: int = 1, timeout: int = 3600) -> Measure:
    """Une passe llama-bench ; Measure(None, None) si la configuration ne tient pas."""
    bench_bin = Path(binary).parent / "llama-bench"
    cmd = [str(bench_bin), "-m", model, "-fa", "on", "-p", "128", "-n", str(n_gen),
           "-d", str(depth), "-r", str(reps), "-o", "md"] + args
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return Measure(None, None)
    out = r.stdout.decode("utf-8", "ignore")
    pp, tg = _PP_RE.search(out), _TG_RE.search(out)
    return Measure(float(pp.group(1)) if pp else None, float(tg.group(1)) if tg else None)


# ── Placement des experts (MoE plus gros que la VRAM cumulée) ────────────────

def _exp_rule(layers: List[int], buffer: str) -> str:
    """Règle -ot envoyant les experts de ces couches vers un tampon (CPU, Vulkan1…)."""
    alt = "|".join(str(i) for i in sorted(layers))
    return rf"blk\.({alt})\.ffn_(gate|up|down)_exps\.weight={buffer}"


def expert_args(n_layers: int, on_primary: int, secondary: List[Tuple[str, int]],
                n_gpus: int) -> List[str]:
    """Arguments llama.cpp : tout le chaud sur le GPU principal, puis les experts des
    DERNIÈRES couches sur le principal, les suivantes sur chaque GPU secondaire, et le
    reste en RAM.

    Piège mesuré : `-sm none` retire les autres GPU du planificateur de llama.cpp, qui
    refuse alors de calculer des poids qu'on y a placés (« buffer that cannot run the
    operation »). On garde donc tous les GPU déclarés avec une répartition quasi nulle
    pour les secondaires (`-ts 99/1`).
    """
    layers = list(range(n_layers))
    top = n_layers - on_primary
    remaining = layers[:top]                    # couches dont les experts ne sont pas sur le principal
    rules = []
    for buffer, k in secondary:
        if k > 0 and remaining:
            take, remaining = remaining[-k:], remaining[:-k]
            rules.append(_exp_rule(take, buffer))
    if remaining:
        rules.append(_exp_rule(remaining, "CPU"))
    args = ["-ngl", "99"]
    if n_gpus > 1:
        args += ["-ts", "/".join(["99"] + ["1"] * (n_gpus - 1))]
    if rules:
        args += ["-ot", ";".join(rules)]
    return args


@dataclass
class Plan:
    model: str
    regime: str
    args: List[str]
    predicted_tg: Optional[float] = None
    measured: Optional[Measure] = None
    detail: Dict = field(default_factory=dict)
    trace: List[str] = field(default_factory=list)


def plan_moe_tiers(prof: ModelProfile, devices: List[dict], binary, env: dict,
                   report: Callable = print, depth: int = 512,
                   reserve_gib: float = 1.5) -> Optional[Plan]:
    """MoE plus gros que la VRAM : chaud sur le GPU principal, experts par étages.

    3 mesures : (1) principal rempli d'experts, reste en RAM ; (2) principal à moitié ;
    (3) principal rempli + GPU secondaire rempli. Modèle linéaire
    t = base + n_principal·c_p + n_secondaire·c_s + n_RAM·c_ram, puis remplissage
    glouton des étages par coût croissant, sous contrainte de capacité.
    """
    L = prof.n_layers
    per_layer = max(prof.expert_bytes_per_layer.values()) / GIB
    gpus = devices  # ordre de llama.cpp ; le 1er est considéré principal (voir plan())
    primary, secondary = gpus[0], gpus[1:]

    def cap_layers(dev, extra_hot_gib=0.0) -> int:
        free = (dev.get("free_mib") or dev["total_mib"]) / 1024
        return max(0, int((free - reserve_gib - extra_hot_gib) // per_layer))

    k_p = min(L, cap_layers(primary, prof.hot_bytes / GIB))
    report(f"  experts : {L} couches × {per_layer:.2f} GiB · chaud {prof.hot_bytes / GIB:.1f} GiB "
           f"sur {primary['name']} + jusqu'à {k_p} couches d'experts")
    n = len(gpus)
    names = [f"Vulkan{i}" for i in range(n)]  # tampons llama.cpp des GPU locaux

    def run(k_primary, sec_counts, label):
        while k_primary >= 0:
            args = expert_args(L, k_primary, list(zip(names[1:], sec_counts)), n)
            m = bench(binary, prof.shards[0], args, env, depth=depth)
            if m.ok:
                report(f"  {label:<46} {m.tg:6.2f} tok/s (prefill {m.pp or 0:.0f})")
                return k_primary, m
            report(f"  {label:<46} ne tient pas → un cran de moins")
            k_primary -= 1                      # garantie : on recule, on ne plante pas
        return 0, Measure(None, None)

    k_p, m1 = run(k_p, [0] * (n - 1), f"principal : {k_p} couches d'experts, reste en RAM")
    if not m1.ok:
        return None
    k_half = k_p // 2
    _, m2 = run(k_half, [0] * (n - 1), f"principal : {k_half} couches d'experts")
    if not m2.ok or k_p == k_half:
        return None
    t1, t2 = 1 / m1.tg, 1 / m2.tg
    # passer (k_p - k_half) couches de la RAM au principal fait gagner t2 - t1
    d_ram_minus_p = (t2 - t1) / (k_p - k_half)          # = c_ram - c_p
    plan_sec, detail = [0] * (n - 1), {"c_ram_minus_primary_ms": round(d_ram_minus_p * 1e3, 3)}
    best_t, best_args = t1, expert_args(L, k_p, [], n)

    for j, dev in enumerate(secondary):
        k_s = min(L - k_p, cap_layers(dev))
        if k_s <= 0:
            continue
        counts = [0] * (n - 1)
        counts[j] = k_s
        _, m3 = run(k_p, counts, f"+ {dev['name'][:22]} : {k_s} couches")
        if not m3.ok:
            continue
        gain_per_layer = (t1 - 1 / m3.tg) / k_s             # = c_ram - c_secondaire
        detail[f"gain_par_couche_{names[j + 1]}_ms"] = round(gain_per_layer * 1e3, 3)
        if gain_per_layer > 0 and 1 / m3.tg < best_t:
            best_t, plan_sec[j] = 1 / m3.tg, k_s
            best_args = expert_args(L, k_p, list(zip(names[1:], plan_sec)), n)
            report(f"    → {dev['name'][:22]} utile : {gain_per_layer * 1e3:.2f} ms gagnés par couche d'experts")
        else:
            report(f"    → {dev['name'][:22]} NON retenu : il coûte plus qu'il ne rapporte "
                   f"({gain_per_layer * 1e3:+.2f} ms par couche) pour ce modèle")

    return Plan(model=prof.path, regime="moe-tiers", args=best_args,
                predicted_tg=1 / best_t, measured=None,
                detail={**detail, "experts_principal": k_p,
                        "experts_secondaires": dict(zip(names[1:], plan_sec)),
                        "experts_RAM": L - k_p - sum(plan_sec)})


# ── Répartition des couches (le modèle tient dans la VRAM cumulée) ───────────

def plan_layer_split(prof: ModelProfile, devices: List[dict], binary, env: dict,
                     report: Callable = print, depth: int = 512,
                     reserve_gib: float = 1.0) -> Optional[Plan]:
    """t = Σ f_d·T_d : autant de mesures que de GPU pour obtenir les T_d, puis remplissage
    du GPU le moins cher jusqu'à sa capacité, et ainsi de suite."""
    import numpy as np
    n = len(devices)
    size = prof.total_bytes / GIB
    caps = [max(0.0, min(0.97, ((d.get("free_mib") or d["total_mib"]) / 1024 - reserve_gib) / size))
            for d in devices]
    total = [d["total_mib"] for d in devices]
    prorata = [t / sum(total) for t in total]
    splits = [prorata]
    for i in range(n - 1):                      # une répartition de plus par GPU supplémentaire
        s = [(1 - caps[i]) * p / (1 - prorata[i]) if j != i else caps[i] for j, p in enumerate(prorata)]
        splits.append(s)
    rows, ys = [], []
    for s in splits:
        m = bench(binary, prof.shards[0], ["-ngl", "99", "-ts", "/".join(f"{x:.4f}" for x in s)],
                  env, depth=depth, n_gen=64)
        label = " / ".join(f"{x * 100:.0f}%" for x in s)
        if not m.ok:
            report(f"  {label:<24} ne tient pas")
            continue
        report(f"  {label:<24} {m.tg:6.2f} tok/s (prefill {m.pp or 0:.0f})")
        rows.append(s)
        ys.append(1 / m.tg)
    if len(rows) < n:
        return None
    T = np.linalg.lstsq(np.array(rows), np.array(ys), rcond=None)[0]
    order = sorted(range(n), key=lambda i: T[i])            # moins cher d'abord
    f, left = [0.0] * n, 1.0
    for i in order:
        f[i] = min(caps[i], left)
        left -= f[i]
    if left > 1e-6:
        return None
    pred = 1 / float(np.dot(f, T))
    report(f"  coût relatif par GPU : " + ", ".join(f"{devices[i]['name'][:18]} {T[i] * 1e3:.2f} ms" for i in range(n)))
    return Plan(model=prof.path, regime="layer-split",
                args=["-ngl", "99", "-ts", "/".join(f"{x:.4f}" for x in f)],
                predicted_tg=pred, detail={"split": [round(x, 4) for x in f]})


# ── Point d'entrée ────────────────────────────────────────────────────────────

def plan(model_path: str, binary, report: Callable = print, verify: bool = True,
         depth: int = 512) -> Optional[Plan]:
    from core.llama_server_backend import backend_devices, _runtime_env
    env = _runtime_env(binary)
    report("Lecture de l'en-tête GGUF…")
    prof = profile_model(model_path)
    devices = [d for d in backend_devices(binary) if not d.get("rpc")]
    vram = sum((d.get("free_mib") or d["total_mib"]) for d in devices) / 1024
    report(f"Modèle {prof.total_bytes / GIB:.1f} GiB, {prof.n_layers} couches"
           + (f", MoE {prof.expert_used}/{prof.expert_count} experts" if prof.is_moe else ", dense")
           + f" · VRAM disponible {vram:.1f} GiB sur {len(devices)} GPU")
    if not devices:
        report("Aucun GPU visible par llama.cpp.")
        return None
    if prof.total_bytes / GIB <= vram - 1.0 * len(devices) and len(devices) > 1:
        report("Régime : le modèle tient dans la VRAM cumulée → répartition des couches")
        p = plan_layer_split(prof, devices, binary, env, report, depth=depth)
    elif prof.is_moe and prof.hot_bytes / GIB < (devices[0].get("free_mib") or devices[0]["total_mib"]) / 1024 - 1.5:
        report("Régime : MoE plus gros que la VRAM → chaud sur le GPU principal, experts par étages")
        p = plan_moe_tiers(prof, devices, binary, env, report, depth=depth)
    else:
        report("Régime non couvert (dense plus gros que la VRAM, ou partie chaude trop grosse) : "
               "utiliser le débordement par couches (-ngl) — à venir.")
        return None
    if p is None:
        report("Aucun placement valide trouvé.")
        return None
    report(f"Placement retenu, débit prédit {p.predicted_tg:.1f} tok/s")
    if verify:
        m = bench(binary, prof.shards[0], p.args, env, depth=depth, n_gen=64, reps=2)
        p.measured = m
        report(f"Vérification : {m.tg:.2f} tok/s mesurés (prédit {p.predicted_tg:.2f}, "
               f"écart {100 * (m.tg / p.predicted_tg - 1):+.1f} %)" if m.ok else "Vérification : ÉCHEC")
    _save(p)
    return p


def _save(p: Plan) -> None:
    try:
        cache = json.loads(PLAN_CACHE.read_text()) if PLAN_CACHE.exists() else {}
    except Exception:
        cache = {}
    key = f"{Path(p.model).name}|{Path(p.model).stat().st_size}"
    cache[key] = {"regime": p.regime, "args": p.args, "predicted_tg": p.predicted_tg,
                  "measured_tg": p.measured.tg if p.measured else None, "detail": p.detail}
    PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PLAN_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def cached_plan_args(model_path: str) -> Optional[List[str]]:
    """Arguments llama.cpp d'un plan déjà calculé pour ce modèle (sinon None)."""
    try:
        cache = json.loads(PLAN_CACHE.read_text())
        return cache[f"{Path(model_path).name}|{Path(model_path).stat().st_size}"]["args"]
    except Exception:
        return None


__all__ = ["plan", "profile_model", "expert_args", "cached_plan_args", "ModelProfile", "Plan"]

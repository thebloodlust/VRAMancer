"""Script release simplifié.

Actions:
 - Lit version core.__version__
 - Génère SBOM (requirements -> cyclonedx-like minimal)
 - Affiche instructions tag git
"""
from __future__ import annotations
import json, subprocess, sys

def build_sbom():
    try:
        import pkg_resources
    except Exception:
        print("pkg_resources indisponible", file=sys.stderr)
        return []
    deps = []
    for d in pkg_resources.working_set:
        deps.append({"name": d.project_name, "version": d.version})
    return deps

def main():
    from core import __version__
    sbom = {"version": __version__, "dependencies": build_sbom()}
    with open("SBOM.json","w") as f:
        json.dump(sbom, f, indent=2)
    print("SBOM.json généré")
    print(f"Tag suggéré: git tag v{__version__} && git push origin v{__version__}")

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Création d'un environnement virtuel reproductible.

Usage :
    python scripts/bootstrap_env.py
"""
from __future__ import annotations
import os, sys, subprocess, venv, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
VENV_DIR = ROOT / ".venv"

def run(cmd: list[str]):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    if not VENV_DIR.exists():
        print("[ENV] Création venv ...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    py = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    print(f"[ENV] Python: {py}")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(py), "-m", "pip", "install", "-r", "requirements.txt"])
    print("[ENV] OK. Activez avec:")
    if os.name == "nt":
        print("   .\\.venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")

if __name__ == "__main__":
    main()
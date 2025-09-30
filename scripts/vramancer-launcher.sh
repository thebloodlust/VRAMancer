#!/bin/bash
# Lanceur universel VRAMancer avec choix d'interface dashboard

set -e

CHOICE="auto"

if [ -z "$1" ]; then
  echo "Choisissez l'interface dashboard :"
  echo "  1) Qt (PyQt5)"
  echo "  2) Tkinter"
  echo "  3) Web (navigateur)"
  echo "  4) CLI (console)"
  echo "  5) Auto (détection automatique)"
  read -p "Votre choix [1-5, défaut: 5] : " REPLY
  case $REPLY in
    1) CHOICE="qt";;
    2) CHOICE="tk";;
    3) CHOICE="web";;
    4) CHOICE="cli";;
    *) CHOICE="auto";;
  esac
else
  CHOICE="$1"
fi

exec python3 -m dashboard.launcher --mode "$CHOICE"

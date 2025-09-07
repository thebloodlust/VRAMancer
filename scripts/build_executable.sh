#!/usr/bin/env bash
set -e

# Nettoyage
rm -rf dist build *.spec

# Build pour Linux
pyinstaller \
  --onefile \
  --name vramancer \
  --add-data "core/:core" \
  --add-data "dashboard/:dashboard" \
  --add-data "data/:data" \
  cli/main.py

# Le binaire se trouve dans dist/vramancer

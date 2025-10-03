#!/usr/bin/env bash
set -euo pipefail
VERSION=$(grep '^version' setup.cfg | head -1 | cut -d'=' -f2 | xargs)
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
  echo "Tag v$VERSION existe déjà" >&2
  exit 1
fi
echo "Création du tag v$VERSION"
git add setup.cfg CHANGELOG.md || true
git commit -m "chore: release v$VERSION" || true
git tag -a "v$VERSION" -m "Release v$VERSION"
echo "Push tags ? (y/N)"; read -r ans
if [[ "$ans" == "y" || "$ans" == "Y" ]]; then
  git push --follow-tags
else
  echo "Tag local créé. Utilisez: git push --follow-tags"
fi

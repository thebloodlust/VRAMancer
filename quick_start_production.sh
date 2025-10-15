#!/bin/bash
# Quick Start Production - VRAMancer v1.1.0

set -e

echo "🚀 VRAMancer - Quick Start Production"
echo "======================================"
echo ""

# Vérifier si déjà configuré
if [ -n "$VRM_AUTH_SECRET" ] && [ ${#VRM_AUTH_SECRET} -ge 32 ]; then
    echo "✅ VRM_AUTH_SECRET déjà défini"
else
    echo "🔐 Génération VRM_AUTH_SECRET..."
    export VRM_AUTH_SECRET=$(openssl rand -hex 32)
    echo "export VRM_AUTH_SECRET='$VRM_AUTH_SECRET'" >> ~/.bashrc
    echo "✅ VRM_AUTH_SECRET généré et sauvegardé dans ~/.bashrc"
fi

# Configuration production
echo "⚙️  Configuration production..."
export VRM_PRODUCTION=1
export VRM_API_DEBUG=0
export VRM_LOG_JSON=1
export VRM_LOG_LEVEL=INFO

echo "✅ Configuration appliquée:"
echo "   VRM_PRODUCTION=1"
echo "   VRM_API_DEBUG=0"
echo "   VRM_LOG_JSON=1"
echo ""

# Validation
echo "🔍 Validation de la configuration..."
if [ -x "./scripts/check_production_ready.sh" ]; then
    ./scripts/check_production_ready.sh
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Configuration validée !"
        echo ""
        echo "🚀 Démarrage de l'API production..."
        echo ""
        python api.py
    else
        echo ""
        echo "❌ Validation échouée. Consultez SECURITY_PRODUCTION.md"
        exit 1
    fi
else
    echo "⚠️  Script de validation non trouvé"
    echo "   Démarrage direct..."
    python api.py
fi

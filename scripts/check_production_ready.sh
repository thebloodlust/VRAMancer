#!/bin/bash
# check_production_ready.sh
# Script de validation de configuration production pour VRAMancer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Compteurs
ERRORS=0
WARNINGS=0
PASSED=0

# Fonction d'affichage
print_header() {
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  VRAMancer - Validation Configuration Production${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
}

print_check() {
    echo -e "\n${BLUE}🔍 $1${NC}"
}

print_pass() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED++))
}

print_fail() {
    echo -e "${RED}❌ $1${NC}"
    ((ERRORS++))
}

print_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNINGS++))
}

# Fonction de vérification des variables d'environnement
check_env_var() {
    local var_name=$1
    local required=$2
    local min_length=${3:-0}
    
    if [ -z "${!var_name}" ]; then
        if [ "$required" == "true" ]; then
            print_fail "$var_name non défini (OBLIGATOIRE)"
            return 1
        else
            print_warn "$var_name non défini (optionnel)"
            return 0
        fi
    fi
    
    local value="${!var_name}"
    local length=${#value}
    
    if [ $min_length -gt 0 ] && [ $length -lt $min_length ]; then
        print_fail "$var_name trop court ($length < $min_length caractères)"
        return 1
    fi
    
    print_pass "$var_name défini correctement"
    return 0
}

# Fonction de vérification de valeur
check_env_value() {
    local var_name=$1
    local expected_value=$2
    local message=$3
    
    if [ "${!var_name}" == "$expected_value" ]; then
        print_fail "$message (${var_name}=${!var_name})"
        return 1
    fi
    
    print_pass "${var_name}=${!var_name} (OK)"
    return 0
}

# Début de la validation
print_header

# =============================================================================
# 1. Vérification des secrets critiques
# =============================================================================
print_check "Vérification des secrets et authentification"

check_env_var "VRM_AUTH_SECRET" "true" 32
check_env_var "VRM_API_TOKEN" "false" 16

# Vérifier que admin par défaut est désactivé
if [ "$VRM_DISABLE_DEFAULT_ADMIN" == "1" ]; then
    print_pass "Compte admin par défaut désactivé"
else
    print_warn "Compte admin par défaut ACTIVÉ - Changez le mot de passe!"
fi

# =============================================================================
# 2. Vérification mode debug et test
# =============================================================================
print_check "Vérification modes debug/test"

check_env_value "VRM_API_DEBUG" "1" "Mode DEBUG activé (doit être 0)"
check_env_value "VRM_TEST_MODE" "1" "Mode TEST activé (doit être 0)"
check_env_value "VRM_TEST_RELAX_SECURITY" "1" "Sécurité relaxée (doit être 0)"

# Vérifier que debug Flask n'est pas activé dans les fichiers
if grep -r "debug=True" "$PROJECT_ROOT"/*.py 2>/dev/null | grep -v "archive/" | grep -v "#" > /dev/null; then
    print_fail "debug=True trouvé dans des fichiers Python"
    grep -r "debug=True" "$PROJECT_ROOT"/*.py 2>/dev/null | grep -v "archive/" | grep -v "#" | head -5
else
    print_pass "Aucun debug=True trouvé dans les fichiers Python"
fi

# =============================================================================
# 3. Vérification sécurité réseau
# =============================================================================
print_check "Vérification sécurité réseau"

# Rate limiting
if [ "$VRM_DISABLE_RATE_LIMIT" == "1" ]; then
    print_warn "Rate limiting DÉSACTIVÉ (acceptable uniquement en dev)"
else
    print_pass "Rate limiting activé"
    
    # Vérifier limite
    RATE_MAX=${VRM_RATE_MAX:-200}
    if [ $RATE_MAX -lt 50 ]; then
        print_warn "VRM_RATE_MAX très bas ($RATE_MAX)"
    elif [ $RATE_MAX -gt 1000 ]; then
        print_warn "VRM_RATE_MAX très élevé ($RATE_MAX)"
    else
        print_pass "VRM_RATE_MAX=$RATE_MAX (raisonnable)"
    fi
fi

# Rotation de secrets
if [ "$VRM_DISABLE_SECRET_ROTATION" == "1" ]; then
    print_warn "Rotation de secrets désactivée"
else
    print_pass "Rotation de secrets activée"
fi

# =============================================================================
# 4. Vérification logging
# =============================================================================
print_check "Vérification configuration logging"

LOG_LEVEL=${VRM_LOG_LEVEL:-INFO}
if [ "$LOG_LEVEL" == "DEBUG" ]; then
    print_warn "LOG_LEVEL=DEBUG (verbose, peut impacter perf)"
elif [ "$LOG_LEVEL" == "ERROR" ]; then
    print_warn "LOG_LEVEL=ERROR (peut manquer des infos importantes)"
else
    print_pass "LOG_LEVEL=$LOG_LEVEL (OK)"
fi

# Vérifier répertoire logs
LOG_DIR=${VRM_LOG_DIR:-logs}
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR" 2>/dev/null || true
fi

if [ -d "$LOG_DIR" ] && [ -w "$LOG_DIR" ]; then
    print_pass "Répertoire logs accessible: $LOG_DIR"
else
    print_fail "Répertoire logs non accessible: $LOG_DIR"
fi

# =============================================================================
# 5. Vérification API
# =============================================================================
print_check "Vérification API"

API_PORT=${VRM_API_PORT:-5030}
API_HOST=${VRM_API_HOST:-0.0.0.0}

print_pass "API configurée sur ${API_HOST}:${API_PORT}"

# Tester si l'API répond (si démarrée)
if command -v curl &> /dev/null; then
    if curl -s --max-time 2 "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
        HEALTH=$(curl -s "http://localhost:${API_PORT}/health" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
        
        if [ "$HEALTH" == "healthy" ]; then
            print_pass "API répond correctement (health: $HEALTH)"
        else
            print_fail "API répond mais health != healthy (health: $HEALTH)"
        fi
    else
        print_warn "API non démarrée ou non accessible (test skippé)"
    fi
else
    print_warn "curl non disponible (test API skippé)"
fi

# =============================================================================
# 6. Vérification fichiers sensibles
# =============================================================================
print_check "Vérification fichiers sensibles"

# Vérifier que .env n'est pas versionné
if [ -f ".env" ]; then
    if git check-ignore .env > /dev/null 2>&1; then
        print_pass ".env ignoré par git"
    else
        print_fail ".env NON ignoré par git (risque de leak)"
    fi
fi

# Vérifier présence .gitignore
if [ -f ".gitignore" ]; then
    if grep -q "\.env" .gitignore; then
        print_pass ".gitignore contient .env"
    else
        print_warn ".gitignore ne contient pas .env"
    fi
fi

# Vérifier qu'il n'y a pas de secrets hardcodés
if grep -r "password.*=.*['\"].*['\"]" "$PROJECT_ROOT"/*.py 2>/dev/null | grep -v "archive/" | grep -v "#" | grep -v "password_hash" > /dev/null; then
    print_warn "Possibles mots de passe hardcodés trouvés"
    grep -r "password.*=.*['\"].*['\"]" "$PROJECT_ROOT"/*.py 2>/dev/null | grep -v "archive/" | grep -v "#" | grep -v "password_hash" | head -3
fi

# =============================================================================
# 7. Vérification dépendances
# =============================================================================
print_check "Vérification dépendances Python"

if command -v python3 &> /dev/null; then
    # Vérifier que PyJWT est installé (pour auth)
    if python3 -c "import jwt" 2>/dev/null; then
        print_pass "PyJWT installé (authentification OK)"
    else
        print_fail "PyJWT non installé (authentification NON fonctionnelle)"
    fi
    
    # Vérifier Flask
    if python3 -c "import flask" 2>/dev/null; then
        FLASK_VERSION=$(python3 -c "import flask; print(flask.__version__)" 2>/dev/null)
        print_pass "Flask installé (version: $FLASK_VERSION)"
    else
        print_fail "Flask non installé"
    fi
else
    print_fail "Python3 non disponible"
fi

# =============================================================================
# 8. Vérification permissions
# =============================================================================
print_check "Vérification permissions fichiers"

# Scripts doivent être exécutables
for script in vrm_start.sh build_macos_dmg.sh; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            print_pass "$script est exécutable"
        else
            print_warn "$script n'est pas exécutable"
        fi
    fi
done

# =============================================================================
# RAPPORT FINAL
# =============================================================================
echo -e "\n${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Rapport Final${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

echo -e "\n${GREEN}✅ Tests réussis   : $PASSED${NC}"
if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Avertissements : $WARNINGS${NC}"
fi
if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}❌ Erreurs         : $ERRORS${NC}"
fi

echo -e "\n${BLUE}════════════════════════════════════════════════════════${NC}"

# Recommandations
if [ $ERRORS -gt 0 ]; then
    echo -e "\n${RED}🚨 ATTENTION : Des erreurs critiques ont été détectées${NC}"
    echo -e "${RED}   Le déploiement en production est DÉCONSEILLÉ${NC}"
    echo -e "\n${YELLOW}📖 Consultez SECURITY_PRODUCTION.md pour les solutions${NC}"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "\n${YELLOW}⚠️  Des avertissements ont été émis${NC}"
    echo -e "${YELLOW}   Vérifiez les points ci-dessus avant déploiement${NC}"
    echo -e "\n${BLUE}📖 Référence : SECURITY_PRODUCTION.md${NC}"
    exit 0
else
    echo -e "\n${GREEN}✅ Configuration production validée avec succès !${NC}"
    echo -e "${GREEN}   Le déploiement peut être effectué${NC}"
    exit 0
fi

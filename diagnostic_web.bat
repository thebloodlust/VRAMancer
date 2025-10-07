@echo off
echo ============================================================
echo   DIAGNOSTIC WEB DETAILLE
echo ============================================================

echo.
echo === TEST PORTS ET CONNECTIVITE ===

echo 🔍 Test port 5030 (API)...
python -c "
import requests
try:
    r = requests.get('http://localhost:5030/health', timeout=5)
    print('✅ Port 5030 API:', r.json())
except Exception as e:
    print('❌ Port 5030:', e)
"

echo.
echo 🔍 Test port 8080 (Debug Web)...
python -c "
import requests
try:
    r = requests.get('http://localhost:8080', timeout=5)
    print('✅ Port 8080 accessible, status:', r.status_code)
    if r.status_code == 200:
        content = r.text
        if 'Vérification...' in content:
            print('⚠️  Interface bloquée sur vérification')
            print('🔍 Recherche JavaScript errors...')
            if 'testConnectivity' in content:
                print('✅ Fonction testConnectivity présente')
            else:
                print('❌ Fonction testConnectivity manquante')
        else:
            print('✅ Contenu chargé correctement')
except Exception as e:
    print('❌ Port 8080:', e)
"

echo.
echo 🔍 Test port 5000 (Dashboard Avancé)...
python -c "
import requests
try:
    r = requests.get('http://localhost:5000', timeout=5)
    print('✅ Port 5000:', r.status_code)
except Exception as e:
    print('❌ Port 5000:', e)
"

echo.
echo 🔍 Test port 5003 (Mobile)...
python -c "
import requests
try:
    r = requests.get('http://localhost:5003', timeout=5)
    print('✅ Port 5003:', r.status_code)
except Exception as e:
    print('❌ Port 5003:', e)
"

pause
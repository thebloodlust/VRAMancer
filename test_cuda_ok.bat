@echo off
echo ==========================================
echo    TEST CUDA RTX 4060 Ti - VALIDÉ
echo ==========================================
echo.

echo 🎉 CUDA détecté en True ! Excellent !
echo.

echo [TEST 1] Vérification PyTorch CUDA
echo ------------------------------------
python -c "import torch; print(f'✅ CUDA: {torch.cuda.is_available()}'); print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'✅ Mémoire: {torch.cuda.get_device_properties(0).total_memory // 1024**3 if torch.cuda.is_available() else 0} GB')"
echo.

echo [TEST 2] Diagnostic VRAMancer complet
echo ------------------------------------
python diagnostic_cuda.py
echo.

echo [TEST 3] Backend VRAMancer avec CUDA
echo ------------------------------------
python -c "
import sys, os
sys.path.insert(0, os.getcwd())
from core.utils import detect_backend, enumerate_devices
print('Backend détecté:', detect_backend())
devices = enumerate_devices()
print(f'Devices trouvés: {len(devices)}')
for d in devices:
    print(f'  - {d[\"name\"]} ({d[\"backend\"]})')
    if 'memory' in d:
        print(f'    Mémoire: {d[\"memory\"]} MB')
"
echo.

echo ==========================================
echo         LANCEMENT VRAMANCER
echo ==========================================
echo.
echo 🚀 Maintenant que CUDA fonctionne :
echo.
echo 1. LANCEMENT RECOMMANDÉ :
echo    vramancer_auto.bat
echo.
echo 2. OU MANUEL :
echo    api_permanente.bat
echo    systray_vramancer.bat
echo.
echo 3. TEST WEB :
echo    dashboard_web_avance.bat
echo.
echo ==========================================
echo.
pause
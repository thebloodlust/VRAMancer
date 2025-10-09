# VRAMancer - Simple Launch Guide

## Windows Launch Options

### Option 1: Ultra Simple (No special characters)
```
vrm_start.bat
```

### Option 2: Standard English
```
start_vramancer_simple.bat
```

### Option 3: Full Featured
```
vramancer_hub.bat
```

## Expected Results

After launching any of these files:
- API starts on port 5030
- Choose your interface (1-4)
- System Tray recommended for full experience

## Fixed Issues

- Mobile Dashboard: GPU error resolved
- Windows encoding: All special characters removed
- RTX 4060: Adaptive MB/GB display working

## Manual Launch (if needed)

```bash
# Terminal 1 - API
python api_simple.py

# Terminal 2 - Interface
python systray_vramancer.py           # System Tray
python dashboard/dashboard_qt.py      # Qt
python dashboard/dashboard_web_advanced.py  # Web
python mobile/dashboard_mobile.py     # Mobile
```

All launchers now work without encoding issues!
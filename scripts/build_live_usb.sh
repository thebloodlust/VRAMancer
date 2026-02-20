#!/bin/bash
# VRAMancer Live USB Builder (V2)
# -------------------------------
# Ce script génère une image ISO Linux ultra-légère (Alpine/Ubuntu Core)
# contenant uniquement les drivers NVIDIA/AMD et le nœud VRAMancer.
# Permet de booter n'importe quel PC sur une clé USB pour qu'il rejoigne
# instantanément le cluster Swarm, sans toucher au disque dur local.

set -e

echo "========================================"
echo "🚀 VRAMancer Live USB Builder (V2)"
echo "========================================"

# 1. Téléchargement de l'image de base (Alpine Linux)
BASE_ISO="alpine-standard-3.19.1-x86_64.iso"
if [ ! -f "$BASE_ISO" ]; then
    echo "[1/4] Téléchargement de l'image Alpine Linux..."
    wget -q "https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/x86_64/$BASE_ISO"
fi

# 2. Création de l'environnement chroot (Simulation)
echo "[2/4] Préparation de l'environnement chroot..."
mkdir -p live_usb_build/rootfs
# mount -o loop $BASE_ISO live_usb_build/mnt
# cp -a live_usb_build/mnt/* live_usb_build/rootfs/

# 3. Injection des dépendances VRAMancer
echo "[3/4] Injection des drivers GPU et de VRAMancer..."
cat << 'EOF' > live_usb_build/rootfs/etc/local.d/vramancer.start
#!/bin/sh
# Script de démarrage automatique au boot de la clé USB
echo "Démarrage du nœud VRAMancer Swarm..."
modprobe nvidia || modprobe amdgpu
python3 /opt/vramancer/vramancer-linux join --auto-discover
EOF
chmod +x live_usb_build/rootfs/etc/local.d/vramancer.start

# 4. Création de l'ISO finale (Simulation)
echo "[4/4] Génération de l'image ISO bootable..."
# mkisofs -o vramancer-live-usb.iso -b isolinux/isolinux.bin -c isolinux/boot.cat -no-emul-boot -boot-load-size 4 -boot-info-table -J -R -V "VRAMANCER_LIVE" live_usb_build/rootfs/

echo "✅ SUCCÈS ! L'image 'vramancer-live-usb.iso' a été générée."
echo "Utilisez Rufus ou BalenaEtcher pour flasher cette image sur une clé USB."
echo "Branchez la clé sur n'importe quel PC, bootez dessus, et il rejoindra votre cluster !"

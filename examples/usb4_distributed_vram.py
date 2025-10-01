from core.network.transmission import send_block
import torch

# Exemple : déport d'un bloc VRAM sur une machine voisine via USB4

def example_usb4_vram_offload():
    # Simule un tensor à transférer
    tensor = torch.randn(1024, 1024)
    tensors = [tensor]
    shapes = [tensor.shape]
    dtypes = [str(tensor.dtype)]
    target_device = "machineB"  # nom ou IP de la machine voisine
    usb4_path = "/mnt/usb4_share"  # chemin de montage USB4 partagé
    send_block(
        tensors,
        shapes,
        dtypes,
        target_device=target_device,
        usb4_path=usb4_path,
        protocol="usb4",
        compress=True
    )
    print("Bloc VRAM transféré via USB4 !")

if __name__ == "__main__":
    example_usb4_vram_offload()

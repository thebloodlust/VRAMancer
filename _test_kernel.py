import torch, os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6;9.0+PTX'
from core.kv_quantizer import KVCacheCompressor

device = 'cuda:1'
comp = KVCacheCompressor(head_dim=128, bits_per_angle=3, qjl_dim=64).to(device)

kv = torch.randn(2, 128, device=device, dtype=torch.float16)
result = comp.compress(kv)
torch.cuda.synchronize()
print("Done")

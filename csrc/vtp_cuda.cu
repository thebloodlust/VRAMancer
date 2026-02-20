#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

torch::Tensor fast_p2p_transfer_cuda(torch::Tensor src, int dst_device) {
    TORCH_CHECK(src.is_cuda(), "Source tensor must be on CUDA");
    int src_device = src.get_device();

    if (src_device == dst_device) {
        return src;
    }

    // Switch context to destination device
    at::cuda::CUDAGuard device_guard(dst_device);
    
    // Allocate memory on destination GPU without initializing
    auto dst = torch::empty_like(src, src.options().device(torch::kCUDA, dst_device));

    // Get the current CUDA stream for the destination device
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Perform asynchronous Peer-to-Peer copy
    // This bypasses the CPU entirely if P2P is enabled (requires Resizable BAR for large tensors)
    cudaError_t err = cudaMemcpyPeerAsync(
        dst.data_ptr(), dst_device,
        src.data_ptr(), src_device,
        src.numel() * src.element_size(),
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyPeerAsync failed: ", cudaGetErrorString(err));

    return dst;
}

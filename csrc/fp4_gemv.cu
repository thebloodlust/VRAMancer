/*
 * VRAMancer FP4 W4A16 GEMV — Hand-optimized CUDA kernel.
 *
 * THE PROBLEM: PyTorch 2.10 lacks optimized CUTLASS FP4 GEMM for Blackwell.
 *   torch._scaled_mm (W4A4) = 3x SLOWER than BF16 baseline.
 *   Triton W4A16 GEMM = correct but ~5x slower than cuBLAS BF16.
 *
 * OUR SOLUTION: Hand-written CUDA GEMV for the decode hot path (M=1).
 *   - Weights stored as FP4 E2M1 packed (2 per uint8) → 62% VRAM savings
 *   - Activations in BF16 native (zero quantization cost)
 *   - E2M1 LUT in shared memory (bank-conflict-free scatter)
 *   - Vectorized 4-byte weight loads (coalesced in K, not N)
 *   - Warp-level K-reduction via __shfl_xor_sync (no shared memory)
 *   - One warp per output element → perfect parallelism
 *
 * ARCHITECTURE:
 *   Grid: (cdiv(N, BLOCK_WARPS),)
 *   Block: (32 * BLOCK_WARPS,)  [BLOCK_WARPS warps per block]
 *   Each warp computes one output element by splitting K across 32 lanes.
 *   Lane i processes bytes [i*chunk, (i+1)*chunk) of the packed weight row.
 *   After dot product, warp shuffle reduction merges the 32 partial sums.
 */

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// E2M1 FP4 lookup table: nibble → float32
__constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,      // positive
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // negative
};


template<int BLOCK_WARPS>
__global__ void fp4_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,      // [K] activation vector
    const uint8_t* __restrict__ w_qdata,       // [N, K//2] packed FP4 weights
    const __nv_bfloat16* __restrict__ w_scale,  // [N, K//16] block scales (BF16, saves 50% DRAM)
    __nv_bfloat16* __restrict__ out,           // [N] output vector
    const int N,
    const int K
) {
    // Shared memory LUT: eliminates constant memory serialization.
    // Constant memory broadcasts (all lanes same addr) but SERIALIZES scatter
    // (16 different nibbles → up to 16x slower). Shared memory handles scatter
    // with bank-conflict-free access (16 entries × 4 bytes → banks 0-15).
    __shared__ float lut[16];
    if (threadIdx.x < 16) {
        lut[threadIdx.x] = FP4_LUT[threadIdx.x];
    }
    __syncthreads();

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * BLOCK_WARPS + warp_id;

    if (n >= N) return;

    const int half_K = K >> 1;
    const uint8_t* __restrict__ w_row = w_qdata + (long long)n * half_K;
    const __nv_bfloat16* __restrict__ s_row = w_scale + (long long)n * (K >> 4);

    float acc = 0.0f;

    // COALESCED warp-cooperative loads: all 32 lanes read consecutive uint32s
    // → 128 bytes per step (one cache line), perfect coalescing.
    // Each lane processes 4 bytes = 8 FP4 values = 8 K elements.
    // All 8 K elements per lane fall within ONE 16-element scale group
    // (because 8 < 16), so we preload one scale per lane per step.
    const int BYTES_PER_STEP = 128;
    const int steps = half_K / BYTES_PER_STEP;

    for (int step = 0; step < steps; step++) {
        const int base_byte = step * BYTES_PER_STEP + lane * 4;

        // Coalesced uint32 load
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(w_row + base_byte);

        // ONE scale per lane per step (8 K elements < 16-element scale group)
        const float s = __bfloat162float(__ldg(&s_row[base_byte >> 3]));

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint8_t packed = (packed4 >> (j * 8)) & 0xFF;
            const int k_even = (base_byte + j) << 1;

            // Shared memory LUT: bank-conflict-free scatter access
            acc += lut[packed & 0x0F] * s * __bfloat162float(x[k_even])
                 + lut[(packed >> 4) & 0x0F] * s * __bfloat162float(x[k_even + 1]);
        }
    }

    // Remainder
    const int rem_start = steps * BYTES_PER_STEP;
    if (rem_start + lane < half_K) {
        const int b = rem_start + lane;
        const uint8_t packed = w_row[b];
        const int k_even = b << 1;
        const float s = __bfloat162float(__ldg(&s_row[k_even >> 4]));
        acc += lut[packed & 0x0F] * s * __bfloat162float(x[k_even])
             + lut[(packed >> 4) & 0x0F] * s * __bfloat162float(x[k_even + 1]);
    }

    // Warp-level butterfly reduction (no shared memory needed)
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, mask);
    }

    // Lane 0 writes the final result
    if (lane == 0) {
        out[n] = __float2bfloat16(acc);
    }
}


// ========================================================================
// FP4 W4A16 GEMM for M > 1 (prefill) — dequant + cuBLAS in two steps
// Uses the GEMV kernel for each row when M is small, or dequant+cublas for large M.
// ========================================================================

__global__ void fp4_dequant_kernel(
    const uint8_t* __restrict__ w_qdata,  // [N, K//2] packed FP4
    const __nv_bfloat16* __restrict__ w_scale,  // [N, K//16] block scales (BF16)
    __nv_bfloat16* __restrict__ w_bf16,   // [N, K] output dequantized weights
    const int N,
    const int K
) {
    // Grid: (cdiv(N * K/2, 256),)
    // Each thread dequants one packed byte → 2 BF16 values
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_bytes = N * (K >> 1);
    if (tid >= total_bytes) return;

    const int n = tid / (K >> 1);
    const int byte_idx = tid % (K >> 1);
    const int k_even = byte_idx << 1;

    const uint8_t packed = w_qdata[tid];
    const float s = __bfloat162float(w_scale[(long long)n * (K >> 4) + (k_even >> 4)]);

    const float val_even = FP4_LUT[packed & 0x0F] * s;
    const float val_odd  = FP4_LUT[(packed >> 4) & 0x0F] * s;

    const long long out_idx = (long long)n * K + k_even;
    w_bf16[out_idx]     = __float2bfloat16(val_even);
    w_bf16[out_idx + 1] = __float2bfloat16(val_odd);
}


// ========================================================================
// Python bindings
// ========================================================================

torch::Tensor fp4_gemv_cuda(
    torch::Tensor x,            // [K] or [1, K] bfloat16
    torch::Tensor w_qdata,      // [N, K//2] uint8
    torch::Tensor w_scale_row   // [N, K//16] bfloat16 (or float32 auto-converted)
) {
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(w_qdata.is_cuda(), "w_qdata must be on CUDA");
    TORCH_CHECK(w_scale_row.is_cuda(), "w_scale must be on CUDA");

    const bool squeeze = x.dim() == 1;
    auto x_flat = squeeze ? x : x.reshape({-1});
    const int K = x_flat.size(0);
    const int N = w_qdata.size(0);

    TORCH_CHECK(K % 64 == 0, "K must be divisible by 64, got ", K);
    TORCH_CHECK(w_qdata.size(1) == K / 2, "w_qdata shape mismatch");
    TORCH_CHECK(w_scale_row.size(1) == K / 16, "w_scale shape mismatch");

    // Auto-convert float32 scales to bfloat16 (saves 50% DRAM bandwidth)
    auto w_scale_bf16 = w_scale_row.dtype() == torch::kBFloat16
        ? w_scale_row
        : w_scale_row.to(torch::kBFloat16);

    auto out = torch::empty({N}, torch::dtype(torch::kBFloat16).device(x.device()));

    constexpr int BLOCK_WARPS = 4;  // 4 warps = 128 threads per block
    const int grid_size = (N + BLOCK_WARPS - 1) / BLOCK_WARPS;

    fp4_gemv_kernel<BLOCK_WARPS><<<grid_size, BLOCK_WARPS * 32, 0,
                                    c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_flat.data_ptr<at::BFloat16>()),
        w_qdata.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(w_scale_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        N, K
    );

    return squeeze ? out : out.unsqueeze(0);
}


torch::Tensor fp4_dequant_cuda(
    torch::Tensor w_qdata,      // [N, K//2] uint8
    torch::Tensor w_scale_row   // [N, K//16] bfloat16 (or float32 auto-converted)
) {
    TORCH_CHECK(w_qdata.is_cuda(), "w_qdata must be on CUDA");
    const int N = w_qdata.size(0);
    const int half_K = w_qdata.size(1);
    const int K = half_K * 2;

    auto w_scale_bf16 = w_scale_row.dtype() == torch::kBFloat16
        ? w_scale_row
        : w_scale_row.to(torch::kBFloat16);

    auto w_bf16 = torch::empty({N, K}, torch::dtype(torch::kBFloat16).device(w_qdata.device()));

    const int total_bytes = N * half_K;
    const int threads = 256;
    const int blocks = (total_bytes + threads - 1) / threads;

    fp4_dequant_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        w_qdata.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(w_scale_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(w_bf16.data_ptr<at::BFloat16>()),
        N, K
    );

    return w_bf16;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp4_gemv", &fp4_gemv_cuda,
          "FP4 W4A16 GEMV (M=1 decode) — hand-optimized CUDA kernel");
    m.def("fp4_dequant", &fp4_dequant_cuda,
          "FP4 → BF16 dequantization kernel (for M>1 prefill via cuBLAS)");
}

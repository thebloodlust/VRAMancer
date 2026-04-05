/**
 * VRAMancer TurboQuant Fused CUDA Kernel
 *
 * Fuses the entire PolarQuant compress + decode + QJL pipeline into
 * a single kernel launch per batch, eliminating ~80 separate PyTorch
 * kernel launches per compress() call.
 *
 * Pipeline per row:
 *   1. Hadamard rotation (butterfly, in-register)
 *   2. Polar encode (recursive cart→polar, quantize angles)
 *   3. Polar decode (reconstruct from quantized angles)
 *   4. Residual = rotated - reconstructed
 *   5. QJL projection (matmul with JL matrix, sign bits + norm)
 *
 * Supports head_dim up to 256 (padded to power-of-2).
 * One thread-block per row. Threads = padded_dim (max 256).
 *
 * Compile: JIT via torch.utils.cpp_extension.load()
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Max supported padded dimension (head_dim 128 → 128 threads, 256 → 256)
#define MAX_DIM 256
#define MAX_LEVELS 8  // log2(256)

// ─── Shared memory layout ────────────────────────────────────────
// We use shared memory for the Hadamard butterfly and polar stages.
// Layout: float smem[MAX_DIM] for the working vector.

// ─── Hadamard butterfly (in shared memory) ───────────────────────

__device__ void hadamard_butterfly_smem(float* smem, const float* signs,
                                         int dim, int tid) {
    // Apply sign flip: x[i] *= signs[i]
    if (tid < dim) {
        smem[tid] *= signs[tid];
    }
    __syncthreads();

    // Butterfly passes: log2(dim) stages
    for (int h = 1; h < dim; h <<= 1) {
        if (tid < dim) {
            int block_id = tid / (2 * h);
            int offset_in_block = tid % (2 * h);
            int base = block_id * 2 * h;

            float a, b;
            if (offset_in_block < h) {
                a = smem[base + offset_in_block];
                b = smem[base + offset_in_block + h];
            } else {
                a = smem[base + offset_in_block - h];
                b = smem[base + offset_in_block];
            }
            __syncthreads();

            if (offset_in_block < h) {
                smem[base + offset_in_block] = a + b;
                smem[base + offset_in_block + h] = a - b;
            }
        }
        __syncthreads();
    }

    // Scale by 1/sqrt(dim)
    if (tid < dim) {
        smem[tid] *= rsqrtf((float)dim);
    }
    __syncthreads();
}

// Inverse Hadamard: H^T @ x then multiply by signs
__device__ void hadamard_inverse_smem(float* smem, const float* signs,
                                       int dim, int tid) {
    // Butterfly (same as forward — H is symmetric)
    for (int h = 1; h < dim; h <<= 1) {
        if (tid < dim) {
            int block_id = tid / (2 * h);
            int offset_in_block = tid % (2 * h);
            int base = block_id * 2 * h;

            float a, b;
            if (offset_in_block < h) {
                a = smem[base + offset_in_block];
                b = smem[base + offset_in_block + h];
            } else {
                a = smem[base + offset_in_block - h];
                b = smem[base + offset_in_block];
            }
            __syncthreads();

            if (offset_in_block < h) {
                smem[base + offset_in_block] = a + b;
                smem[base + offset_in_block + h] = a - b;
            }
        }
        __syncthreads();
    }

    if (tid < dim) {
        smem[tid] *= rsqrtf((float)dim) * signs[tid];
    }
    __syncthreads();
}


// ─── Fused TurboQuant compress kernel ────────────────────────────
//
// Grid:  (num_rows,)
// Block: (padded_dim,) — one thread per dimension element
//
// Inputs:
//   kv_in:      [N, head_dim] float — raw KV vectors
//   signs:      [padded_dim] float — Hadamard ±1 signs
//   jl_matrix:  [qjl_dim, padded_dim] float — JL projection
//
// Outputs:
//   radius_out:     [N] float16 — final polar radius
//   angles_out:     [N, total_angle_elems] uint8 — packed quantized angles
//   qjl_signs_out:  [N, qjl_dim] uint8 — sign bits of JL projection
//   qjl_norms_out:  [N] float16 — residual norms
//
// The angles are packed sequentially: level 0 has dim/2 elements,
// level 1 has dim/4, ..., level n_levels-1 has 1.
// Total = dim/2 + dim/4 + ... + 1 = dim - 1.

__global__ void turboquant_compress_kernel(
    const float* __restrict__ kv_in,      // [N, head_dim]
    const float* __restrict__ signs,       // [padded_dim]
    const float* __restrict__ jl_matrix,   // [qjl_dim, padded_dim]
    __half* __restrict__ radius_out,       // [N]
    unsigned char* __restrict__ angles_out, // [N, padded_dim - 1]
    unsigned char* __restrict__ qjl_signs_out, // [N, qjl_dim]
    __half* __restrict__ qjl_norms_out,    // [N]
    int N,
    int head_dim,
    int padded_dim,
    int n_levels,
    int bits_per_angle,
    int qjl_dim
) {
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    int n_bins = 1 << bits_per_angle;

    // ── Shared memory: 2 × padded_dim floats ──
    // smem_rotated: the rotated vector (persists for residual computation)
    // smem_work: working buffer for polar encode + decode
    extern __shared__ float shared[];
    float* smem_rotated = shared;                    // [padded_dim]
    float* smem_work = shared + padded_dim;          // [padded_dim]

    // ── Step 0: Load input + pad ──
    float val = 0.0f;
    if (tid < head_dim) {
        val = kv_in[row * head_dim + tid];
    }
    smem_rotated[tid] = val;
    __syncthreads();

    // ── Step 1: Hadamard rotation (in-place in smem_rotated) ──
    hadamard_butterfly_smem(smem_rotated, signs, padded_dim, tid);

    // Copy rotated to work buffer for polar encode
    smem_work[tid] = smem_rotated[tid];
    __syncthreads();

    // ── Step 2: Polar encode (recursive, in smem_work) ──
    // After each level, the "active" part of smem_work shrinks by half.
    // We write quantized angles to angles_out.
    // Angle layout in output: [level0: dim/2 elems][level1: dim/4 elems]...
    int angle_offset = 0;
    int active_dim = padded_dim;

    for (int level = 0; level < n_levels; level++) {
        int half_dim = active_dim / 2;

        if (tid < half_dim) {
            float a = smem_work[tid * 2];
            float b = smem_work[tid * 2 + 1];

            float r = sqrtf(a * a + b * b + 1e-12f);
            float theta = atan2f(b, a);

            // Quantize angle
            float lo, hi;
            if (level == 0) {
                lo = -3.14159265358979f;
                hi = 3.14159265358979f;
            } else {
                lo = 0.0f;
                hi = 1.57079632679490f;  // pi/2
            }

            float normalized = (theta - lo) / (hi - lo);
            int idx = __float2int_rd(normalized * n_bins);  // floor, matches Python clamp
            idx = max(0, min(n_bins - 1, idx));

            // Write quantized angle
            angles_out[row * (padded_dim - 1) + angle_offset + tid] = (unsigned char)idx;

            // Write radius for next level
            smem_work[tid] = r;
        }
        __syncthreads();

        angle_offset += half_dim;
        active_dim = half_dim;
    }

    // smem_work[0] = final radius
    if (tid == 0) {
        radius_out[row] = __float2half(smem_work[0]);
    }
    __syncthreads();

    // ── Step 3: Polar decode (reconstruct from quantized angles) ──
    // Start from radius, expand back to padded_dim.
    // CRITICAL: read radius into register BEFORE writing expanded pairs,
    // because smem[tid] overlaps with smem[tid*2]/smem[tid*2+1] output range.
    if (tid == 0) {
        smem_work[0] = __half2float(radius_out[row]);
    }
    __syncthreads();

    for (int level = n_levels - 1; level >= 0; level--) {
        int level_offset = 0;
        {
            int d = padded_dim;
            for (int l = 0; l < level; l++) {
                level_offset += d / 2;
                d /= 2;
            }
        }
        int half_dim = padded_dim >> (level + 1);

        // Read radius into register (before any writes to avoid race)
        float r = 0.0f;
        if (tid < half_dim) {
            r = smem_work[tid];
        }
        __syncthreads();  // all reads done before any writes

        if (tid < half_dim) {
            unsigned char q_idx = angles_out[row * (padded_dim - 1) + level_offset + tid];

            float lo, hi;
            if (level == 0) {
                lo = -3.14159265358979f;
                hi = 3.14159265358979f;
            } else {
                lo = 0.0f;
                hi = 1.57079632679490f;
            }

            float angle = lo + ((float)q_idx + 0.5f) * (hi - lo) / (float)n_bins;
            float cos_a, sin_a;
            sincosf(angle, &sin_a, &cos_a);

            smem_work[tid * 2] = r * cos_a;
            smem_work[tid * 2 + 1] = r * sin_a;
        }
        __syncthreads();
    }

    // smem_work now contains the reconstructed vector [padded_dim]

    // ── Step 4: Compute residual = rotated - reconstructed ──
    float residual_val = smem_rotated[tid] - smem_work[tid];

    // Compute residual norm (block-wide reduction)
    float r2 = residual_val * residual_val;

    // Warp reduction for norm²
    // Use the extra 8 floats at end of dynamic shared memory (avoids
    // static/dynamic shared memory interaction issues on SM 12.0+)
    float* warp_sums = shared + 2 * padded_dim;  // [8 floats]
    unsigned mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset >>= 1) {
        r2 += __shfl_xor_sync(mask, r2, offset);
    }
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = r2;
    }
    __syncthreads();

    float norm_sq = 0.0f;
    if (tid == 0) {
        int num_warps = (padded_dim + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            norm_sq += warp_sums[w];
        }
        qjl_norms_out[row] = __float2half(sqrtf(norm_sq));
    }
    __syncthreads();

    // ── Step 5: QJL projection — sign(residual @ JL^T) ──
    // Each thread computes one or more JL projections.
    // JL matrix is [qjl_dim, padded_dim].
    // We need: projected[j] = sum_i(residual[i] * jl_matrix[j, i])

    // Store residual in smem_rotated (reuse buffer)
    smem_rotated[tid] = residual_val;
    __syncthreads();

    // Each thread handles ceil(qjl_dim / padded_dim) JL dimensions
    for (int j = tid; j < qjl_dim; j += padded_dim) {
        float dot = 0.0f;
        const float* jl_row = jl_matrix + j * padded_dim;
        for (int i = 0; i < padded_dim; i++) {
            dot += smem_rotated[i] * jl_row[i];
        }
        qjl_signs_out[row * qjl_dim + j] = (dot > 0.0f) ? 1 : 0;
    }
}


// ─── Fused TurboQuant decode kernel ──────────────────────────────
//
// Polar decode + inverse Hadamard in a single launch.
//
// Grid:  (num_rows,)
// Block: (padded_dim,)

__global__ void turboquant_decompress_kernel(
    const __half* __restrict__ radius_in,      // [N]
    const unsigned char* __restrict__ angles_in, // [N, padded_dim - 1]
    const float* __restrict__ signs,             // [padded_dim]
    float* __restrict__ out,                     // [N, head_dim]
    int N,
    int head_dim,
    int padded_dim,
    int n_levels,
    int bits_per_angle
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int n_bins = 1 << bits_per_angle;

    extern __shared__ float shared[];
    float* smem = shared;

    // Start with radius
    if (tid == 0) {
        smem[0] = __half2float(radius_in[row]);
    }
    __syncthreads();

    // Polar decode (same fix as compress step 3 — read before write to avoid race)
    for (int level = n_levels - 1; level >= 0; level--) {
        int level_offset = 0;
        {
            int d = padded_dim;
            for (int l = 0; l < level; l++) {
                level_offset += d / 2;
                d /= 2;
            }
        }
        int half_dim = padded_dim >> (level + 1);

        float r = 0.0f;
        if (tid < half_dim) {
            r = smem[tid];
        }
        __syncthreads();

        if (tid < half_dim) {
            unsigned char q_idx = angles_in[row * (padded_dim - 1) + level_offset + tid];

            float lo, hi;
            if (level == 0) {
                lo = -3.14159265358979f;
                hi = 3.14159265358979f;
            } else {
                lo = 0.0f;
                hi = 1.57079632679490f;
            }

            float angle = lo + ((float)q_idx + 0.5f) * (hi - lo) / (float)n_bins;
            float cos_a, sin_a;
            sincosf(angle, &sin_a, &cos_a);

            smem[tid * 2] = r * cos_a;
            smem[tid * 2 + 1] = r * sin_a;
        }
        __syncthreads();
    }

    // Inverse Hadamard
    hadamard_inverse_smem(smem, signs, padded_dim, tid);

    // Write output (only head_dim elements, strip padding)
    if (tid < head_dim) {
        out[row * head_dim + tid] = smem[tid];
    }
}


// ─── PyTorch bindings ────────────────────────────────────────────

std::vector<torch::Tensor> turboquant_compress(
    torch::Tensor kv,          // [N, head_dim] float32 on CUDA
    torch::Tensor signs,       // [padded_dim] float32
    torch::Tensor jl_matrix,   // [qjl_dim, padded_dim] float32
    int padded_dim,
    int n_levels,
    int bits_per_angle,
    int qjl_dim
) {
    TORCH_CHECK(kv.is_cuda(), "kv must be on CUDA");
    TORCH_CHECK(kv.dim() == 2, "kv must be [N, head_dim]");

    int N = kv.size(0);
    int head_dim = kv.size(1);

    auto options_f16 = torch::TensorOptions().dtype(torch::kHalf).device(kv.device());
    auto options_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(kv.device());

    auto radius = torch::empty({N}, options_f16);
    auto angles = torch::empty({N, padded_dim - 1}, options_u8);
    auto qjl_signs = torch::empty({N, qjl_dim}, options_u8);
    auto qjl_norms = torch::empty({N}, options_f16);

    int threads = padded_dim;
    int blocks = N;
    // Shared memory: 2 * padded_dim floats + 8 floats for warp_sums
    int smem_bytes = (2 * padded_dim + 8) * sizeof(float);

    turboquant_compress_kernel<<<blocks, threads, smem_bytes>>>(
        kv.data_ptr<float>(),
        signs.data_ptr<float>(),
        jl_matrix.data_ptr<float>(),
        reinterpret_cast<__half*>(radius.data_ptr<at::Half>()),
        angles.data_ptr<unsigned char>(),
        qjl_signs.data_ptr<unsigned char>(),
        reinterpret_cast<__half*>(qjl_norms.data_ptr<at::Half>()),
        N, head_dim, padded_dim, n_levels, bits_per_angle, qjl_dim
    );

    return {radius, angles, qjl_signs, qjl_norms};
}


torch::Tensor turboquant_decompress(
    torch::Tensor radius,      // [N] float16
    torch::Tensor angles,      // [N, padded_dim - 1] uint8
    torch::Tensor signs,       // [padded_dim] float32
    int head_dim,
    int padded_dim,
    int n_levels,
    int bits_per_angle
) {
    TORCH_CHECK(radius.is_cuda(), "radius must be on CUDA");
    int N = radius.size(0);

    auto out = torch::empty({N, head_dim},
                            torch::TensorOptions().dtype(torch::kFloat32).device(radius.device()));

    int threads = padded_dim;
    int blocks = N;
    int smem_bytes = padded_dim * sizeof(float);

    turboquant_decompress_kernel<<<blocks, threads, smem_bytes>>>(
        reinterpret_cast<const __half*>(radius.data_ptr<at::Half>()),
        angles.data_ptr<unsigned char>(),
        signs.data_ptr<float>(),
        out.data_ptr<float>(),
        N, head_dim, padded_dim, n_levels, bits_per_angle
    );

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compress", &turboquant_compress,
          "TurboQuant fused compress (Hadamard + PolarQuant + QJL)");
    m.def("decompress", &turboquant_decompress,
          "TurboQuant fused decompress (PolarQuant decode + inverse Hadamard)");
}

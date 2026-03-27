/**
 * VRAMancer PagedAttention Decode Kernel
 *
 * Computes single-token attention directly from the paged KV cache pool,
 * without materializing contiguous KV tensors. This eliminates the overhead
 * of to_hf_cache() / from_hf_cache() during autoregressive decode.
 *
 * Architecture:
 *   - One warp (32 threads) per (attention_head, batch) pair
 *   - No shared memory, no __syncthreads — pure warp-level ops
 *   - Online softmax (numerically stable, single-pass over KV)
 *   - GQA support (num_heads != num_kv_heads)
 *   - fp32 and fp16 KV pool variants
 *   - TurboQuant QJL 1-bit + Polar Radius extensions (WIP decompression loop)
 *
 * KV Pool layout: [max_pages, num_layers, 2(K/V), num_kv_heads, page_size, head_dim]
 *
 * References:
 *   - vLLM PagedAttention (Kwon et al., 2023)
 *   - FlashDecoding (Dao et al., 2023)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

// ---------------------------------------------------------------------------
// Warp-level primitives (no barriers needed)
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// ---------------------------------------------------------------------------
// PagedAttention V1 — decode kernel (fp32 KV pool)
// ---------------------------------------------------------------------------
// Grid:  (num_heads, batch_size)
// Block: (32,) — one warp
//
// Each lane handles ceil(head_dim / 32) dimensions of the query/output.
// All lanes cooperate on the Q·K dot product via warp_reduce_sum.
// Online softmax avoids a second pass over KV.

__global__ void paged_attention_decode_f32_kernel(
    const float* __restrict__ query,         // [B, H, D]
    const float* __restrict__ kv_pool,       // flat [P, L, 2, KH, PS, D]
    const int32_t* __restrict__ page_table,  // [B, MAX_PP]
    const int32_t* __restrict__ context_lens, // [B]
    float* __restrict__ output,              // [B, H, D]
    const int layer_idx,
    const int head_dim,
    const int num_heads,
    const int num_kv_heads,
    const int page_size,
    const int max_pages_per_seq,
    const int num_layers,
    const float scale
) {
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int lane = threadIdx.x;  // 0..31

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    // GQA head mapping
    const int kv_head_idx = (num_kv_heads < num_heads)
        ? head_idx / (num_heads / num_kv_heads)
        : head_idx;

    // Pre-compute strides for flat KV pool indexing
    const long long stride_page  = (long long)num_layers * 2 * num_kv_heads * page_size * head_dim;
    const long long stride_layer = (long long)2 * num_kv_heads * page_size * head_dim;
    const long long stride_kv    = (long long)num_kv_heads * page_size * head_dim;
    const long long stride_head  = (long long)page_size * head_dim;

    const long long k_head_offset = (long long)layer_idx * stride_layer
                                  + 0LL * stride_kv
                                  + (long long)kv_head_idx * stride_head;
    const long long v_head_offset = (long long)layer_idx * stride_layer
                                  + 1LL * stride_kv
                                  + (long long)kv_head_idx * stride_head;

    // Each lane handles multiple dimensions (head_dim / 32 rounded up)
    const int dims_per_lane = (head_dim + 31) / 32;

    // Load query into registers
    float q_reg[8];  // supports head_dim up to 256 (8 * 32)
    const int q_base = batch_idx * num_heads * head_dim + head_idx * head_dim;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = lane + i * 32;
        q_reg[i] = (i < dims_per_lane && d < head_dim) ? query[q_base + d] : 0.0f;
    }

    // Online softmax state
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const int32_t* my_pages = page_table + (long long)batch_idx * max_pages_per_seq;
    const int num_pages = (ctx_len + page_size - 1) / page_size;

    for (int pi = 0; pi < num_pages; pi++) {
        const int page_id = my_pages[pi];
        if (page_id < 0) break;

        const int tokens_in_page = min(page_size, ctx_len - pi * page_size);
        const long long page_base = (long long)page_id * stride_page;

        for (int t = 0; t < tokens_in_page; t++) {
            // Q · K dot product (distributed across warp lanes)
            float partial = 0.0f;
            const long long k_offset = page_base + k_head_offset + (long long)t * head_dim;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int d = lane + i * 32;
                if (i < dims_per_lane && d < head_dim) {
                    partial += q_reg[i] * kv_pool[k_offset + d];
                }
            }

            // Warp-level reduction → all lanes get the same score
            float score = warp_reduce_sum(partial) * scale;

            // Online softmax update
            float new_max = fmaxf(max_score, score);
            float correction = __expf(max_score - new_max);
            float exp_score = __expf(score - new_max);

            // Accumulate weighted V
            const long long v_offset = page_base + v_head_offset + (long long)t * head_dim;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int d = lane + i * 32;
                if (i < dims_per_lane && d < head_dim) {
                    acc[i] = acc[i] * correction + kv_pool[v_offset + d] * exp_score;
                }
            }
            sum_exp = sum_exp * correction + exp_score;
            max_score = new_max;
        }
    }

    // Write normalized output
    const int o_base = batch_idx * num_heads * head_dim + head_idx * head_dim;
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = lane + i * 32;
        if (i < dims_per_lane && d < head_dim) {
            output[o_base + d] = acc[i] * inv_sum;
        }
    }
}


// ---------------------------------------------------------------------------
// PagedAttention V1 — decode kernel (fp16 KV pool, fp32 compute)
// ---------------------------------------------------------------------------

__global__ void paged_attention_decode_f16kv_kernel(
    const float* __restrict__ query,
    const __half* __restrict__ kv_pool,      // fp16 KV pool
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ context_lens,
    float* __restrict__ output,
    const int layer_idx,
    const int head_dim,
    const int num_heads,
    const int num_kv_heads,
    const int page_size,
    const int max_pages_per_seq,
    const int num_layers,
    const float scale
) {
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int lane = threadIdx.x;

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int kv_head_idx = (num_kv_heads < num_heads)
        ? head_idx / (num_heads / num_kv_heads)
        : head_idx;

    const long long stride_page  = (long long)num_layers * 2 * num_kv_heads * page_size * head_dim;
    const long long stride_layer = (long long)2 * num_kv_heads * page_size * head_dim;
    const long long stride_kv    = (long long)num_kv_heads * page_size * head_dim;
    const long long stride_head  = (long long)page_size * head_dim;

    const long long k_head_offset = (long long)layer_idx * stride_layer
                                  + (long long)kv_head_idx * stride_head;
    const long long v_head_offset = (long long)layer_idx * stride_layer
                                  + stride_kv
                                  + (long long)kv_head_idx * stride_head;

    const int dims_per_lane = (head_dim + 31) / 32;

    float q_reg[8];
    const int q_base = batch_idx * num_heads * head_dim + head_idx * head_dim;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = lane + i * 32;
        q_reg[i] = (i < dims_per_lane && d < head_dim) ? query[q_base + d] : 0.0f;
    }

    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float acc[8] = {0};

    const int32_t* my_pages = page_table + (long long)batch_idx * max_pages_per_seq;
    const int num_pages = (ctx_len + page_size - 1) / page_size;

    for (int pi = 0; pi < num_pages; pi++) {
        const int page_id = my_pages[pi];
        if (page_id < 0) break;

        const int tokens_in_page = min(page_size, ctx_len - pi * page_size);
        const long long page_base = (long long)page_id * stride_page;

        for (int t = 0; t < tokens_in_page; t++) {
            float partial = 0.0f;
            const long long k_offset = page_base + k_head_offset + (long long)t * head_dim;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int d = lane + i * 32;
                if (i < dims_per_lane && d < head_dim) {
                    partial += q_reg[i] * __half2float(kv_pool[k_offset + d]);
                }
            }

            float score = warp_reduce_sum(partial) * scale;

            float new_max = fmaxf(max_score, score);
            float correction = __expf(max_score - new_max);
            float exp_score = __expf(score - new_max);

            const long long v_offset = page_base + v_head_offset + (long long)t * head_dim;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int d = lane + i * 32;
                if (i < dims_per_lane && d < head_dim) {
                    acc[i] = acc[i] * correction
                           + __half2float(kv_pool[v_offset + d]) * exp_score;
                }
            }
            sum_exp = sum_exp * correction + exp_score;
            max_score = new_max;
        }
    }

    const int o_base = batch_idx * num_heads * head_dim + head_idx * head_dim;
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = lane + i * 32;
        if (i < dims_per_lane && d < head_dim) {
            output[o_base + d] = acc[i] * inv_sum;
        }
    }
}


// ---------------------------------------------------------------------------
// PagedAttention V1 — decode kernel (Q4 quantized KV pool, fp32 compute)
// ---------------------------------------------------------------------------
// KV data stored as packed uint8 (2 × 4-bit values per byte).
// Per-group scale (fp16) + zero-point (fp16) for dequantization:
//   value = (nibble - zero_point) * scale
// Group size is fixed (32 elements → 1 scale + 1 zp per 32-dim group).
// Layout: kv_pool_q4  = [P, L, 2, KH, PS, D/2]  (packed nibbles)
//         kv_scales   = [P, L, 2, KH, PS, D/GROUP] (fp16)
//         kv_zeros    = [P, L, 2, KH, PS, D/GROUP] (fp16)

#define Q4_GROUP_SIZE 32

__device__ __forceinline__ float dequant_q4(
    uint8_t packed, int nibble_idx, float scale, float zero_point
) {
    int val = (nibble_idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    return (static_cast<float>(val) - zero_point) * scale;
}

__global__ void paged_attention_decode_q4_kernel(
    const float* __restrict__ query,           // [B, H, D]
    const uint8_t* __restrict__ kv_pool_q4,    // packed [P, L, 2, KH, PS, D/2]
    const __half* __restrict__ kv_scales,       // [P, L, 2, KH, PS, num_groups]
    const __half* __restrict__ kv_zeros,        // [P, L, 2, KH, PS, num_groups]
    const int32_t* __restrict__ page_table,    // [B, MAX_PP]
    const int32_t* __restrict__ context_lens,  // [B]
    float* __restrict__ output,                // [B, H, D]
    const int layer_idx,
    const int head_dim,
    const int num_heads,
    const int num_kv_heads,
    const int page_size,
    const int max_pages_per_seq,
    const int num_layers,
    const float scale
) {
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int lane = threadIdx.x;

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int kv_head_idx = (num_kv_heads < num_heads)
        ? head_idx / (num_heads / num_kv_heads)
        : head_idx;

    const int half_dim = head_dim / 2;
    const int num_groups = (head_dim + Q4_GROUP_SIZE - 1) / Q4_GROUP_SIZE;

    // Strides for packed Q4 pool (D/2 per token)
    const long long stride_page_q  = (long long)num_layers * 2 * num_kv_heads * page_size * half_dim;
    const long long stride_layer_q = (long long)2 * num_kv_heads * page_size * half_dim;
    const long long stride_kv_q    = (long long)num_kv_heads * page_size * half_dim;
    const long long stride_head_q  = (long long)page_size * half_dim;

    // Strides for scales/zeros (num_groups per token)
    const long long stride_page_s  = (long long)num_layers * 2 * num_kv_heads * page_size * num_groups;
    const long long stride_layer_s = (long long)2 * num_kv_heads * page_size * num_groups;
    const long long stride_kv_s    = (long long)num_kv_heads * page_size * num_groups;
    const long long stride_head_s  = (long long)page_size * num_groups;

    const long long k_q_off = (long long)layer_idx * stride_layer_q + (long long)kv_head_idx * stride_head_q;
    const long long v_q_off = (long long)layer_idx * stride_layer_q + stride_kv_q + (long long)kv_head_idx * stride_head_q;
    const long long k_s_off = (long long)layer_idx * stride_layer_s + (long long)kv_head_idx * stride_head_s;
    const long long v_s_off = (long long)layer_idx * stride_layer_s + stride_kv_s + (long long)kv_head_idx * stride_head_s;

    const int dims_per_lane = (head_dim + 31) / 32;

    float q_reg[8];
    const int q_base = batch_idx * num_heads * head_dim + head_idx * head_dim;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = lane + i * 32;
        q_reg[i] = (i < dims_per_lane && d < head_dim) ? query[q_base + d] : 0.0f;
    }

    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float acc[8] = {0};

    const int32_t* my_pages = page_table + (long long)batch_idx * max_pages_per_seq;
    const int num_pages = (ctx_len + page_size - 1) / page_size;

    for (int pi = 0; pi < num_pages; pi++) {
        const int page_id = my_pages[pi];
        if (page_id < 0) break;

        const int tokens_in_page = min(page_size, ctx_len - pi * page_size);
        const long long pbase_q = (long long)page_id * stride_page_q;
        const long long pbase_s = (long long)page_id * stride_page_s;

        for (int t = 0; t < tokens_in_page; t++) {
            // Q · K (with inline Q4 dequantization)
            float partial = 0.0f;
            const long long k_off_t = pbase_q + k_q_off + (long long)t * half_dim;
            const long long k_sc_t  = pbase_s + k_s_off + (long long)t * num_groups;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int d = lane + i * 32;
                if (i < dims_per_lane && d < head_dim) {
                    int byte_idx = d / 2;
                    int nibble   = d % 2;
                    int group    = d / Q4_GROUP_SIZE;
                    uint8_t packed = kv_pool_q4[k_off_t + byte_idx];
                    float sc = __half2float(kv_scales[k_sc_t + group]);
                    float zp = __half2float(kv_zeros[k_sc_t + group]);
                    float k_val = dequant_q4(packed, nibble, sc, zp);
                    partial += q_reg[i] * k_val;
                }
            }

            float score = warp_reduce_sum(partial) * scale;

            float new_max = fmaxf(max_score, score);
            float correction = __expf(max_score - new_max);
            float exp_score = __expf(score - new_max);

            // Accumulate V (inline Q4 dequantization)
            const long long v_off_t = pbase_q + v_q_off + (long long)t * half_dim;
            const long long v_sc_t  = pbase_s + v_s_off + (long long)t * num_groups;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int d = lane + i * 32;
                if (i < dims_per_lane && d < head_dim) {
                    int byte_idx = d / 2;
                    int nibble   = d % 2;
                    int group    = d / Q4_GROUP_SIZE;
                    uint8_t packed = kv_pool_q4[v_off_t + byte_idx];
                    float sc = __half2float(kv_scales[v_sc_t + group]);
                    float zp = __half2float(kv_zeros[v_sc_t + group]);
                    float v_val = dequant_q4(packed, nibble, sc, zp);
                    acc[i] = acc[i] * correction + v_val * exp_score;
                }
            }
            sum_exp = sum_exp * correction + exp_score;
            max_score = new_max;
        }
    }

    const int o_base = batch_idx * num_heads * head_dim + head_idx * head_dim;
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = lane + i * 32;
        if (i < dims_per_lane && d < head_dim) {
            output[o_base + d] = acc[i] * inv_sum;
        }
    }
}


// ---------------------------------------------------------------------------
// C++ dispatch functions (called from Python via pybind11)
// ---------------------------------------------------------------------------

torch::Tensor paged_attention_decode(
    torch::Tensor query,         // [B, H, D] fp32
    torch::Tensor kv_pool,       // [P, L, 2, KH, PS, D] fp32 or fp16
    torch::Tensor page_table,    // [B, MAX_PP] int32
    torch::Tensor context_lens,  // [B] int32
    int layer_idx,
    float scale
) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    TORCH_CHECK(kv_pool.is_cuda(), "kv_pool must be CUDA");
    TORCH_CHECK(query.dim() == 3, "query must be [B, H, D]");
    TORCH_CHECK(kv_pool.dim() == 6, "kv_pool must be [P, L, 2, KH, PS, D]");

    const int batch_size = query.size(0);
    const int num_heads = query.size(1);
    const int head_dim = query.size(2);
    const int num_layers = kv_pool.size(1);
    const int num_kv_heads = kv_pool.size(3);
    const int page_size = kv_pool.size(4);
    const int max_pages_per_seq = page_table.size(1);

    TORCH_CHECK(head_dim <= 256, "head_dim must be <= 256");
    TORCH_CHECK(kv_pool.size(5) == head_dim,
                "kv_pool head_dim must match query head_dim");

    auto output = torch::zeros({batch_size, num_heads, head_dim},
                               query.options());

    const dim3 grid(num_heads, batch_size);
    const dim3 block(32);  // one warp

    if (kv_pool.scalar_type() == at::ScalarType::Half) {
        paged_attention_decode_f16kv_kernel<<<grid, block>>>(
            query.data_ptr<float>(),
            reinterpret_cast<const __half*>(kv_pool.data_ptr<at::Half>()),
            page_table.data_ptr<int32_t>(),
            context_lens.data_ptr<int32_t>(),
            output.data_ptr<float>(),
            layer_idx, head_dim, num_heads, num_kv_heads,
            page_size, max_pages_per_seq, num_layers, scale
        );
    } else {
        paged_attention_decode_f32_kernel<<<grid, block>>>(
            query.data_ptr<float>(),
            kv_pool.data_ptr<float>(),
            page_table.data_ptr<int32_t>(),
            context_lens.data_ptr<int32_t>(),
            output.data_ptr<float>(),
            layer_idx, head_dim, num_heads, num_kv_heads,
            page_size, max_pages_per_seq, num_layers, scale
        );
    }

    return output;
}


torch::Tensor paged_attention_decode_q4(
    torch::Tensor query,          // [B, H, D] fp32
    torch::Tensor kv_pool_q4,     // [P, L, 2, KH, PS, D/2] uint8 packed nibbles
    torch::Tensor kv_scales,      // [P, L, 2, KH, PS, num_groups] fp16
    torch::Tensor kv_zeros,       // [P, L, 2, KH, PS, num_groups] fp16
    torch::Tensor page_table,     // [B, MAX_PP] int32
    torch::Tensor context_lens,   // [B] int32
    int layer_idx,
    float scale
) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    TORCH_CHECK(kv_pool_q4.is_cuda(), "kv_pool_q4 must be CUDA");
    TORCH_CHECK(kv_scales.is_cuda(), "kv_scales must be CUDA");
    TORCH_CHECK(query.dim() == 3, "query must be [B, H, D]");
    TORCH_CHECK(kv_pool_q4.dim() == 6, "kv_pool_q4 must be [P, L, 2, KH, PS, D/2]");

    const int batch_size = query.size(0);
    const int num_heads = query.size(1);
    const int head_dim = query.size(2);
    const int num_layers = kv_pool_q4.size(1);
    const int num_kv_heads = kv_pool_q4.size(3);
    const int page_size = kv_pool_q4.size(4);
    const int max_pages_per_seq = page_table.size(1);

    TORCH_CHECK(head_dim <= 256, "head_dim must be <= 256");
    TORCH_CHECK(kv_pool_q4.size(5) == head_dim / 2,
                "kv_pool_q4 last dim must be head_dim/2 (packed nibbles)");

    auto output = torch::zeros({batch_size, num_heads, head_dim},
                               query.options());

    const dim3 grid(num_heads, batch_size);
    const dim3 block(32);

    paged_attention_decode_q4_kernel<<<grid, block>>>(
        query.data_ptr<float>(),
        kv_pool_q4.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(kv_scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(kv_zeros.data_ptr<at::Half>()),
        page_table.data_ptr<int32_t>(),
        context_lens.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        layer_idx, head_dim, num_heads, num_kv_heads,
        page_size, max_pages_per_seq, num_layers, scale
    );

    return output;
}


// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_decode",
          &paged_attention_decode,
          "PagedAttention decode — single-token attention from paged KV pool",
          py::arg("query"),
          py::arg("kv_pool"),
          py::arg("page_table"),
          py::arg("context_lens"),
          py::arg("layer_idx"),
          py::arg("scale"));
    m.def("paged_attention_decode_q4",
          &paged_attention_decode_q4,
          "PagedAttention decode — Q4 quantized KV pool with fused dequantization",
          py::arg("query"),
          py::arg("kv_pool_q4"),
          py::arg("kv_scales"),
          py::arg("kv_zeros"),
          py::arg("page_table"),
          py::arg("context_lens"),
          py::arg("layer_idx"),
          py::arg("scale"));
}

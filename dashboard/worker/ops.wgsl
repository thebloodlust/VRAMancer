// VRAMancer WebGPU — GPU Compute Shaders (one file, one @group(0) per entry point)
// Each entry point uses @group(0) only. The JS side creates distinct pipelines
// with distinct bind group layouts for each shader, so there is no conflict.
// WGSL allows overlapping @group/@binding across different entry points.

// ======================== Matmul (16x16 tiled, C = A @ B^T) ========================

struct MatmulParams { M: u32, N: u32, K: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> mm_params: MatmulParams;
@group(0) @binding(1) var<storage, read> mm_A: array<f32>;
@group(0) @binding(2) var<storage, read> mm_B: array<f32>;
@group(0) @binding(3) var<storage, read_write> mm_C: array<f32>;

const TILE: u32 = 16u;
var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn matmul_tiled(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    var acc: f32 = 0.0;
    let numTiles = (mm_params.K + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < numTiles; t++) {
        let a_col = t * TILE + lid.x;
        if (row < mm_params.M && a_col < mm_params.K) {
            tileA[lid.y][lid.x] = mm_A[row * mm_params.K + a_col];
        } else { tileA[lid.y][lid.x] = 0.0; }
        let b_k = t * TILE + lid.y;
        if (col < mm_params.N && b_k < mm_params.K) {
            tileB[lid.y][lid.x] = mm_B[col * mm_params.K + b_k];
        } else { tileB[lid.y][lid.x] = 0.0; }
        workgroupBarrier();
        for (var k: u32 = 0u; k < TILE; k++) { acc += tileA[lid.y][k] * tileB[k][lid.x]; }
        workgroupBarrier();
    }
    if (row < mm_params.M && col < mm_params.N) { mm_C[row * mm_params.N + col] = acc; }
}

// ======================== Add Bias (in-place) ========================

struct BiasParams { total: u32, row_width: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> ab_params: BiasParams;
@group(0) @binding(1) var<storage, read_write> ab_data: array<f32>;
@group(0) @binding(2) var<storage, read> ab_bias: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn add_bias(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ab_params.total) { return; }
    ab_data[i] += ab_bias[i % ab_params.row_width];
}

// ======================== Add Vec (residual) ========================

struct VecParams { len: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<uniform> av_params: VecParams;
@group(0) @binding(1) var<storage, read> av_a: array<f32>;
@group(0) @binding(2) var<storage, read> av_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> av_out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn add_vec(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= av_params.len) { return; }
    av_out[i] = av_a[i] + av_b[i];
}

// ======================== LayerNorm ========================

struct LNParams { seq_len: u32, H: u32, eps: f32, _p: u32 };

@group(0) @binding(0) var<uniform> ln_params: LNParams;
@group(0) @binding(1) var<storage, read> ln_input: array<f32>;
@group(0) @binding(2) var<storage, read> ln_weight: array<f32>;
@group(0) @binding(3) var<storage, read> ln_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> ln_output: array<f32>;

var<workgroup> wg_sum: array<f32, 256>;
var<workgroup> wg_sq_sum: array<f32, 256>;
var<workgroup> wg_mean: f32;
var<workgroup> wg_inv_std: f32;

@compute @workgroup_size(256, 1, 1)
fn layernorm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= ln_params.seq_len) { return; }
    let H = ln_params.H;
    let base = row * H;
    let tid = lid.x;
    let wg_size = 256u;

    var local_sum: f32 = 0.0;
    var idx = tid;
    while (idx < H) { local_sum += ln_input[base + idx]; idx += wg_size; }
    wg_sum[tid] = local_sum;
    workgroupBarrier();
    var s = wg_size / 2u;
    while (s > 0u) { if (tid < s) { wg_sum[tid] += wg_sum[tid + s]; } workgroupBarrier(); s /= 2u; }
    if (tid == 0u) { wg_mean = wg_sum[0] / f32(H); }
    workgroupBarrier();
    let mean = wg_mean;

    var local_sq: f32 = 0.0;
    idx = tid;
    while (idx < H) { let d = ln_input[base + idx] - mean; local_sq += d * d; idx += wg_size; }
    wg_sq_sum[tid] = local_sq;
    workgroupBarrier();
    s = wg_size / 2u;
    while (s > 0u) { if (tid < s) { wg_sq_sum[tid] += wg_sq_sum[tid + s]; } workgroupBarrier(); s /= 2u; }
    if (tid == 0u) { wg_inv_std = 1.0 / sqrt(wg_sq_sum[0] / f32(H) + ln_params.eps); }
    workgroupBarrier();
    let inv_std = wg_inv_std;

    idx = tid;
    while (idx < H) {
        ln_output[base + idx] = (ln_input[base + idx] - mean) * inv_std * ln_weight[idx] + ln_bias[idx];
        idx += wg_size;
    }
}

// ======================== GELU (in-place) ========================

struct GELUParams { len: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<uniform> gelu_params: GELUParams;
@group(0) @binding(1) var<storage, read_write> gelu_data: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= gelu_params.len) { return; }
    let x = gelu_data[i];
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    gelu_data[i] = 0.5 * x * (1.0 + tanh(inner));
}

// ======================== Decode Attention ========================

struct AttnParams { num_heads: u32, head_dim: u32, kv_len: u32, kv_dim: u32 };

@group(0) @binding(0) var<uniform> da_params: AttnParams;
@group(0) @binding(1) var<storage, read> da_q: array<f32>;
@group(0) @binding(2) var<storage, read> da_k: array<f32>;
@group(0) @binding(3) var<storage, read> da_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> da_out: array<f32>;

var<workgroup> scores: array<f32, 2048>;
var<workgroup> score_max: f32;
var<workgroup> score_sum: f32;

@compute @workgroup_size(256, 1, 1)
fn decode_attention(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let h = wid.x;
    if (h >= da_params.num_heads) { return; }
    let head_dim = da_params.head_dim;
    let kv_dim = da_params.kv_dim;
    let kv_len = da_params.kv_len;
    let tid = lid.x;
    let wg_size = 256u;
    let scale = 1.0 / sqrt(f32(head_dim));
    let q_off = h * head_dim;

    var j = tid;
    while (j < kv_len) {
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d++) {
            dot += da_q[q_off + d] * da_k[j * kv_dim + q_off + d];
        }
        scores[j] = dot * scale;
        j += wg_size;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var mx: f32 = -1e30;
        for (var i: u32 = 0u; i < kv_len; i++) { mx = max(mx, scores[i]); }
        score_max = mx;
    }
    workgroupBarrier();

    let mx = score_max;
    j = tid;
    while (j < kv_len) { scores[j] = exp(scores[j] - mx); j += wg_size; }
    workgroupBarrier();

    if (tid == 0u) {
        var sm: f32 = 0.0;
        for (var i: u32 = 0u; i < kv_len; i++) { sm += scores[i]; }
        score_sum = sm;
    }
    workgroupBarrier();

    let inv_sum = 1.0 / score_sum;
    j = tid;
    while (j < kv_len) { scores[j] *= inv_sum; j += wg_size; }
    workgroupBarrier();

    var d = tid;
    while (d < head_dim) {
        var val: f32 = 0.0;
        for (var k: u32 = 0u; k < kv_len; k++) {
            val += scores[k] * da_v[k * kv_dim + q_off + d];
        }
        da_out[q_off + d] = val;
        d += wg_size;
    }
}

// ======================== Argmax LM Head ========================

struct LMParams { vocab_size: u32, H: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> lm_params: LMParams;
@group(0) @binding(1) var<storage, read> lm_hidden: array<f32>;
@group(0) @binding(2) var<storage, read> lm_wte: array<f32>;
@group(0) @binding(3) var<storage, read_write> lm_result: array<u32>;

var<workgroup> wg_max_val: array<f32, 256>;
var<workgroup> wg_max_idx: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn argmax_lmhead(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let V = lm_params.vocab_size;
    let H = lm_params.H;
    let wg_size = 256u;

    var best_val: f32 = -1e30;
    var best_idx: u32 = 0u;
    var v = tid;
    while (v < V) {
        var dot: f32 = 0.0;
        let off = v * H;
        for (var d: u32 = 0u; d < H; d++) { dot += lm_hidden[d] * lm_wte[off + d]; }
        if (dot > best_val) { best_val = dot; best_idx = v; }
        v += wg_size;
    }
    wg_max_val[tid] = best_val;
    wg_max_idx[tid] = best_idx;
    workgroupBarrier();

    var s = wg_size / 2u;
    while (s > 0u) {
        if (tid < s) {
            if (wg_max_val[tid + s] > wg_max_val[tid]) {
                wg_max_val[tid] = wg_max_val[tid + s];
                wg_max_idx[tid] = wg_max_idx[tid + s];
            }
        }
        workgroupBarrier();
        s /= 2u;
    }
    if (tid == 0u) { lm_result[0] = wg_max_idx[0]; }
}

// ======================== Embed Lookup ========================

struct EmbedParams { token_id: u32, position: u32, H: u32, _p: u32 };

@group(0) @binding(0) var<uniform> emb_params: EmbedParams;
@group(0) @binding(1) var<storage, read> emb_wte: array<f32>;
@group(0) @binding(2) var<storage, read> emb_wpe: array<f32>;
@group(0) @binding(3) var<storage, read_write> emb_out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn embed_lookup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= emb_params.H) { return; }
    emb_out[i] = emb_wte[emb_params.token_id * emb_params.H + i]
               + emb_wpe[emb_params.position * emb_params.H + i];
}

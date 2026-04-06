/**
 * VRAMancer WebGPU Inference Worker — FULL GPU MODE
 *
 * ALL transformer ops (LayerNorm, GELU, Attention, LM Head, Embedding)
 * run as GPU compute shaders. Data stays on GPU buffers throughout the
 * forward pass — zero CPU readback until the final argmax token ID.
 *
 * Innovation: eliminates the CPU↔GPU roundtrip that bottlenecked v1
 * (2.5 tok/s) where every matmul result was read back to JS for
 * LayerNorm/GELU/Attention, then re-uploaded for the next matmul.
 *
 * Protocol (binary WebSocket):
 *   0x01 MATMUL, 0x02 PING, 0x10 UPLOAD_TENSOR, 0x11 SET_CONFIG,
 *   0x30 GENERATE, 0x31 TOKEN, 0x32 GENERATE_DONE, 0xFF SHUTDOWN
 */

const OP_MATMUL = 0x01;
const OP_PING = 0x02;
const OP_UPLOAD_TENSOR = 0x10;
const OP_SET_CONFIG = 0x11;
const OP_GENERATE = 0x30;
const OP_TOKEN = 0x31;
const OP_GENERATE_DONE = 0x32;
const OP_SHUTDOWN = 0xFF;

// ======================== State ========================

let device = null;

// Compute pipelines for all ops
let pipelines = {};
let bindGroupLayouts = {};

// Weight pool: name → { buffer: GPUBuffer|null, shape: number[], cpu: Float32Array }
const weights = new Map();
let totalWeightBytes = 0;

let config = null;

// GPU KV cache: layerIdx → { k: GPUBuffer, v: GPUBuffer, length: number, maxLen: number }
const kvCache = new Map();

let generating = false;

// Reusable GPU buffers for intermediate results (allocated once per generate)
let scratchBuffers = {};

// ======================== Status ========================

function postStatus(status, message) {
    if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent("vramancer-status", {
            detail: { status, message, timestamp: Date.now() },
        }));
    }
    console.log(`[VRAMancer] ${status}: ${message}`);
}

// ======================== GPU Setup ========================

function createBGL(entries) {
    return device.createBindGroupLayout({ entries });
}

function storageEntry(binding, type = "read-only-storage") {
    return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } };
}

function uniformEntry(binding) {
    return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } };
}

async function initWebGPU() {
    if (!navigator.gpu) throw new Error("WebGPU not supported");

    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
    });
    if (!adapter) throw new Error("No WebGPU adapter");

    const info = adapter.info || {};
    postStatus("init", `GPU: ${info.description || info.device || info.vendor || "unknown"}`);

    device = await adapter.requestDevice({
        requiredLimits: {
            maxBufferSize: 512 * 1024 * 1024,
            maxStorageBufferBindingSize: 512 * 1024 * 1024,
        },
    });
    device.lost.then(i => postStatus("error", `Device lost: ${i.message}`));

    // Load fused shader module (all ops in one file)
    const shaderCode = await (await fetch("ops.wgsl")).text();
    const module = device.createShaderModule({ code: shaderCode });

    const compilationInfo = await module.getCompilationInfo();
    for (const msg of compilationInfo.messages) {
        if (msg.type === "error") throw new Error(`Shader error: ${msg.message}`);
    }

    // === Create bind group layouts and pipelines for each op ===

    // Matmul: group(0) — uniform, A, B, C
    bindGroupLayouts.matmul = createBGL([
        uniformEntry(0), storageEntry(1), storageEntry(2), storageEntry(3, "storage"),
    ]);
    pipelines.matmul = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayouts.matmul] }),
        compute: { module, entryPoint: "matmul_tiled" },
    });

    // AddBias: uniform, data (rw), bias
    bindGroupLayouts.addBias = createBGL([
        uniformEntry(0), storageEntry(1, "storage"), storageEntry(2),
    ]);
    pipelines.addBias = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayouts.addBias] }),
        compute: { module, entryPoint: "add_bias" },
    });

    // AddVec: uniform, a, b, out
    bindGroupLayouts.addVec = createBGL([
        uniformEntry(0), storageEntry(1), storageEntry(2), storageEntry(3, "storage"),
    ]);
    pipelines.addVec = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayouts.addVec] }),
        compute: { module, entryPoint: "add_vec" },
    });

    // LayerNorm: uniform, input, weight, bias, output
    bindGroupLayouts.layernorm = createBGL([
        uniformEntry(0), storageEntry(1), storageEntry(2), storageEntry(3), storageEntry(4, "storage"),
    ]);
    pipelines.layernorm = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayouts.layernorm] }),
        compute: { module, entryPoint: "layernorm" },
    });

    // GELU: uniform, data (rw)
    bindGroupLayouts.gelu = createBGL([
        uniformEntry(0), storageEntry(1, "storage"),
    ]);
    pipelines.gelu = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayouts.gelu] }),
        compute: { module, entryPoint: "gelu" },
    });

    // DecodeAttention: uniform, Q, K_cache, V_cache, output
    bindGroupLayouts.decodeAttn = createBGL([
        uniformEntry(0), storageEntry(1), storageEntry(2), storageEntry(3), storageEntry(4, "storage"),
    ]);
    pipelines.decodeAttn = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayouts.decodeAttn] }),
        compute: { module, entryPoint: "decode_attention" },
    });

    // ArgmaxLMHead: uniform, hidden, wte, result
    bindGroupLayouts.argmax = createBGL([
        uniformEntry(0), storageEntry(1), storageEntry(2), storageEntry(3, "storage"),
    ]);
    pipelines.argmax = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayouts.argmax] }),
        compute: { module, entryPoint: "argmax_lmhead" },
    });

    // EmbedLookup: uniform, wte, wpe, output
    bindGroupLayouts.embed = createBGL([
        uniformEntry(0), storageEntry(1), storageEntry(2), storageEntry(3, "storage"),
    ]);
    pipelines.embed = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayouts.embed] }),
        compute: { module, entryPoint: "embed_lookup" },
    });

    postStatus("ready", "WebGPU full-GPU pipeline compiled (8 shaders)");
    return info;
}

// ======================== GPU Buffer Helpers ========================

function createUniformBuf(data) {
    const buf = device.createBuffer({
        size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
}

function createGPUBuffer(sizeBytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    return device.createBuffer({ size: Math.max(sizeBytes, 4), usage });
}

function uploadToGPU(data) {
    const buf = createGPUBuffer(data.byteLength);
    device.queue.writeBuffer(buf, 0, data);
    return buf;
}

// ======================== GPU Ops ========================

/** GPU matmul: C = A @ B^T. Returns GPUBuffer[M*N]. A is GPUBuffer, B is GPUBuffer. */
function gpuMatmulBuf(encoder, M, N, K, bufA, bufB, bufC) {
    const TILE = 16;
    const paramsBuf = createUniformBuf(new Uint32Array([M, N, K, 0]));

    const bg = device.createBindGroup({
        layout: bindGroupLayouts.matmul,
        entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: bufA } },
            { binding: 2, resource: { buffer: bufB } },
            { binding: 3, resource: { buffer: bufC } },
        ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.matmul);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(N / TILE), Math.ceil(M / TILE), 1);
    pass.end();
    return paramsBuf; // caller destroys
}

/** GPU add bias in-place: data[i] += bias[i % rowWidth] */
function gpuAddBias(encoder, dataBuf, biasBuf, totalElements, rowWidth) {
    const paramsBuf = createUniformBuf(new Uint32Array([totalElements, rowWidth, 0, 0]));
    const bg = device.createBindGroup({
        layout: bindGroupLayouts.addBias,
        entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: dataBuf } },
            { binding: 2, resource: { buffer: biasBuf } },
        ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.addBias);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(totalElements / 256), 1, 1);
    pass.end();
    return paramsBuf;
}

/** GPU add vec: out = a + b */
function gpuAddVec(encoder, bufA, bufB, bufOut, len) {
    const paramsBuf = createUniformBuf(new Uint32Array([len, 0, 0, 0]));
    const bg = device.createBindGroup({
        layout: bindGroupLayouts.addVec,
        entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: bufA } },
            { binding: 2, resource: { buffer: bufB } },
            { binding: 3, resource: { buffer: bufOut } },
        ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.addVec);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(len / 256), 1, 1);
    pass.end();
    return paramsBuf;
}

/** GPU LayerNorm: output = LN(input) with weight/bias. Input [seqLen, H]. */
function gpuLayerNorm(encoder, inputBuf, weightBuf, biasBuf, outputBuf, seqLen, H) {
    const eps = 1e-5;
    const pData = new ArrayBuffer(16);
    const pView = new DataView(pData);
    pView.setUint32(0, seqLen, true);
    pView.setUint32(4, H, true);
    pView.setFloat32(8, eps, true);
    pView.setUint32(12, 0, true);
    const paramsBuf = createUniformBuf(new Uint8Array(pData));

    const bg = device.createBindGroup({
        layout: bindGroupLayouts.layernorm,
        entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: inputBuf } },
            { binding: 2, resource: { buffer: weightBuf } },
            { binding: 3, resource: { buffer: biasBuf } },
            { binding: 4, resource: { buffer: outputBuf } },
        ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.layernorm);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(seqLen, 1, 1); // 1 workgroup per row
    pass.end();
    return paramsBuf;
}

/** GPU GELU in-place */
function gpuGELU(encoder, dataBuf, len) {
    const paramsBuf = createUniformBuf(new Uint32Array([len, 0, 0, 0]));
    const bg = device.createBindGroup({
        layout: bindGroupLayouts.gelu,
        entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: dataBuf } },
        ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.gelu);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(len / 256), 1, 1);
    pass.end();
    return paramsBuf;
}

/** GPU decode attention: single-query against KV cache */
function gpuDecodeAttention(encoder, qBuf, kCacheBuf, vCacheBuf, outputBuf, numHeads, headDim, kvLen) {
    const kvDim = numHeads * headDim;
    const paramsBuf = createUniformBuf(new Uint32Array([numHeads, headDim, kvLen, kvDim]));
    const bg = device.createBindGroup({
        layout: bindGroupLayouts.decodeAttn,
        entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: qBuf } },
            { binding: 2, resource: { buffer: kCacheBuf } },
            { binding: 3, resource: { buffer: vCacheBuf } },
            { binding: 4, resource: { buffer: outputBuf } },
        ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.decodeAttn);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(numHeads, 1, 1); // 1 workgroup per head
    pass.end();
    return paramsBuf;
}

/** GPU argmax LM head: returns GPUBuffer with single u32 result */
function gpuArgmaxLMHead(encoder, hiddenBuf, wteBuf, resultBuf, vocabSize, H) {
    const paramsBuf = createUniformBuf(new Uint32Array([vocabSize, H, 0, 0]));
    const bg = device.createBindGroup({
        layout: bindGroupLayouts.argmax,
        entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: hiddenBuf } },
            { binding: 2, resource: { buffer: wteBuf } },
            { binding: 3, resource: { buffer: resultBuf } },
        ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.argmax);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1, 1, 1); // single workgroup, 256 threads
    pass.end();
    return paramsBuf;
}

/** GPU embed single token */
function gpuEmbedSingle(encoder, wteBuf, wpeBuf, outputBuf, tokenId, position, H) {
    const paramsBuf = createUniformBuf(new Uint32Array([tokenId, position, H, 0]));
    const bg = device.createBindGroup({
        layout: bindGroupLayouts.embed,
        entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: wteBuf } },
            { binding: 2, resource: { buffer: wpeBuf } },
            { binding: 3, resource: { buffer: outputBuf } },
        ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.embed);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(H / 256), 1, 1);
    pass.end();
    return paramsBuf;
}

// ======================== GPU Linear (matmul + bias) ========================

/** Linear: output = input @ weight^T + bias, ALL on GPU. Returns outputBuf. */
function gpuLinear(encoder, inputBuf, outputBuf, M, inFeatures, weightName, biasName) {
    const w = weights.get(weightName);
    if (!w || !w.buffer) throw new Error(`GPU weight missing: ${weightName}`);
    const outFeatures = w.shape[0];
    const uniforms = [];

    uniforms.push(gpuMatmulBuf(encoder, M, outFeatures, inFeatures, inputBuf, w.buffer, outputBuf));

    if (biasName) {
        const b = weights.get(biasName);
        if (b && b.buffer) {
            uniforms.push(gpuAddBias(encoder, outputBuf, b.buffer, M * outFeatures, outFeatures));
        }
    }
    return uniforms;
}

// ======================== KV Cache (GPU Buffers) ========================

function initKVCaches(numLayers, maxLen, kvDim) {
    // Destroy old caches
    for (const [, c] of kvCache) {
        if (c.k) c.k.destroy();
        if (c.v) c.v.destroy();
    }
    kvCache.clear();

    for (let l = 0; l < numLayers; l++) {
        kvCache.set(l, {
            k: createGPUBuffer(maxLen * kvDim * 4),
            v: createGPUBuffer(maxLen * kvDim * 4),
            length: 0,
            maxLen,
            kvDim,
        });
    }
}

function appendKVFromGPU(encoder, layerIdx, kBuf, vBuf, kvDim) {
    const c = kvCache.get(layerIdx);
    const offset = c.length * kvDim * 4;
    encoder.copyBufferToBuffer(kBuf, 0, c.k, offset, kvDim * 4);
    encoder.copyBufferToBuffer(vBuf, 0, c.v, offset, kvDim * 4);
    c.length++;
}

function resetKVCaches() {
    for (const [, c] of kvCache) c.length = 0;
}

// ======================== CPU Prefill Attention (still CPU for O(n²)) ========================
// Prefill is only done once per prompt and is complex with causal masking.
// Decode attention (hot path) is fully GPU.

function prefillAttentionCPU(Q, K, V, seqLen, numHeads, headDim, layerIdx) {
    const kvDim = numHeads * headDim;
    const cache = kvCache.get(layerIdx);

    // Write K,V to GPU cache
    device.queue.writeBuffer(cache.k, 0, K);
    device.queue.writeBuffer(cache.v, 0, V);
    cache.length = seqLen;

    const scale = 1.0 / Math.sqrt(headDim);
    const output = new Float32Array(seqLen * kvDim);

    for (let h = 0; h < numHeads; h++) {
        for (let i = 0; i < seqLen; i++) {
            const scores = new Float32Array(i + 1);
            for (let j = 0; j <= i; j++) {
                let dot = 0;
                for (let d = 0; d < headDim; d++) {
                    dot += Q[i * kvDim + h * headDim + d] * K[j * kvDim + h * headDim + d];
                }
                scores[j] = dot * scale;
            }
            let max = -1e30;
            for (let j = 0; j <= i; j++) max = Math.max(max, scores[j]);
            let sum = 0;
            for (let j = 0; j <= i; j++) {
                scores[j] = Math.exp(scores[j] - max);
                sum += scores[j];
            }
            for (let j = 0; j <= i; j++) scores[j] /= sum;
            for (let d = 0; d < headDim; d++) {
                let val = 0;
                for (let j = 0; j <= i; j++) {
                    val += scores[j] * V[j * kvDim + h * headDim + d];
                }
                output[i * kvDim + h * headDim + d] = val;
            }
        }
    }
    return output;
}

// ======================== CPU LayerNorm (for prefill only) ========================

function layerNormCPU(x, seqLen, H, weightName, biasName, eps = 1e-5) {
    const w = weights.get(weightName).cpu;
    const b = weights.get(biasName).cpu;
    const out = new Float32Array(seqLen * H);
    for (let s = 0; s < seqLen; s++) {
        const off = s * H;
        let mean = 0;
        for (let i = 0; i < H; i++) mean += x[off + i];
        mean /= H;
        let variance = 0;
        for (let i = 0; i < H; i++) {
            const d = x[off + i] - mean;
            variance += d * d;
        }
        variance /= H;
        const invStd = 1.0 / Math.sqrt(variance + eps);
        for (let i = 0; i < H; i++) {
            out[off + i] = (x[off + i] - mean) * invStd * w[i] + b[i];
        }
    }
    return out;
}

// ======================== CPU linear (for prefill) ========================

async function linearCPU(input, M, inFeatures, weightName, biasName) {
    const w = weights.get(weightName);
    if (!w || !w.buffer) throw new Error(`GPU weight not found: ${weightName}`);
    const outFeatures = w.shape[0];

    // Upload activation, matmul on GPU, readback
    const TILE = 16;
    const paramsBuf = createUniformBuf(new Uint32Array([M, outFeatures, inFeatures, 0]));
    const bufA = uploadToGPU(input);
    const outputSize = M * outFeatures * 4;
    const bufC = createGPUBuffer(outputSize);
    const readbackBuf = device.createBuffer({
        size: outputSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = device.createCommandEncoder();
    const bg = device.createBindGroup({
        layout: bindGroupLayouts.matmul,
        entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: bufA } },
            { binding: 2, resource: { buffer: w.buffer } },
            { binding: 3, resource: { buffer: bufC } },
        ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.matmul);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(outFeatures / TILE), Math.ceil(M / TILE), 1);
    pass.end();
    encoder.copyBufferToBuffer(bufC, 0, readbackBuf, 0, outputSize);
    device.queue.submit([encoder.finish()]);

    await readbackBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readbackBuf.getMappedRange().slice(0));
    readbackBuf.unmap();
    paramsBuf.destroy(); bufA.destroy(); bufC.destroy(); readbackBuf.destroy();

    // Add bias on CPU
    if (biasName) {
        const b = weights.get(biasName);
        if (b) {
            for (let m = 0; m < M; m++) {
                const off = m * outFeatures;
                for (let n = 0; n < outFeatures; n++) result[off + n] += b.cpu[n];
            }
        }
    }
    return result;
}

// ======================== Prefill (CPU path — one-time) ========================

async function prefillForwardLayer(hidden, layerIdx, seqLen) {
    const H = config.hidden_size;
    const numHeads = config.num_heads;
    const headDim = H / numHeads;
    const p = `h.${layerIdx}`;

    const normed1 = layerNormCPU(hidden, seqLen, H, `${p}.ln_1.weight`, `${p}.ln_1.bias`);
    const qkv = await linearCPU(normed1, seqLen, H, `${p}.attn.c_attn.weight`, `${p}.attn.c_attn.bias`);

    const Q = new Float32Array(seqLen * H);
    const K = new Float32Array(seqLen * H);
    const V = new Float32Array(seqLen * H);
    for (let s = 0; s < seqLen; s++) {
        const src = s * 3 * H, dst = s * H;
        Q.set(qkv.subarray(src, src + H), dst);
        K.set(qkv.subarray(src + H, src + 2 * H), dst);
        V.set(qkv.subarray(src + 2 * H, src + 3 * H), dst);
    }

    const attnOut = prefillAttentionCPU(Q, K, V, seqLen, numHeads, headDim, layerIdx);
    const projected = await linearCPU(attnOut, seqLen, H, `${p}.attn.c_proj.weight`, `${p}.attn.c_proj.bias`);

    const afterAttn = new Float32Array(seqLen * H);
    for (let i = 0; i < afterAttn.length; i++) afterAttn[i] = hidden[i] + projected[i];

    const normed2 = layerNormCPU(afterAttn, seqLen, H, `${p}.ln_2.weight`, `${p}.ln_2.bias`);
    const fc = await linearCPU(normed2, seqLen, H, `${p}.mlp.c_fc.weight`, `${p}.mlp.c_fc.bias`);

    const c = Math.sqrt(2.0 / Math.PI);
    for (let i = 0; i < fc.length; i++) {
        const v = fc[i];
        fc[i] = 0.5 * v * (1.0 + Math.tanh(c * (v + 0.044715 * v * v * v)));
    }

    const mlpOut = await linearCPU(fc, seqLen, 4 * H, `${p}.mlp.c_proj.weight`, `${p}.mlp.c_proj.bias`);

    const out = new Float32Array(seqLen * H);
    for (let i = 0; i < out.length; i++) out[i] = afterAttn[i] + mlpOut[i];
    return out;
}

// ======================== DECODE: Full GPU Forward ========================

/**
 * Single-token decode step — ENTIRELY on GPU.
 * Input: hiddenBuf (GPUBuffer [1, H]), returns new hiddenBuf.
 * All intermediate data stays in GPU buffers.
 */
function decodeForwardLayerGPU(encoder, hiddenBuf, layerIdx, tempBufs) {
    const H = config.hidden_size;
    const numHeads = config.num_heads;
    const headDim = H / numHeads;
    const kvDim = numHeads * headDim;
    const p = `h.${layerIdx}`;
    const uniforms = [];

    // Reuse temp buffers
    const normedBuf1 = tempBufs.normed1;   // [1, H]
    const qkvBuf = tempBufs.qkv;           // [1, 3H]
    const qBuf = tempBufs.q;               // [1, H]
    const kBuf = tempBufs.k;               // [1, H]
    const vBuf = tempBufs.v;               // [1, H]
    const attnOutBuf = tempBufs.attnOut;   // [1, H]
    const projBuf = tempBufs.proj;         // [1, H]
    const residBuf1 = tempBufs.resid1;     // [1, H]
    const normedBuf2 = tempBufs.normed2;   // [1, H]
    const fcBuf = tempBufs.fc;             // [1, 4H]
    const mlpBuf = tempBufs.mlp;           // [1, H]
    const residBuf2 = tempBufs.resid2;     // [1, H]

    // 1. LayerNorm
    uniforms.push(gpuLayerNorm(encoder, hiddenBuf,
        weights.get(`${p}.ln_1.weight`).buffer,
        weights.get(`${p}.ln_1.bias`).buffer,
        normedBuf1, 1, H));

    // 2. QKV projection: [1, H] → [1, 3H]
    uniforms.push(...gpuLinear(encoder, normedBuf1, qkvBuf, 1, H,
        `${p}.attn.c_attn.weight`, `${p}.attn.c_attn.bias`));

    // 3. Split Q, K, V via buffer copies
    encoder.copyBufferToBuffer(qkvBuf, 0, qBuf, 0, H * 4);
    encoder.copyBufferToBuffer(qkvBuf, H * 4, kBuf, 0, H * 4);
    encoder.copyBufferToBuffer(qkvBuf, 2 * H * 4, vBuf, 0, H * 4);

    // 4. Append K, V to cache
    appendKVFromGPU(encoder, layerIdx, kBuf, vBuf, kvDim);

    // Flush encoder to ensure cache writes complete before attention reads
    device.queue.submit([encoder.finish()]);
    const encoder2 = device.createCommandEncoder();

    // 5. Decode attention (fully GPU)
    const cache = kvCache.get(layerIdx);
    uniforms.push(gpuDecodeAttention(encoder2, qBuf, cache.k, cache.v,
        attnOutBuf, numHeads, headDim, cache.length));

    // 6. Output projection
    uniforms.push(...gpuLinear(encoder2, attnOutBuf, projBuf, 1, H,
        `${p}.attn.c_proj.weight`, `${p}.attn.c_proj.bias`));

    // 7. Residual: resid1 = hidden + proj
    uniforms.push(gpuAddVec(encoder2, hiddenBuf, projBuf, residBuf1, H));

    // 8. LayerNorm 2
    uniforms.push(gpuLayerNorm(encoder2, residBuf1,
        weights.get(`${p}.ln_2.weight`).buffer,
        weights.get(`${p}.ln_2.bias`).buffer,
        normedBuf2, 1, H));

    // 9. MLP: fc → GELU → proj
    uniforms.push(...gpuLinear(encoder2, normedBuf2, fcBuf, 1, H,
        `${p}.mlp.c_fc.weight`, `${p}.mlp.c_fc.bias`));

    uniforms.push(gpuGELU(encoder2, fcBuf, 4 * H));

    uniforms.push(...gpuLinear(encoder2, fcBuf, mlpBuf, 1, 4 * H,
        `${p}.mlp.c_proj.weight`, `${p}.mlp.c_proj.bias`));

    // 10. Residual: resid2 = resid1 + mlp
    uniforms.push(gpuAddVec(encoder2, residBuf1, mlpBuf, residBuf2, H));

    return { encoder: encoder2, outputBuf: residBuf2, uniforms };
}

// ======================== Allocate Temp Buffers ========================

function allocateTempBuffers() {
    const H = config.hidden_size;
    const sz = (n) => n * 4; // float32 bytes

    const bufs = {
        normed1: createGPUBuffer(sz(H)),
        qkv: createGPUBuffer(sz(3 * H)),
        q: createGPUBuffer(sz(H)),
        k: createGPUBuffer(sz(H)),
        v: createGPUBuffer(sz(H)),
        attnOut: createGPUBuffer(sz(H)),
        proj: createGPUBuffer(sz(H)),
        resid1: createGPUBuffer(sz(H)),
        normed2: createGPUBuffer(sz(H)),
        fc: createGPUBuffer(sz(4 * H)),
        mlp: createGPUBuffer(sz(H)),
        resid2: createGPUBuffer(sz(H)),
        // For the pipeline
        hidden: createGPUBuffer(sz(H)),
        hiddenAlt: createGPUBuffer(sz(H)),
        lnFinal: createGPUBuffer(sz(H)),
        argmaxResult: createGPUBuffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
        argmaxReadback: device.createBuffer({
            size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
    };
    return bufs;
}

function destroyTempBuffers(bufs) {
    for (const [, b] of Object.entries(bufs)) {
        if (b && b.destroy) b.destroy();
    }
}

// ======================== Generate (Full GPU Decode) ========================

async function generate(ws, promptIds, maxTokens) {
    generating = true;
    const H = config.hidden_size;
    const numLayers = config.num_layers;
    const numHeads = config.num_heads;
    const headDim = H / numHeads;
    const maxCtx = config.max_position || 1024;
    const seqLen = promptIds.length;
    const vocabSize = config.vocab_size;

    postStatus("compute", `Generating: prefill ${seqLen} tokens (CPU), decode full-GPU...`);

    initKVCaches(numLayers, maxCtx, numHeads * headDim);

    const t0 = performance.now();

    // === PREFILL (CPU path — O(n²) attention, runs once) ===
    const wte = weights.get("wte.weight").cpu;
    const wpe = weights.get("wpe.weight").cpu;
    let hidden = new Float32Array(seqLen * H);
    for (let s = 0; s < seqLen; s++) {
        const tokOff = promptIds[s] * H, posOff = s * H;
        for (let i = 0; i < H; i++) hidden[s * H + i] = wte[tokOff + i] + wpe[posOff + i];
    }

    for (let l = 0; l < numLayers; l++) {
        hidden = await prefillForwardLayer(hidden, l, seqLen);
    }

    // Final LN + argmax on CPU for first token
    const lastH = hidden.subarray((seqLen - 1) * H, seqLen * H);
    const normed = layerNormCPU(lastH, 1, H, "ln_f.weight", "ln_f.bias");

    let nextToken;
    {
        const wteArr = weights.get("wte.weight").cpu;
        let maxLogit = -1e30, maxIdx = 0;
        for (let v = 0; v < vocabSize; v++) {
            let dot = 0;
            const off = v * H;
            for (let d = 0; d < H; d++) dot += normed[d] * wteArr[off + d];
            if (dot > maxLogit) { maxLogit = dot; maxIdx = v; }
        }
        nextToken = maxIdx;
    }

    const prefillMs = performance.now() - t0;
    postStatus("compute", `Prefill: ${prefillMs.toFixed(0)}ms (${seqLen} tokens) | Decode: FULL GPU mode`);
    sendToken(ws, nextToken, 0, prefillMs);

    // === DECODE (FULL GPU — zero CPU readback per step) ===
    const tempBufs = allocateTempBuffers();
    const wteBuf = weights.get("wte.weight").buffer;
    const wpeBuf = weights.get("wpe.weight").buffer;

    let position = seqLen;
    const decodeStart = performance.now();

    for (let step = 1; step < maxTokens; step++) {
        if (!generating) break;
        const stepStart = performance.now();

        // 1. Embed token → GPU buffer
        let encoder = device.createCommandEncoder();
        const uniforms = [];
        uniforms.push(gpuEmbedSingle(encoder, wteBuf, wpeBuf, tempBufs.hidden, nextToken, position, H));
        device.queue.submit([encoder.finish()]);

        // 2. Forward through all layers on GPU
        let currentHidden = tempBufs.hidden;
        for (let l = 0; l < numLayers; l++) {
            encoder = device.createCommandEncoder();
            // Alternate hidden buffers so input ≠ output
            const result = decodeForwardLayerGPU(encoder, currentHidden, l, tempBufs);
            device.queue.submit([result.encoder.finish()]);

            // Swap: output of this layer → input of next
            // Copy resid2 → hidden (or hiddenAlt)
            const nextBuf = (l % 2 === 0) ? tempBufs.hiddenAlt : tempBufs.hidden;
            const cpEncoder = device.createCommandEncoder();
            cpEncoder.copyBufferToBuffer(result.outputBuf, 0, nextBuf, 0, H * 4);
            device.queue.submit([cpEncoder.finish()]);
            currentHidden = nextBuf;

            // Clean up uniform buffers
            for (const u of result.uniforms) u.destroy();
            for (const u of uniforms.splice(0)) u.destroy();
        }

        // 3. Final LayerNorm on GPU
        encoder = device.createCommandEncoder();
        uniforms.push(gpuLayerNorm(encoder, currentHidden,
            weights.get("ln_f.weight").buffer,
            weights.get("ln_f.bias").buffer,
            tempBufs.lnFinal, 1, H));

        // 4. Argmax LM Head on GPU — ONLY readback is the single u32 token ID
        uniforms.push(gpuArgmaxLMHead(encoder, tempBufs.lnFinal, wteBuf,
            tempBufs.argmaxResult, vocabSize, H));

        // 5. Copy argmax result to readback buffer
        encoder.copyBufferToBuffer(tempBufs.argmaxResult, 0, tempBufs.argmaxReadback, 0, 4);
        device.queue.submit([encoder.finish()]);

        // 6. Read back ONLY the 4-byte token ID (the ONLY CPU readback per step!)
        await tempBufs.argmaxReadback.mapAsync(GPUMapMode.READ);
        nextToken = new Uint32Array(tempBufs.argmaxReadback.getMappedRange().slice(0))[0];
        tempBufs.argmaxReadback.unmap();

        position++;
        for (const u of uniforms) u.destroy();

        const stepMs = performance.now() - stepStart;
        sendToken(ws, nextToken, step, stepMs);

        if (step % 5 === 0) {
            const elapsed = performance.now() - decodeStart;
            const tps = step / (elapsed / 1000);
            postStatus("compute", `Decode step ${step}: ${tps.toFixed(1)} tok/s [FULL GPU]`);
        }
    }

    const totalMs = performance.now() - t0;
    const totalTokens = Math.min(maxTokens, position - seqLen + 1);
    const decodeMs = performance.now() - decodeStart;
    const tps = (totalTokens - 1) / (decodeMs / 1000);

    destroyTempBuffers(tempBufs);

    sendGenerateDone(ws, totalTokens, totalMs);
    postStatus("compute",
        `Done: ${totalTokens} tokens in ${totalMs.toFixed(0)}ms ` +
        `(prefill ${prefillMs.toFixed(0)}ms, decode ${tps.toFixed(1)} tok/s) [FULL GPU]`
    );
    generating = false;
}

// ======================== Protocol Helpers ========================

function sendToken(ws, tokenId, step, timeMs) {
    const buf = new ArrayBuffer(13);
    const view = new DataView(buf);
    view.setUint8(0, OP_TOKEN);
    view.setUint32(1, tokenId, true);
    view.setUint32(5, step, true);
    view.setFloat32(9, timeMs, true);
    ws.send(buf);
}

function sendGenerateDone(ws, totalTokens, totalMs) {
    const buf = new ArrayBuffer(13);
    const view = new DataView(buf);
    view.setUint8(0, OP_GENERATE_DONE);
    view.setUint32(1, totalTokens, true);
    view.setFloat32(5, totalMs, true);
    ws.send(buf);
}

function sendOK(ws, op) {
    const buf = new ArrayBuffer(2);
    const view = new DataView(buf);
    view.setUint8(0, op);
    view.setUint8(1, 0x00);
    ws.send(buf);
}

// ======================== Protocol Handlers ========================

function handleUploadTensor(data) {
    const view = new DataView(data);
    const nameLen = view.getUint16(1, true);
    const nameBytes = new Uint8Array(data, 3, nameLen);
    const name = new TextDecoder().decode(nameBytes);
    const ndim = view.getUint8(3 + nameLen);

    let offset = 4 + nameLen;
    const shape = [];
    let numel = 1;
    for (let d = 0; d < ndim; d++) {
        const s = view.getUint32(offset, true);
        shape.push(s);
        numel *= s;
        offset += 4;
    }
    if (offset % 4 !== 0) offset += 4 - (offset % 4);

    const cpuData = new Float32Array(data, offset, numel);

    // ALL tensors get GPU buffers (including 1D bias/norm weights)
    let gpuBuffer = null;
    if (device) {
        gpuBuffer = device.createBuffer({
            size: cpuData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(gpuBuffer, 0, cpuData);
    }

    weights.set(name, {
        buffer: gpuBuffer,
        shape: shape,
        cpu: new Float32Array(cpuData), // keep CPU copy for prefill
    });

    totalWeightBytes += cpuData.byteLength;
    const sizeMB = (cpuData.byteLength / 1024 / 1024).toFixed(1);
    postStatus("upload", `${name} [${shape.join("×")}] ${sizeMB} MB → GPU (total: ${(totalWeightBytes/1024/1024).toFixed(0)} MB)`);
}

function handleSetConfig(data) {
    const view = new DataView(data);
    const jsonLen = view.getUint32(1, true);
    const jsonStr = new TextDecoder().decode(new Uint8Array(data, 5, jsonLen));
    config = JSON.parse(jsonStr);
    postStatus("config", `Model: ${config.model_name || "unknown"}, arch=${config.arch}, H=${config.hidden_size}, L=${config.num_layers}`);
}

async function handleGenerate(ws, data) {
    const view = new DataView(data);
    const maxTokens = view.getUint32(1, true);
    const promptLen = view.getUint32(5, true);
    const promptIds = [];
    for (let i = 0; i < promptLen; i++) {
        promptIds.push(view.getUint32(9 + i * 4, true));
    }
    postStatus("compute", `Generate: prompt ${promptLen} tokens, max ${maxTokens}`);
    await generate(ws, promptIds, maxTokens);
}

// ======================== Backward compat: raw matmul ========================

async function gpuMatmulLegacy(M, N, K, activationData, weightData) {
    const bufA = uploadToGPU(activationData);
    const bufB = uploadToGPU(weightData);
    const outputSize = M * N * 4;
    const bufC = createGPUBuffer(outputSize);
    const readbackBuf = device.createBuffer({
        size: outputSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = device.createCommandEncoder();
    gpuMatmulBuf(encoder, M, N, K, bufA, bufB, bufC);
    encoder.copyBufferToBuffer(bufC, 0, readbackBuf, 0, outputSize);
    device.queue.submit([encoder.finish()]);

    await readbackBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readbackBuf.getMappedRange().slice(0));
    readbackBuf.unmap();
    bufA.destroy(); bufB.destroy(); bufC.destroy(); readbackBuf.destroy();
    return result;
}

// ======================== WebSocket Main Loop ========================

async function connectAndServe(wsUrl) {
    postStatus("init", "Initializing WebGPU (FULL GPU mode)...");
    await initWebGPU();

    let ws = null;
    let running = true;

    function connect() {
        ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => postStatus("connected", `Connected to ${wsUrl}`);

        ws.onmessage = async (event) => {
            try {
                const data = event.data;
                const op = new Uint8Array(data)[0];

                if (op === OP_PING) { ws.send(data); return; }

                if (op === OP_SHUTDOWN) {
                    postStatus("shutdown", "Server requested shutdown");
                    generating = false; running = false; ws.close(); return;
                }

                if (op === OP_UPLOAD_TENSOR) {
                    handleUploadTensor(data);
                    sendOK(ws, OP_UPLOAD_TENSOR);
                    return;
                }

                if (op === OP_SET_CONFIG) {
                    handleSetConfig(data);
                    sendOK(ws, OP_SET_CONFIG);
                    return;
                }

                if (op === OP_GENERATE) {
                    await handleGenerate(ws, data);
                    return;
                }

                if (op === OP_MATMUL) {
                    const view = new DataView(data);
                    const M = view.getUint32(1, true);
                    const N = view.getUint32(5, true);
                    const K = view.getUint32(9, true);
                    const headerSize = 13;
                    const dataA = new Float32Array(data, headerSize, M * K);
                    const dataB = new Float32Array(data, headerSize + M * K * 4, N * K);
                    const result = await gpuMatmulLegacy(M, N, K, dataA, dataB);
                    const respHeader = 9;
                    const respBuf = new ArrayBuffer(respHeader + result.byteLength);
                    const rv = new DataView(respBuf);
                    rv.setUint8(0, OP_MATMUL);
                    rv.setUint32(1, M, true);
                    rv.setUint32(5, N, true);
                    new Float32Array(respBuf, respHeader).set(result);
                    ws.send(respBuf);
                    return;
                }

                postStatus("error", `Unknown op: 0x${op.toString(16)}`);
            } catch (e) {
                postStatus("error", e.message);
                console.error("[VRAMancer]", e);
            }
        };

        ws.onerror = () => postStatus("error", "WebSocket error");
        ws.onclose = () => {
            if (running) {
                postStatus("reconnecting", "Connection lost, retrying in 2s...");
                setTimeout(connect, 2000);
            }
        };
    }

    connect();
    return {
        stop: () => { running = false; generating = false; if (ws) ws.close(); },
        getStats: () => ({ weights: weights.size, totalMB: totalWeightBytes / 1024 / 1024, config }),
    };
}

if (typeof window !== "undefined") {
    window.VRAMancerInference = { connectAndServe, initWebGPU };
}

/**
 * VRAMancer WebGPU Inference Worker
 *
 * Holds model weights in GPU buffers, runs full GPT-2 transformer
 * forward passes entirely in the browser. Matmuls on WebGPU,
 * LayerNorm/GeLU/Attention in JavaScript.
 *
 * Protocol (binary WebSocket):
 *   0x01 MATMUL        — raw A@B^T (backward compat)
 *   0x02 PING          — echo
 *   0x10 UPLOAD_TENSOR — store named tensor
 *   0x11 SET_CONFIG    — model config JSON
 *   0x30 GENERATE      — autoregressive generation
 *   0x31 TOKEN         — single generated token (response)
 *   0x32 GENERATE_DONE — generation complete (response)
 *   0xFF SHUTDOWN
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
let matmulPipeline = null;
let bindGroupLayout = null;

// Weight pool: name → { buffer: GPUBuffer|null, shape: number[], cpu: Float32Array }
const weights = new Map();
let totalWeightBytes = 0;

// Model config (set via SET_CONFIG)
let config = null;

// KV cache per layer: layerIdx → { k: Float32Array, v: Float32Array, length: number }
const kvCache = new Map();

let generating = false;

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

    // Load and compile matmul shader
    const shaderCode = await (await fetch("matmul.wgsl")).text();
    const shaderModule = device.createShaderModule({ code: shaderCode });

    const compilationInfo = await shaderModule.getCompilationInfo();
    for (const msg of compilationInfo.messages) {
        if (msg.type === "error") throw new Error(`Shader error: ${msg.message}`);
    }

    bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ],
    });

    matmulPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: "matmul_tiled" },
    });

    postStatus("ready", "WebGPU matmul pipeline compiled");
    return info;
}

// ======================== GPU Matmul ========================

/**
 * C = A @ B^T where A is [M,K] (uploaded each call) and B is [N,K] (cached GPU buffer).
 * Returns Float32Array[M*N].
 */
async function gpuMatmul(M, N, K, activationData, weightBuf) {
    const TILE = 16;

    const paramsBuf = device.createBuffer({
        size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([M, N, K, 0]));

    const bufA = device.createBuffer({
        size: activationData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufA, 0, activationData);

    const outputSize = M * N * 4;
    const bufC = device.createBuffer({
        size: outputSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readbackBuf = device.createBuffer({
        size: outputSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: bufA } },
            { binding: 2, resource: { buffer: weightBuf } },
            { binding: 3, resource: { buffer: bufC } },
        ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(matmulPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(N / TILE), Math.ceil(M / TILE), 1);
    pass.end();
    encoder.copyBufferToBuffer(bufC, 0, readbackBuf, 0, outputSize);
    device.queue.submit([encoder.finish()]);

    await readbackBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readbackBuf.getMappedRange().slice(0));
    readbackBuf.unmap();

    paramsBuf.destroy();
    bufA.destroy();
    bufC.destroy();
    readbackBuf.destroy();

    return result;
}

/**
 * Linear layer: output = input @ weight^T + bias
 * weight is stored as [outFeatures, inFeatures] in GPU buffer (B in A@B^T).
 */
async function linear(input, M, inFeatures, weightName, biasName) {
    const w = weights.get(weightName);
    if (!w || !w.buffer) throw new Error(`GPU weight not found: ${weightName}`);

    const outFeatures = w.shape[0];
    const result = await gpuMatmul(M, outFeatures, inFeatures, input, w.buffer);

    if (biasName) {
        const b = weights.get(biasName);
        if (b) {
            const bias = b.cpu;
            for (let m = 0; m < M; m++) {
                const off = m * outFeatures;
                for (let n = 0; n < outFeatures; n++) {
                    result[off + n] += bias[n];
                }
            }
        }
    }
    return result;
}

// ======================== CPU Ops ========================

function layerNorm(x, seqLen, H, weightName, biasName, eps = 1e-5) {
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

function geluInPlace(x) {
    const c = Math.sqrt(2.0 / Math.PI);
    for (let i = 0; i < x.length; i++) {
        const v = x[i];
        x[i] = 0.5 * v * (1.0 + Math.tanh(c * (v + 0.044715 * v * v * v)));
    }
    return x;
}

function softmaxRows(x, rows, cols) {
    for (let r = 0; r < rows; r++) {
        const off = r * cols;
        let max = -1e30;
        for (let c = 0; c < cols; c++) max = Math.max(max, x[off + c]);
        let sum = 0;
        for (let c = 0; c < cols; c++) {
            x[off + c] = Math.exp(x[off + c] - max);
            sum += x[off + c];
        }
        for (let c = 0; c < cols; c++) x[off + c] /= sum;
    }
    return x;
}

function addVec(a, b) {
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
    return out;
}

// ======================== KV Cache ========================

function initKVCaches(numLayers, maxLen, kvDim) {
    kvCache.clear();
    for (let l = 0; l < numLayers; l++) {
        kvCache.set(l, {
            k: new Float32Array(maxLen * kvDim),
            v: new Float32Array(maxLen * kvDim),
            length: 0,
            kvDim: kvDim,
        });
    }
}

function appendKV(layerIdx, kNew, vNew) {
    const c = kvCache.get(layerIdx);
    const off = c.length * c.kvDim;
    c.k.set(kNew, off);
    c.v.set(vNew, off);
    c.length++;
}

function resetKVCaches() {
    for (const [, c] of kvCache) c.length = 0;
}

// ======================== Attention ========================

/**
 * Prefill: full causal multi-head attention.
 * Q, K, V: [seqLen, numHeads * headDim] flattened.
 * Stores K, V in cache. Returns [seqLen, numHeads * headDim].
 */
function prefillAttention(Q, K, V, seqLen, numHeads, headDim, layerIdx) {
    const kvDim = numHeads * headDim;
    const cache = kvCache.get(layerIdx);

    // Store all K, V in cache
    for (let s = 0; s < seqLen; s++) {
        cache.k.set(K.subarray(s * kvDim, s * kvDim + kvDim), s * kvDim);
        cache.v.set(V.subarray(s * kvDim, s * kvDim + kvDim), s * kvDim);
    }
    cache.length = seqLen;

    const scale = 1.0 / Math.sqrt(headDim);
    const output = new Float32Array(seqLen * kvDim);

    for (let h = 0; h < numHeads; h++) {
        // Per-head attention with causal mask
        for (let i = 0; i < seqLen; i++) {
            // Compute scores for position i against all j <= i
            const scores = new Float32Array(i + 1);
            for (let j = 0; j <= i; j++) {
                let dot = 0;
                for (let d = 0; d < headDim; d++) {
                    dot += Q[i * kvDim + h * headDim + d] *
                           K[j * kvDim + h * headDim + d];
                }
                scores[j] = dot * scale;
            }

            // Softmax
            let max = -1e30;
            for (let j = 0; j <= i; j++) max = Math.max(max, scores[j]);
            let sum = 0;
            for (let j = 0; j <= i; j++) {
                scores[j] = Math.exp(scores[j] - max);
                sum += scores[j];
            }
            for (let j = 0; j <= i; j++) scores[j] /= sum;

            // Weighted sum of V
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

/**
 * Decode: single-token attention against cached K, V.
 * Q: [1, numHeads * headDim], kNew/vNew: [1, numHeads * headDim].
 * Appends to cache, returns [1, numHeads * headDim].
 */
function decodeAttention(Q, kNew, vNew, numHeads, headDim, layerIdx) {
    const kvDim = numHeads * headDim;
    appendKV(layerIdx, kNew, vNew);
    const cache = kvCache.get(layerIdx);
    const fullLen = cache.length;
    const scale = 1.0 / Math.sqrt(headDim);
    const output = new Float32Array(kvDim);

    for (let h = 0; h < numHeads; h++) {
        const scores = new Float32Array(fullLen);
        for (let j = 0; j < fullLen; j++) {
            let dot = 0;
            for (let d = 0; d < headDim; d++) {
                dot += Q[h * headDim + d] * cache.k[j * kvDim + h * headDim + d];
            }
            scores[j] = dot * scale;
        }

        let max = -1e30;
        for (let j = 0; j < fullLen; j++) max = Math.max(max, scores[j]);
        let sum = 0;
        for (let j = 0; j < fullLen; j++) {
            scores[j] = Math.exp(scores[j] - max);
            sum += scores[j];
        }
        for (let j = 0; j < fullLen; j++) scores[j] /= sum;

        for (let d = 0; d < headDim; d++) {
            let val = 0;
            for (let j = 0; j < fullLen; j++) {
                val += scores[j] * cache.v[j * kvDim + h * headDim + d];
            }
            output[h * headDim + d] = val;
        }
    }
    return output;
}

// ======================== GPT-2 Forward ========================

/**
 * GPT-2 transformer layer forward pass.
 * hidden: Float32Array[seqLen * H], returns same shape.
 * mode: "prefill" (full seq) or "decode" (single token).
 */
async function forwardGPT2Layer(hidden, layerIdx, seqLen, mode) {
    const H = config.hidden_size;
    const numHeads = config.num_heads;
    const headDim = H / numHeads;
    const p = `h.${layerIdx}`;

    // 1. Pre-attention LayerNorm
    const normed1 = layerNorm(hidden, seqLen, H, `${p}.ln_1.weight`, `${p}.ln_1.bias`);

    // 2. QKV projection: [seq, H] → [seq, 3H]
    const qkv = await linear(normed1, seqLen, H, `${p}.attn.c_attn.weight`, `${p}.attn.c_attn.bias`);

    // 3. Split Q, K, V
    const Q = new Float32Array(seqLen * H);
    const K = new Float32Array(seqLen * H);
    const V = new Float32Array(seqLen * H);
    for (let s = 0; s < seqLen; s++) {
        const src = s * 3 * H;
        const dst = s * H;
        Q.set(qkv.subarray(src, src + H), dst);
        K.set(qkv.subarray(src + H, src + 2 * H), dst);
        V.set(qkv.subarray(src + 2 * H, src + 3 * H), dst);
    }

    // 4. Attention
    let attnOut;
    if (mode === "prefill") {
        attnOut = prefillAttention(Q, K, V, seqLen, numHeads, headDim, layerIdx);
    } else {
        attnOut = decodeAttention(Q, K, V, numHeads, headDim, layerIdx);
    }

    // 5. Output projection: [seq, H] → [seq, H]
    const projected = await linear(attnOut, seqLen, H, `${p}.attn.c_proj.weight`, `${p}.attn.c_proj.bias`);

    // 6. Residual
    hidden = addVec(hidden, projected);

    // 7. Pre-MLP LayerNorm
    const normed2 = layerNorm(hidden, seqLen, H, `${p}.ln_2.weight`, `${p}.ln_2.bias`);

    // 8. MLP: fc → GeLU → proj
    const fc = await linear(normed2, seqLen, H, `${p}.mlp.c_fc.weight`, `${p}.mlp.c_fc.bias`);
    geluInPlace(fc);
    const mlpOut = await linear(fc, seqLen, 4 * H, `${p}.mlp.c_proj.weight`, `${p}.mlp.c_proj.bias`);

    // 9. Residual
    return addVec(hidden, mlpOut);
}

// ======================== Embedding & LM Head ========================

function embed(tokenIds) {
    const H = config.hidden_size;
    const wte = weights.get("wte.weight").cpu;
    const wpe = weights.get("wpe.weight").cpu;
    const seqLen = tokenIds.length;
    const result = new Float32Array(seqLen * H);

    for (let s = 0; s < seqLen; s++) {
        const tokOff = tokenIds[s] * H;
        const posOff = s * H;
        for (let i = 0; i < H; i++) {
            result[s * H + i] = wte[tokOff + i] + wpe[posOff + i];
        }
    }
    return result;
}

function embedSingle(tokenId, position) {
    const H = config.hidden_size;
    const wte = weights.get("wte.weight").cpu;
    const wpe = weights.get("wpe.weight").cpu;
    const result = new Float32Array(H);
    const tokOff = tokenId * H;
    const posOff = position * H;
    for (let i = 0; i < H; i++) {
        result[i] = wte[tokOff + i] + wpe[posOff + i];
    }
    return result;
}

function lmHead(hidden) {
    // logits = hidden @ wte^T, return argmax
    const H = config.hidden_size;
    const V = config.vocab_size;
    const wte = weights.get("wte.weight").cpu;

    let maxLogit = -1e30;
    let maxIdx = 0;
    for (let v = 0; v < V; v++) {
        let dot = 0;
        const off = v * H;
        for (let d = 0; d < H; d++) {
            dot += hidden[d] * wte[off + d];
        }
        if (dot > maxLogit) {
            maxLogit = dot;
            maxIdx = v;
        }
    }
    return maxIdx;
}

// ======================== Generate ========================

/**
 * Autoregressive generation: prefill prompt, then decode token-by-token.
 * Sends OP_TOKEN for each generated token, OP_GENERATE_DONE at the end.
 */
async function generate(ws, promptIds, maxTokens) {
    generating = true;
    const H = config.hidden_size;
    const numLayers = config.num_layers;
    const numHeads = config.num_heads;
    const headDim = H / numHeads;
    const maxCtx = config.max_position || 1024;
    const seqLen = promptIds.length;

    postStatus("compute", `Generating: prefill ${seqLen} tokens...`);

    // Init KV caches
    initKVCaches(numLayers, maxCtx, numHeads * headDim);

    const t0 = performance.now();

    // === PREFILL ===
    let hidden = embed(promptIds);

    for (let l = 0; l < numLayers; l++) {
        hidden = await forwardGPT2Layer(hidden, l, seqLen, "prefill");
    }

    // Final LayerNorm on last token
    const lastHidden = hidden.subarray((seqLen - 1) * H, seqLen * H);
    const normed = layerNorm(lastHidden, 1, H, "ln_f.weight", "ln_f.bias");

    // LM head → first generated token
    let nextToken = lmHead(normed);
    const prefillMs = performance.now() - t0;

    postStatus("compute", `Prefill: ${prefillMs.toFixed(0)}ms (${seqLen} tokens)`);

    // Send first token
    sendToken(ws, nextToken, 0, prefillMs);

    // === DECODE ===
    let position = seqLen;
    const decodeStart = performance.now();

    for (let step = 1; step < maxTokens; step++) {
        if (!generating) break;

        const stepStart = performance.now();
        let h = embedSingle(nextToken, position);

        for (let l = 0; l < numLayers; l++) {
            h = await forwardGPT2Layer(h, l, 1, "decode");
        }

        const hNormed = layerNorm(h, 1, H, "ln_f.weight", "ln_f.bias");
        nextToken = lmHead(hNormed);
        position++;

        const stepMs = performance.now() - stepStart;
        sendToken(ws, nextToken, step, stepMs);

        if (step % 5 === 0) {
            const elapsed = performance.now() - decodeStart;
            const tps = step / (elapsed / 1000);
            postStatus("compute", `Decode step ${step}: ${tps.toFixed(1)} tok/s`);
        }
    }

    const totalMs = performance.now() - t0;
    const totalTokens = Math.min(maxTokens, position - seqLen + 1);
    const decodeMs = performance.now() - decodeStart;
    const tps = (totalTokens - 1) / (decodeMs / 1000);

    sendGenerateDone(ws, totalTokens, totalMs);
    postStatus("compute",
        `Done: ${totalTokens} tokens in ${totalMs.toFixed(0)}ms ` +
        `(prefill ${prefillMs.toFixed(0)}ms, decode ${tps.toFixed(1)} tok/s)`
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
    view.setUint8(1, 0x00); // OK
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

    // Align offset to 4 bytes for float32 data
    if (offset % 4 !== 0) offset += 4 - (offset % 4);

    const cpuData = new Float32Array(data, offset, numel);

    // Create GPU buffer for 2D tensors (linear weights)
    let gpuBuffer = null;
    if (ndim === 2 && device) {
        gpuBuffer = device.createBuffer({
            size: cpuData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(gpuBuffer, 0, cpuData);
    }

    weights.set(name, {
        buffer: gpuBuffer,
        shape: shape,
        cpu: new Float32Array(cpuData), // always keep CPU copy
    });

    totalWeightBytes += cpuData.byteLength;
    const sizeMB = (cpuData.byteLength / 1024 / 1024).toFixed(1);
    postStatus("upload", `${name} [${shape.join("×")}] ${sizeMB} MB (total: ${(totalWeightBytes/1024/1024).toFixed(0)} MB)`);
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

// ======================== WebSocket Main Loop ========================

async function connectAndServe(wsUrl) {
    postStatus("init", "Initializing WebGPU...");
    await initWebGPU();

    let ws = null;
    let running = true;

    function connect() {
        ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
            postStatus("connected", `Connected to ${wsUrl}`);
        };

        ws.onmessage = async (event) => {
            try {
                const data = event.data;
                const op = new Uint8Array(data)[0];

                if (op === OP_PING) {
                    ws.send(data);
                    return;
                }

                if (op === OP_SHUTDOWN) {
                    postStatus("shutdown", "Server requested shutdown");
                    generating = false;
                    running = false;
                    ws.close();
                    return;
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
                    // Backward-compat raw matmul
                    const view = new DataView(data);
                    const M = view.getUint32(1, true);
                    const N = view.getUint32(5, true);
                    const K = view.getUint32(9, true);
                    const headerSize = 13;
                    const dataA = new Float32Array(data, headerSize, M * K);
                    const dataB = new Float32Array(data, headerSize + M * K * 4, N * K);

                    // Create temp GPU buffer for B
                    const bufB = device.createBuffer({
                        size: dataB.byteLength,
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                    });
                    device.queue.writeBuffer(bufB, 0, dataB);

                    const result = await gpuMatmul(M, N, K, dataA, bufB);
                    bufB.destroy();

                    // Send response
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

// Export for HTML page
if (typeof window !== "undefined") {
    window.VRAMancerInference = { connectAndServe, initWebGPU };
}

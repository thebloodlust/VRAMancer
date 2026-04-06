/**
 * VRAMancer WebNN Inference Worker — "WebNPU"
 *
 * Uses the W3C WebNN API to route transformer inference to the best
 * available accelerator: NPU → GPU → CPU.
 *
 * Target devices:
 *   Apple (iPad/iPhone/Mac) : CoreML → Apple Neural Engine (38 TOPS on M4)
 *   Qualcomm (Android/Win)  : QNN → Hexagon NPU
 *   Intel                   : OpenVINO → Intel NPU / GPU
 *   Windows                 : DirectML → NPU / GPU
 *
 * Architecture: Pre-compiled WebNN graphs per transformer layer.
 *   • LN + QKV matmul  → compiled graph (runs on NPU)
 *   • Attention         → JS (dynamic KV length, tiny compute)
 *   • Output projection → compiled graph (runs on NPU)
 *   • LN + MLP (FC+GELU+Proj) → compiled graph (runs on NPU)
 *   • Final LN + LM Head      → compiled graph (runs on NPU)
 *
 * When WebNN is unavailable, the HTML page loads the WebGPU worker instead.
 *
 * Same binary WebSocket protocol as the WebGPU worker:
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

let mlContext = null;
let backendInfo = "";

// Weight pool: name → { data: Float32Array, shape: number[] }
const weights = new Map();
let totalWeightBytes = 0;

let config = null;
let generating = false;

// Pre-compiled WebNN graphs per layer
const layerGraphs = []; // index → { attn: MLGraph, proj: MLGraph, mlp: MLGraph }
let finalLNGraph = null;
let lmHeadGraph = null;

// KV Cache (JS Float32Array — managed in JS for dynamic length)
const kvCacheK = []; // layerIdx → Float32Array[maxLen * H]
const kvCacheV = []; // layerIdx → Float32Array[maxLen * H]
const kvLen = [];    // layerIdx → current length

// Pre-allocated MLTensor pool
let tensorPool = {};
let lastHiddenState = null;

// ======================== Status ========================

function postStatus(status, message) {
    if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent("vramancer-status", {
            detail: { status, message, timestamp: Date.now() },
        }));
    }
    console.log(`[WebNPU] ${status}: ${message}`);
}

// ======================== WebNN Setup ========================

async function initWebNN() {
    if (!navigator.ml) {
        throw new Error("WebNN not supported in this browser");
    }

    // Try accelerators in priority order: NPU → GPU → CPU
    const deviceTypes = ["npu", "gpu", "cpu"];
    for (const deviceType of deviceTypes) {
        try {
            mlContext = await navigator.ml.createContext({ deviceType });
            backendInfo = `WebNN-${deviceType.toUpperCase()}`;
            postStatus("init", `Accelerator: ${backendInfo}`);
            return { backend: backendInfo, deviceType };
        } catch (e) {
            postStatus("init", `WebNN ${deviceType}: ${e.message}`);
        }
    }
    throw new Error("WebNN: no accelerator available");
}

// ======================== Graph Building ========================

/**
 * Build LN1 + QKV projection graph for a single transformer layer.
 * Input: hidden [1, H] → Output: qkv [1, 3H]
 */
async function buildAttnQKVGraph(layerIdx) {
    const H = config.hidden_size;
    const H3 = 3 * H;
    const p = `h.${layerIdx}`;
    const builder = new MLGraphBuilder(mlContext);

    const input = builder.input("input", { dataType: "float32", shape: [1, H] });

    // LayerNorm 1
    const lnW = builder.constant(
        { dataType: "float32", shape: [H] },
        weights.get(`${p}.ln_1.weight`).data
    );
    const lnB = builder.constant(
        { dataType: "float32", shape: [H] },
        weights.get(`${p}.ln_1.bias`).data
    );
    const normed = builder.layerNormalization(input, {
        scale: lnW, bias: lnB, axes: [1], epsilon: 1e-5,
    });

    // QKV matmul: normed[1,H] × W^T[H,3H] → [1,3H]
    // Server sends weight as [3H, H] (transposed Conv1D)
    const W = builder.constant(
        { dataType: "float32", shape: [H3, H] },
        weights.get(`${p}.attn.c_attn.weight`).data
    );
    const WT = builder.transpose(W, { permutation: [1, 0] });
    const mm = builder.matmul(normed, WT);

    // Bias
    const bias = builder.constant(
        { dataType: "float32", shape: [H3] },
        weights.get(`${p}.attn.c_attn.bias`).data
    );
    const biasR = builder.reshape(bias, [1, H3]);
    const output = builder.add(mm, biasR);

    return await builder.build({ "output": output });
}

/**
 * Build attention output projection graph.
 * Input: attn_output [1, H] → Output: projected [1, H]
 */
async function buildProjGraph(layerIdx) {
    const H = config.hidden_size;
    const p = `h.${layerIdx}`;
    const builder = new MLGraphBuilder(mlContext);

    const input = builder.input("input", { dataType: "float32", shape: [1, H] });

    const W = builder.constant(
        { dataType: "float32", shape: [H, H] },
        weights.get(`${p}.attn.c_proj.weight`).data
    );
    const WT = builder.transpose(W, { permutation: [1, 0] });
    const mm = builder.matmul(input, WT);

    const bias = builder.constant(
        { dataType: "float32", shape: [H] },
        weights.get(`${p}.attn.c_proj.bias`).data
    );
    const biasR = builder.reshape(bias, [1, H]);
    const output = builder.add(mm, biasR);

    return await builder.build({ "output": output });
}

/**
 * Build composite MLP graph: LN2 → FC → GELU → Proj.
 * All in one graph = one NPU dispatch per layer.
 * Input: residual [1, H] → Output: mlp_out [1, H]
 */
async function buildMLPGraph(layerIdx) {
    const H = config.hidden_size;
    const H4 = 4 * H;
    const p = `h.${layerIdx}`;
    const builder = new MLGraphBuilder(mlContext);

    const input = builder.input("input", { dataType: "float32", shape: [1, H] });

    // LayerNorm 2
    const lnW = builder.constant(
        { dataType: "float32", shape: [H] },
        weights.get(`${p}.ln_2.weight`).data
    );
    const lnB = builder.constant(
        { dataType: "float32", shape: [H] },
        weights.get(`${p}.ln_2.bias`).data
    );
    const normed = builder.layerNormalization(input, {
        scale: lnW, bias: lnB, axes: [1], epsilon: 1e-5,
    });

    // FC: [1,H] → [1,4H]
    const Wfc = builder.constant(
        { dataType: "float32", shape: [H4, H] },
        weights.get(`${p}.mlp.c_fc.weight`).data
    );
    const WfcT = builder.transpose(Wfc, { permutation: [1, 0] });
    const fc = builder.matmul(normed, WfcT);
    const fcBias = builder.constant(
        { dataType: "float32", shape: [H4] },
        weights.get(`${p}.mlp.c_fc.bias`).data
    );
    const fcBiasR = builder.reshape(fcBias, [1, H4]);
    const fcOut = builder.add(fc, fcBiasR);

    // GELU activation
    const activated = builder.gelu(fcOut);

    // Proj: [1,4H] → [1,H]
    const Wp = builder.constant(
        { dataType: "float32", shape: [H, H4] },
        weights.get(`${p}.mlp.c_proj.weight`).data
    );
    const WpT = builder.transpose(Wp, { permutation: [1, 0] });
    const proj = builder.matmul(activated, WpT);
    const projBias = builder.constant(
        { dataType: "float32", shape: [H] },
        weights.get(`${p}.mlp.c_proj.bias`).data
    );
    const projBiasR = builder.reshape(projBias, [1, H]);
    const output = builder.add(proj, projBiasR);

    return await builder.build({ "output": output });
}

/**
 * Build final LayerNorm graph.
 * Input: hidden [1, H] → Output: normed [1, H]
 */
async function buildFinalLNGraph() {
    const H = config.hidden_size;
    const builder = new MLGraphBuilder(mlContext);

    const input = builder.input("input", { dataType: "float32", shape: [1, H] });
    const lnW = builder.constant(
        { dataType: "float32", shape: [H] },
        weights.get("ln_f.weight").data
    );
    const lnB = builder.constant(
        { dataType: "float32", shape: [H] },
        weights.get("ln_f.bias").data
    );
    const output = builder.layerNormalization(input, {
        scale: lnW, bias: lnB, axes: [1], epsilon: 1e-5,
    });

    return await builder.build({ "output": output });
}

/**
 * Build LM head graph: hidden → logits.
 * Input: normed_hidden [1, H] → Output: logits [1, vocab_size]
 */
async function buildLMHeadGraph() {
    const H = config.hidden_size;
    const V = config.vocab_size;
    const builder = new MLGraphBuilder(mlContext);

    const input = builder.input("input", { dataType: "float32", shape: [1, H] });

    // Tied weights: wte [V, H] → transpose → [H, V]
    const wte = builder.constant(
        { dataType: "float32", shape: [V, H] },
        weights.get("wte.weight").data
    );
    const wteT = builder.transpose(wte, { permutation: [1, 0] });
    const logits = builder.matmul(input, wteT);

    return await builder.build({ "output": logits });
}

/**
 * Compile all WebNN graphs for the loaded model.
 * 3 graphs per layer (attn, proj, mlp) + finalLN + lmHead.
 */
async function buildAllGraphs() {
    const numLayers = config.num_layers;
    const t0 = performance.now();

    layerGraphs.length = 0;
    for (let l = 0; l < numLayers; l++) {
        const attn = await buildAttnQKVGraph(l);
        const proj = await buildProjGraph(l);
        const mlp = await buildMLPGraph(l);
        layerGraphs.push({ attn, proj, mlp });
        postStatus("init", `Layer ${l + 1}/${numLayers} compiled [${backendInfo}]`);
    }

    finalLNGraph = await buildFinalLNGraph();
    lmHeadGraph = await buildLMHeadGraph();

    const elapsed = performance.now() - t0;
    const total = numLayers * 3 + 2;
    postStatus("ready", `${total} WebNN graphs compiled in ${elapsed.toFixed(0)}ms [${backendInfo}]`);
}

// ======================== Tensor Pool ========================

async function allocateTensorPool() {
    const H = config.hidden_size;
    const H3 = 3 * H;
    const V = config.vocab_size;

    tensorPool = {
        inH: await mlContext.createTensor({
            dataType: "float32", shape: [1, H], readable: true, writable: true,
        }),
        outH: await mlContext.createTensor({
            dataType: "float32", shape: [1, H], readable: true, writable: true,
        }),
        out3H: await mlContext.createTensor({
            dataType: "float32", shape: [1, H3], readable: true, writable: true,
        }),
        outV: await mlContext.createTensor({
            dataType: "float32", shape: [1, V], readable: true, writable: true,
        }),
    };
}

function destroyTensorPool() {
    for (const [, t] of Object.entries(tensorPool)) {
        if (t && t.destroy) t.destroy();
    }
    tensorPool = {};
}

// ======================== KV Cache ========================

function initKVCaches(numLayers, maxLen, kvDim) {
    kvCacheK.length = 0;
    kvCacheV.length = 0;
    kvLen.length = 0;
    for (let l = 0; l < numLayers; l++) {
        kvCacheK.push(new Float32Array(maxLen * kvDim));
        kvCacheV.push(new Float32Array(maxLen * kvDim));
        kvLen.push(0);
    }
}

function resetKVCaches() {
    for (let l = 0; l < kvLen.length; l++) kvLen[l] = 0;
}

// ======================== Decode Attention (JS) ========================
// Single-query attention against KV cache.
// Runs in JS because KV length is dynamic (changes every step).
// Compute is tiny: O(nHeads × kvLen × headDim) per token.

function decodeAttentionCPU(Q, layerIdx) {
    const numHeads = config.num_heads;
    const headDim = config.hidden_size / numHeads;
    const H = config.hidden_size;
    const len = kvLen[layerIdx];
    const K = kvCacheK[layerIdx];
    const V = kvCacheV[layerIdx];
    const scale = 1.0 / Math.sqrt(headDim);
    const output = new Float32Array(H);

    for (let h = 0; h < numHeads; h++) {
        const hOff = h * headDim;
        const scores = new Float32Array(len);

        // Q · K^T
        for (let j = 0; j < len; j++) {
            let dot = 0;
            const kBase = j * H + hOff;
            for (let d = 0; d < headDim; d++) {
                dot += Q[hOff + d] * K[kBase + d];
            }
            scores[j] = dot * scale;
        }

        // Softmax
        let max = -1e30;
        for (let j = 0; j < len; j++) if (scores[j] > max) max = scores[j];
        let sum = 0;
        for (let j = 0; j < len; j++) {
            scores[j] = Math.exp(scores[j] - max);
            sum += scores[j];
        }
        for (let j = 0; j < len; j++) scores[j] /= sum;

        // Weighted sum of V
        for (let d = 0; d < headDim; d++) {
            let val = 0;
            for (let j = 0; j < len; j++) {
                val += scores[j] * V[j * H + hOff + d];
            }
            output[hOff + d] = val;
        }
    }
    return output;
}

// ======================== Forward Decode Step ========================

/**
 * Process one token through all layers via WebNN.
 * Per layer: 3 WebNN dispatches (attnQKV, proj, mlp) + 1 JS attention.
 * Updates lastHiddenState.
 */
async function forwardDecodeStep(embed) {
    const H = config.hidden_size;
    const numLayers = config.num_layers;
    let currentHidden = embed;

    for (let l = 0; l < numLayers; l++) {
        const lg = layerGraphs[l];

        // 1. LN1 + QKV projection (WebNN → NPU)
        mlContext.writeTensor(tensorPool.inH, currentHidden);
        mlContext.dispatch(lg.attn,
            { "input": tensorPool.inH },
            { "output": tensorPool.out3H }
        );
        const qkvBuf = await mlContext.readTensor(tensorPool.out3H);
        const qkv = new Float32Array(qkvBuf);

        // 2. Split Q, K, V
        const Q = qkv.subarray(0, H);
        const K = qkv.subarray(H, 2 * H);
        const V = qkv.subarray(2 * H, 3 * H);

        // 3. Append K, V to cache
        const offset = kvLen[l] * H;
        kvCacheK[l].set(K, offset);
        kvCacheV[l].set(V, offset);
        kvLen[l]++;

        // 4. Single-query decode attention (JS — dynamic KV length)
        const attnOut = decodeAttentionCPU(Q, l);

        // 5. Output projection (WebNN → NPU)
        mlContext.writeTensor(tensorPool.inH, attnOut);
        mlContext.dispatch(lg.proj,
            { "input": tensorPool.inH },
            { "output": tensorPool.outH }
        );
        const projBuf = await mlContext.readTensor(tensorPool.outH);
        const projected = new Float32Array(projBuf);

        // 6. Residual connection 1
        const residual = new Float32Array(H);
        for (let i = 0; i < H; i++) residual[i] = currentHidden[i] + projected[i];

        // 7. LN2 + MLP: FC → GELU → Proj (WebNN → NPU, single graph)
        mlContext.writeTensor(tensorPool.inH, residual);
        mlContext.dispatch(lg.mlp,
            { "input": tensorPool.inH },
            { "output": tensorPool.outH }
        );
        const mlpBuf = await mlContext.readTensor(tensorPool.outH);
        const mlpOut = new Float32Array(mlpBuf);

        // 8. Residual connection 2
        currentHidden = new Float32Array(H);
        for (let i = 0; i < H; i++) currentHidden[i] = residual[i] + mlpOut[i];
    }

    lastHiddenState = currentHidden;
}

// ======================== LM Head + Argmax ========================

async function lmHeadArgmax(hidden) {
    const V = config.vocab_size;

    // Final LayerNorm (WebNN → NPU)
    mlContext.writeTensor(tensorPool.inH, hidden);
    mlContext.dispatch(finalLNGraph,
        { "input": tensorPool.inH },
        { "output": tensorPool.outH }
    );
    const lnBuf = await mlContext.readTensor(tensorPool.outH);
    const lnOut = new Float32Array(lnBuf);

    // LM Head matmul: [1,H] × wte^T[H,V] → [1,V] (WebNN → NPU)
    mlContext.writeTensor(tensorPool.inH, lnOut);
    mlContext.dispatch(lmHeadGraph,
        { "input": tensorPool.inH },
        { "output": tensorPool.outV }
    );
    const logitsBuf = await mlContext.readTensor(tensorPool.outV);
    const logits = new Float32Array(logitsBuf);

    // Argmax (JS — sequential scan)
    let maxVal = -Infinity, maxIdx = 0;
    for (let v = 0; v < V; v++) {
        if (logits[v] > maxVal) { maxVal = logits[v]; maxIdx = v; }
    }
    return maxIdx;
}

// ======================== Generate ========================

async function generate(ws, promptIds, maxTokens) {
    generating = true;
    const H = config.hidden_size;
    const numLayers = config.num_layers;
    const vocabSize = config.vocab_size;
    const maxCtx = config.max_position || 1024;
    const seqLen = promptIds.length;

    // Compile WebNN graphs on first generate
    if (layerGraphs.length === 0) {
        postStatus("compute", `Compiling ${numLayers * 3 + 2} graphs for ${backendInfo}...`);
        await buildAllGraphs();
    }
    if (!tensorPool.inH) {
        await allocateTensorPool();
    }

    initKVCaches(numLayers, maxCtx, H);
    const t0 = performance.now();
    const wte = weights.get("wte.weight").data;
    const wpe = weights.get("wpe.weight").data;

    // === PREFILL: process prompt tokens sequentially via WebNN ===
    // Each token goes through all layers (builds KV cache as it goes).
    // Same compute as batch prefill, but sequential. NPU handles the matmuls.
    postStatus("compute", `Prefill: ${seqLen} tokens via ${backendInfo}...`);

    for (let s = 0; s < seqLen; s++) {
        if (!generating) break;
        const tid = promptIds[s];
        const embed = new Float32Array(H);
        for (let i = 0; i < H; i++) {
            embed[i] = wte[tid * H + i] + wpe[s * H + i];
        }
        await forwardDecodeStep(embed);
    }

    // First generated token
    let nextToken = await lmHeadArgmax(lastHiddenState);
    const prefillMs = performance.now() - t0;
    postStatus("compute",
        `Prefill: ${prefillMs.toFixed(0)}ms (${seqLen} tokens) [${backendInfo}]`
    );
    sendToken(ws, nextToken, 0, prefillMs);

    // === DECODE: autoregressive generation via WebNN ===
    let position = seqLen;
    const decodeStart = performance.now();

    for (let step = 1; step < maxTokens; step++) {
        if (!generating) break;
        const stepStart = performance.now();

        // 1. Embed token
        const embed = new Float32Array(H);
        for (let i = 0; i < H; i++) {
            embed[i] = wte[nextToken * H + i] + wpe[position * H + i];
        }

        // 2. Full forward pass (WebNN NPU)
        await forwardDecodeStep(embed);

        // 3. LM head + argmax
        nextToken = await lmHeadArgmax(lastHiddenState);
        position++;

        const stepMs = performance.now() - stepStart;
        sendToken(ws, nextToken, step, stepMs);

        if (step % 5 === 0) {
            const elapsed = performance.now() - decodeStart;
            const tps = step / (elapsed / 1000);
            postStatus("compute",
                `Decode step ${step}: ${tps.toFixed(1)} tok/s [${backendInfo}]`
            );
        }
    }

    const totalMs = performance.now() - t0;
    const totalTokens = Math.min(maxTokens, position - seqLen + 1);
    const decodeMs = performance.now() - decodeStart;
    const tps = (totalTokens - 1) / (decodeMs / 1000);

    sendGenerateDone(ws, totalTokens, totalMs);
    postStatus("compute",
        `Done: ${totalTokens} tokens in ${totalMs.toFixed(0)}ms ` +
        `(prefill ${prefillMs.toFixed(0)}ms, decode ${tps.toFixed(1)} tok/s) [${backendInfo}]`
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

    // Store CPU-side (WebNN graphs build constants from this data)
    weights.set(name, {
        data: new Float32Array(cpuData), // copy — source buffer may be recycled
        shape: shape,
    });

    totalWeightBytes += cpuData.byteLength;
    const sizeMB = (cpuData.byteLength / 1024 / 1024).toFixed(1);
    postStatus("upload",
        `${name} [${shape.join("×")}] ${sizeMB} MB ` +
        `(total: ${(totalWeightBytes / 1024 / 1024).toFixed(0)} MB)`
    );
}

function handleSetConfig(data) {
    const view = new DataView(data);
    const jsonLen = view.getUint32(1, true);
    const jsonStr = new TextDecoder().decode(new Uint8Array(data, 5, jsonLen));
    config = JSON.parse(jsonStr);
    postStatus("config",
        `Model: ${config.model_name || "unknown"}, ` +
        `arch=${config.arch}, H=${config.hidden_size}, L=${config.num_layers}`
    );

    // Reset compiled graphs for new model
    layerGraphs.length = 0;
    finalLNGraph = null;
    lmHeadGraph = null;
    destroyTensorPool();
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

// ======================== Legacy Matmul (OP_MATMUL) ========================

async function webnnMatmulLegacy(M, N, K, dataA, dataB) {
    const builder = new MLGraphBuilder(mlContext);
    const a = builder.input("a", { dataType: "float32", shape: [M, K] });
    const b = builder.constant({ dataType: "float32", shape: [N, K] }, dataB);
    const bT = builder.transpose(b, { permutation: [1, 0] });
    const c = builder.matmul(a, bT);
    const graph = await builder.build({ "output": c });

    const inT = await mlContext.createTensor({
        dataType: "float32", shape: [M, K], writable: true,
    });
    const outT = await mlContext.createTensor({
        dataType: "float32", shape: [M, N], readable: true,
    });
    mlContext.writeTensor(inT, dataA);
    mlContext.dispatch(graph, { "a": inT }, { "output": outT });
    const result = new Float32Array(await mlContext.readTensor(outT));

    inT.destroy();
    outT.destroy();
    return result;
}

// ======================== WebSocket Main Loop ========================

async function connectAndServe(wsUrl) {
    postStatus("init", "Initializing WebNN (WebNPU)...");
    await initWebNN();

    let ws = null;
    let running = true;

    function connect() {
        ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => postStatus("connected",
            `Connected to ${wsUrl} [${backendInfo}]`
        );

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
                    const dataB = new Float32Array(
                        data, headerSize + M * K * 4, N * K
                    );
                    const result = await webnnMatmulLegacy(M, N, K, dataA, dataB);
                    const respHeader = 9;
                    const respBuf = new ArrayBuffer(
                        respHeader + result.byteLength
                    );
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
                console.error("[WebNPU]", e);
            }
        };

        ws.onerror = () => postStatus("error", "WebSocket error");
        ws.onclose = () => {
            if (running) {
                postStatus("reconnecting",
                    "Connection lost, retrying in 2s..."
                );
                setTimeout(connect, 2000);
            }
        };
    }

    connect();
    return {
        stop: () => {
            running = false;
            generating = false;
            if (ws) ws.close();
        },
        getStats: () => ({
            weights: weights.size,
            totalMB: totalWeightBytes / 1024 / 1024,
            config,
            backend: backendInfo,
        }),
    };
}

if (typeof window !== "undefined") {
    window.VRAMancerInference = { connectAndServe, initWebNN };
}

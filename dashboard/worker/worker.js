/**
 * VRAMancer WebGPU Compute Worker
 *
 * Runs in the browser, connects via WebSocket to the Python backend,
 * receives weight/activation tensors, executes tiled matmul on GPU,
 * returns results.
 *
 * Protocol (binary WebSocket frames):
 *   Request:  [op:u8] [M:u32] [N:u32] [K:u32] [A:f32*M*K] [B:f32*N*K]
 *   Response: [op:u8] [M:u32] [N:u32] [C:f32*M*N]
 *   op=0x01: matmul, op=0x02: ping, op=0xFF: shutdown
 */

const OP_MATMUL = 0x01;
const OP_PING = 0x02;
const OP_SHUTDOWN = 0xFF;

let device = null;
let matmulPipeline = null;
let bindGroupLayout = null;

// --- WebGPU initialization ---

async function initWebGPU() {
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported in this browser");
    }
    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
    });
    if (!adapter) {
        throw new Error("No WebGPU adapter found");
    }

    const info = await adapter.requestAdapterInfo();
    console.log(`[VRAMancer] GPU adapter: ${info.vendor} ${info.device} (${info.description})`);

    device = await adapter.requestDevice({
        requiredLimits: {
            maxBufferSize: 256 * 1024 * 1024, // 256 MB
            maxStorageBufferBindingSize: 256 * 1024 * 1024,
        },
    });

    device.lost.then((info) => {
        console.error(`[VRAMancer] Device lost: ${info.message}`);
        postStatus("error", `Device lost: ${info.message}`);
    });

    // Load and compile shader
    const shaderResp = await fetch("matmul.wgsl");
    const shaderCode = await shaderResp.text();
    const shaderModule = device.createShaderModule({ code: shaderCode });

    // Check for compilation errors
    const compilationInfo = await shaderModule.getCompilationInfo();
    for (const msg of compilationInfo.messages) {
        if (msg.type === "error") {
            throw new Error(`Shader compile error: ${msg.message} (line ${msg.lineNum})`);
        }
    }

    bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ],
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
    });

    matmulPipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: "matmul_tiled",
        },
    });

    console.log("[VRAMancer] WebGPU matmul pipeline ready");
    return info;
}

// --- Matrix multiplication ---

async function runMatmul(M, N, K, dataA, dataB) {
    const TILE = 16;

    // Create GPU buffers
    const paramsData = new Uint32Array([M, N, K, 0]);
    const paramsBuf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuf, 0, paramsData);

    const bufA = device.createBuffer({
        size: dataA.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufA, 0, dataA);

    const bufB = device.createBuffer({
        size: dataB.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufB, 0, dataB);

    const outputSize = M * N * 4; // f32
    const bufC = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const readbackBuf = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Bind group
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: bufA } },
            { binding: 2, resource: { buffer: bufB } },
            { binding: 3, resource: { buffer: bufC } },
        ],
    });

    // Dispatch compute
    const workgroupsX = Math.ceil(N / TILE);
    const workgroupsY = Math.ceil(M / TILE);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(matmulPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
    pass.end();

    // Copy result to readback buffer
    encoder.copyBufferToBuffer(bufC, 0, readbackBuf, 0, outputSize);
    device.queue.submit([encoder.finish()]);

    // Read back
    await readbackBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readbackBuf.getMappedRange().slice(0));
    readbackBuf.unmap();

    // Cleanup
    paramsBuf.destroy();
    bufA.destroy();
    bufB.destroy();
    bufC.destroy();
    readbackBuf.destroy();

    return result;
}

// --- WebSocket protocol ---

function parseRequest(buffer) {
    const view = new DataView(buffer);
    const op = view.getUint8(0);

    if (op === OP_PING) {
        return { op };
    }
    if (op === OP_SHUTDOWN) {
        return { op };
    }
    if (op === OP_MATMUL) {
        const M = view.getUint32(1, true);
        const N = view.getUint32(5, true);
        const K = view.getUint32(9, true);

        const headerSize = 13; // 1 + 4 + 4 + 4
        const aSize = M * K * 4;
        const bSize = N * K * 4;

        if (buffer.byteLength < headerSize + aSize + bSize) {
            throw new Error(`Incomplete matmul frame: need ${headerSize + aSize + bSize}, got ${buffer.byteLength}`);
        }

        const dataA = new Float32Array(buffer, headerSize, M * K);
        const dataB = new Float32Array(buffer, headerSize + aSize, N * K);

        return { op, M, N, K, dataA, dataB };
    }

    throw new Error(`Unknown op: 0x${op.toString(16)}`);
}

function buildResponse(op, M, N, resultData) {
    const headerSize = 9; // 1 + 4 + 4
    const buf = new ArrayBuffer(headerSize + resultData.byteLength);
    const view = new DataView(buf);
    view.setUint8(0, op);
    view.setUint32(1, M, true);
    view.setUint32(5, N, true);
    new Float32Array(buf, headerSize).set(resultData);
    return buf;
}

// --- Status reporting ---

function postStatus(status, message) {
    if (typeof window !== "undefined") {
        const event = new CustomEvent("vramancer-status", {
            detail: { status, message, timestamp: Date.now() },
        });
        window.dispatchEvent(event);
    }
    console.log(`[VRAMancer] ${status}: ${message}`);
}

// --- Main connection loop ---

async function connectAndServe(wsUrl) {
    postStatus("init", "Initializing WebGPU...");
    const adapterInfo = await initWebGPU();
    postStatus("ready", `GPU: ${adapterInfo.description || adapterInfo.device}`);

    let ws = null;
    let running = true;
    let opsCompleted = 0;

    function connect() {
        ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
            postStatus("connected", `Connected to ${wsUrl}`);
        };

        ws.onmessage = async (event) => {
            try {
                const req = parseRequest(event.data);

                if (req.op === OP_PING) {
                    // Echo back a pong (same frame)
                    ws.send(event.data);
                    return;
                }

                if (req.op === OP_SHUTDOWN) {
                    postStatus("shutdown", "Server requested shutdown");
                    running = false;
                    ws.close();
                    return;
                }

                if (req.op === OP_MATMUL) {
                    const t0 = performance.now();
                    const result = await runMatmul(req.M, req.N, req.K, req.dataA, req.dataB);
                    const dt = performance.now() - t0;

                    const response = buildResponse(OP_MATMUL, req.M, req.N, result);
                    ws.send(response);

                    opsCompleted++;
                    const gflops = (2 * req.M * req.N * req.K) / (dt * 1e6);
                    postStatus("compute", `matmul ${req.M}x${req.K} * ${req.K}x${req.N} → ${dt.toFixed(1)}ms (${gflops.toFixed(1)} GFLOP/s) [#${opsCompleted}]`);
                }
            } catch (e) {
                postStatus("error", e.message);
                console.error("[VRAMancer] Request error:", e);
            }
        };

        ws.onerror = (e) => {
            postStatus("error", "WebSocket error");
            console.error("[VRAMancer] WS error:", e);
        };

        ws.onclose = () => {
            if (running) {
                postStatus("reconnecting", "Connection lost, retrying in 2s...");
                setTimeout(connect, 2000);
            }
        };
    }

    connect();

    return {
        stop: () => {
            running = false;
            if (ws) ws.close();
        },
        getStats: () => ({ opsCompleted }),
    };
}

// Auto-start if loaded as page script
if (typeof window !== "undefined") {
    window.VRAMancerWorker = { connectAndServe, initWebGPU, runMatmul };
}

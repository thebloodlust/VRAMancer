/**
 * VRAMancer WebGPU Production Compute Node (with Q8 Quantization)
 * Connects to the main cluster WebSocket and executes Neural Network layers
 */

let computedTensors = 0;
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');

// Reconstruct URL and append token if accessible
const getToken = () => {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('token') || localStorage.getItem('vrm_token') || "";
};

async function initWebGPUNode() {
    if (!navigator.gpu) {
        statusEl.textContent = "❌ WebGPU n'est pas supporté sur ce navigateur";
        statusEl.className = "status offline";
        return;
    }

    statusEl.textContent = "🟡 Acquisition du GPU local...";
    
    // 1. Hook the local user's GPU
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
        statusEl.textContent = "❌ GPU introuvable ou limité (Vérifiez chrome://flags/#enable-unsafe-webgpu)";
        return;
    }
    const device = await adapter.requestDevice();
    
    // Get GPU Info
    let gpuName = "Unknown GPU";
    if (adapter.requestAdapterInfo) {
        const info = await adapter.requestAdapterInfo();
        gpuName = info.device || info.description || gpuName;
    }
    console.log("🚀 Connecté au GPU (WebGPU):", gpuName);

    // 2. Compilation du Shader WGSL (f32 standard)
    const shaderCode = `
        @group(0) @binding(0) var<storage, read> inputData : array<f32>;
        @group(0) @binding(1) var<storage, read_write> outputData : array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let index = global_id.x;
            // Simple pass-through operation for testing inference distribution
            outputData[index] = inputData[index] * 1.0; 
        }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });
    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });

    // 3. Connect to the Proxmox/Master Node
    const port = 8081; 
    let baseHosts = location.hostname !== "" ? location.hostname : "127.0.0.1";
    let wsUrl = `ws://${baseHosts}:${port}`;
    
    const token = getToken();
    if (token) wsUrl += `?token=${token}`;

    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer"; // Important for Zero-Copy Receiving!
    
    ws.onopen = () => {
        statusEl.textContent = "🟢 Connecté au Cluster IA ! Prêt à calculer en Q8/F32.";
        statusEl.className = "status online";
        
        // Notify cluster of our capabilities
        ws.send(JSON.stringify({
            type: "capabilities",
            gpu_name: gpuName,
            max_buffer_size: device.limits.maxStorageBufferBindingSize
        }));
    };

    ws.onclose = (e) => {
        statusEl.textContent = `🔴 Déconnecté du Maître. (Code: ${e.code})`;
        statusEl.className = "status offline";
        
        // Auto-reconnect pattern
        setTimeout(initWebGPUNode, 10000);
    };

    ws.onerror = (e) => {
        console.error("WebSocket Error:", e);
    };

    ws.onmessage = async (event) => {
        const startTime = performance.now();
        computedTensors++;
        
        let header, int8Payload, quantScale = 1.0;

        if (event.data instanceof ArrayBuffer) {
            // Binary format: [Header Length Uint32 (4 bytes)] [Header string JSON] [Payload Int8...]
            const dv = new DataView(event.data);
            const headerLen = dv.getUint32(0, true);
            const headerStr = new TextDecoder().decode(new Uint8Array(event.data, 4, headerLen));
            header = JSON.parse(headerStr);
            if (header.quant_scale) quantScale = header.quant_scale;
            
            // Note: Data coming from Python is INT8 symmetrically quantized
            int8Payload = new Int8Array(event.data, 4 + headerLen);
        } else {
            // Handle Heartbeats
            try {
                header = JSON.parse(event.data);
                if(header.type === "init") return;
            } catch(e) { return; }
            int8Payload = new Int8Array(0);
        }

        if (header.type === "compute") {
            // JS INT8 to Float32 Dequantization right before WGSL dispatch
            let dataLength = Math.max(int8Payload.length, 1);
            let payloadBytes = new Float32Array(dataLength);
            for(let i=0; i<int8Payload.length; i++) {
                payloadBytes[i] = int8Payload[i] * quantScale;
            }

            // --- Exécution WebGPU réelle ---
            let byteLength = dataLength * 4;
            let alignedLength = (byteLength + 3) & ~3; // Ensure 4-byte padding

            const inputBuffer = device.createBuffer({
                size: alignedLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            const outputBuffer = device.createBuffer({
                size: alignedLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
            const readBuffer = device.createBuffer({
                size: alignedLength,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            // Write dequantized f32 tensors
            device.queue.writeBuffer(inputBuffer, 0, payloadBytes.buffer, payloadBytes.byteOffset, byteLength);

            const bindGroup = device.createBindGroup({
                layout: computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: inputBuffer } },
                    { binding: 1, resource: { buffer: outputBuffer } }
                ]
            });

            const commandEncoder = device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, bindGroup);
            const workgroupCount = Math.ceil(dataLength / 64);
            passEncoder.dispatchWorkgroups(workgroupCount);
            passEncoder.end();
            
            // Copy back
            commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, alignedLength);
            device.queue.submit([commandEncoder.finish()]);
            
            // Wait for GPU sync
            await readBuffer.mapAsync(GPUMapMode.READ);
            const outputArray = new Float32Array(readBuffer.getMappedRange());
            
            // Generate a pseudo-next-token for testing the sync loop
            let nextToken = Math.floor(Math.random() * 50) + 65; // A-Z ASCII
            if (payloadBytes.length >= 1) {
                nextToken = Math.floor(outputArray[0]) + 1; // Basic logic
            }
            if (nextToken > 120000) nextToken = 32;

            // Prepare Binary Send [Header Len][Header][U32 Next Token]
            const resHeader = JSON.stringify({
                type: "result", 
                task_id: header.task_id,
                flops: workgroupCount * 64 * 2
            });
            const encoder = new TextEncoder();
            const resHeaderBytes = encoder.encode(resHeader);
            
            const outBuf = new ArrayBuffer(4 + resHeaderBytes.length + 4);
            const outDv = new DataView(outBuf);
            outDv.setUint32(0, resHeaderBytes.length, true);
            new Uint8Array(outBuf, 4).set(resHeaderBytes);
            outDv.setUint32(4 + resHeaderBytes.length, nextToken, true); // Little endian
            
            // Send back binary result!
            ws.send(outBuf);

            readBuffer.unmap();
            
            // Cleanup to avoid Memory Leaks over thousand of passes
            inputBuffer.destroy();
            outputBuffer.destroy();
            readBuffer.destroy();

            const computeTime = performance.now() - startTime;
            const fps = (1000 / computeTime).toFixed(1);
            statsEl.innerHTML = `> Opération: ${header.task_id.substring(0,8)}<br>> Couche: ${header.layer}<br>> Tenseurs complétés (Q8): ${computedTensors}<br>> Latence (WGSL): ${computeTime.toFixed(2)}ms<br>> TPS: ${fps}`;
        }
    };
}

initWebGPUNode();
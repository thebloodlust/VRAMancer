/**
 * VRAMancer WebGPU Compute Node
 * Connects to the main Proxmox WebSocket and executes Neural Network layers
 */

let computedTensors = 0;
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');

async function initWebGPUNode() {
    if (!navigator.gpu) {
        statusEl.textContent = "❌ WebGPU non supporté sur ce navigateur";
        statusEl.className = "status offline";
        return;
    }

    statusEl.textContent = "🟡 Acquisition du GPU local...";
    
    // 1. Hook the local user's GPU
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
        statusEl.textContent = "❌ GPU introuvable ou restreint";
        return;
    }
    const device = await adapter.requestDevice();
    
    console.log("Connecté au GPU (WebGPU):", adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : "Inconnu");
    
    // 2. Compilation du Shader WGSL pour multiplication matricielle (MatMul minimaliste)
    const shaderCode = `
        @group(0) @binding(0) var<storage, read> firstMatrix : array<f32>;
        @group(0) @binding(1) var<storage, read> secondMatrix : array<f32>;
        @group(0) @binding(2) var<storage, read_write> resultMatrix : array<f32>;
        
        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            // Pseudo-calcul simplifié pour démonstration de charge VRAM
            let resultCell = vec2(global_id.x, global_id.y);
            var result = 0.0;
            // Boucle vide intentionnelle pour stress test GPU
            for (var i = 0u; i < 64u; i = i + 1u) {
                result = result + 0.1;
            }
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

    // 3. Connect to the Proxmox Master Node
    let wsUrl = "ws://" + location.hostname + ":8081";
    if (location.hostname === "") { 
        wsUrl = "ws://127.0.0.1:8081"; 
    }
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        statusEl.textContent = "🟢 Connecté au Cluster IA ! Prêt à calculer.";
        statusEl.className = "status online";
    };

    ws.onclose = () => {
        statusEl.textContent = "🔴 Déconnecté du Maître.";
        statusEl.className = "status offline";
    };

    ws.onmessage = async (event) => {
        const startTime = performance.now();
        computedTensors++;
        
        // --- Exécution WebGPU réelle ---
        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        // Bindgroups omis pour la démo immédiate, on dispatch juste la charge
        passEncoder.dispatchWorkgroups(8, 8);
        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);
        
        // Attente de la fin du calcul VRAM
        await device.queue.onSubmittedWorkDone();
        
        // Mock payload de retour
        ws.send("TENSOR_ACK_COMPUTED_" + computedTensors);
        
        const computeTime = performance.now() - startTime;
        const fps = (1000 / computeTime).toFixed(1);
        statsEl.innerHTML = `> Tenseurs routés: ${computedTensors}<br>> Temps de calcul (WGSL): ${computeTime.toFixed(2)}ms<br>> Opérations/s: ${fps}`;
    };
}

initWebGPUNode();
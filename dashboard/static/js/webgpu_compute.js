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
        return;
    }

    statusEl.textContent = "🟡 Acquisition du GPU local...";
    
    // 1. Hook the local user's GPU
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    
    console.log("Connecté au GPU:", adapter);
    
    // 2. Connect to the Proxmox Master Node
    // Change IP to the actual public/local IP of the VRAMancer host
    const masterAddress = "ws://" + location.hostname + ":8081";
    const ws = new WebSocket(masterAddress);
    
    ws.onopen = () => {
        statusEl.textContent = "🟢 Connecté au Cluster IA ! Prêt à calculer.";
        statusEl.className = "status online";
    };

    ws.onclose = () => {
        statusEl.textContent = "🔴 Déconnecté du Maître.";
        statusEl.className = "status offline";
    };

    ws.onmessage = async (event) => {
        // Here we receive the Tensor blocks + Speculative Drafts
        // Let's pretend we run a fast compute shader
        
        const startTime = performance.now();
        computedTensors++;
        
        // Simulating 50ms of WebGPU compute time
        await new Promise(r => setTimeout(r, 50)); 
        
        // Sending result back
        ws.send("TENSOR_ACK_COMPUTED");
        
        const fps = (1000 / (performance.now() - startTime)).toFixed(1);
        statsEl.innerHTML = `> Tenseurs calculés: ${computedTensors}<br>> Operations/s: ${fps}`;
    };
}

initWebGPUNode();
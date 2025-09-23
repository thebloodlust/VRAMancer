const routingLog = [
  { block: 0, device: "cuda:0", importance: "critical", size: 800 },
  { block: 1, device: "cpu", importance: "normal", size: 300 },
  { block: 2, device: "nvme", importance: "low", size: 1200 },
];

routingLog.forEach(entry => {
  const line = document.createElement("div");
  line.innerText = `Bloc ${entry.block} → ${entry.device} (${entry.importance}, ${entry.size}MB)`;
  document.body.appendChild(line);
});

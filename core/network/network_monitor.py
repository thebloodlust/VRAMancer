import psutil
import time
import threading

class NetworkMonitor:
    def __init__(self, iface=None, interval=1):
        self.iface = iface or self._get_default_iface()
        self.interval = interval
        self.stats = []
        self.running = False

    def _get_default_iface(self):
        # Prend la première interface Ethernet
        for iface in psutil.net_if_stats():
            if iface.startswith("en") or iface.startswith("eth"):
                return iface
        return list(psutil.net_if_stats().keys())[0]

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _monitor(self):
        prev = psutil.net_io_counters(pernic=True)[self.iface]
        while self.running:
            time.sleep(self.interval)
            curr = psutil.net_io_counters(pernic=True)[self.iface]
            sent = curr.bytes_sent - prev.bytes_sent
            recv = curr.bytes_recv - prev.bytes_recv
            self.stats.append({
                "sent": sent,
                "recv": recv,
                "timestamp": time.time()
            })
            print(f"[NetworkMonitor] {self.iface} | Sent: {sent/1024:.1f} KB | Recv: {recv/1024:.1f} KB")
            prev = curr

if __name__ == "__main__":
    mon = NetworkMonitor()
    mon.start()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        mon.stop()
        print("Monitoring stopped.")

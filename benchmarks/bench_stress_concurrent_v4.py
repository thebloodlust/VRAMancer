"""Stress test V4: continuous batcher behavior under concurrent load.

Usage:
  python benchmarks/bench_stress_concurrent_v4.py             # batcher OFF
  VRM_CONTINUOUS_BATCHING=1 python benchmarks/bench_stress_concurrent_v4.py  # batcher ON
"""
import os, time, threading, statistics


def run_concurrent(pipe, prompt, n_concurrent, max_tokens=100):
    results = [None] * n_concurrent
    threads = []
    barrier = threading.Barrier(n_concurrent)

    def worker(idx):
        barrier.wait()
        t0 = time.perf_counter()
        pipe.generate(prompt, max_new_tokens=max_tokens)
        results[idx] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n_concurrent):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    total_time = time.perf_counter() - t0
    throughput = (n_concurrent * max_tokens) / total_time
    return total_time, throughput, results


def main():
    batcher_on = os.environ.get('VRM_CONTINUOUS_BATCHING', '0') == '1'
    print(f'=== Batcher: {"ON" if batcher_on else "OFF"} ===')

    from core.inference_pipeline import InferencePipeline
    pipe = InferencePipeline(backend_name='huggingface', verbose=False)
    pipe.load('Qwen/Qwen2.5-7B-Instruct', num_gpus=1)

    prompt = 'Explain quantum entanglement.'
    pipe.generate(prompt, max_new_tokens=20)  # warmup

    seq_times = []
    for i in range(3):
        t0 = time.perf_counter()
        pipe.generate(prompt, max_new_tokens=100)
        seq_times.append(100 / (time.perf_counter() - t0))
    print(f'Sequential median: {statistics.median(seq_times):.2f} tok/s')

    for n in [1, 4, 8]:
        total_time, throughput, _ = run_concurrent(pipe, prompt, n, max_tokens=100)
        print(f'N={n}: total {total_time:.2f}s, total throughput {throughput:.2f} tok/s')


if __name__ == '__main__':
    main()

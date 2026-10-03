import time

import torch


def median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def measure_cpu(left, right, repeats):
    for _ in range(3):
        _ = left @ right
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = left @ right
        samples.append((time.perf_counter() - start) * 1000)
    return median(samples)


def measure_gpu(left, right, repeats):
    left_gpu = left.to("cuda")
    right_gpu = right.to("cuda")
    for _ in range(3):
        _ = left_gpu @ right_gpu
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = left_gpu @ right_gpu
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return median(samples)


def main():
    size = 2048
    repeats = 5
    left = torch.randn(size, size)
    right = torch.randn(size, size)
    cpu_ms = measure_cpu(left, right, repeats)
    print(f"CPU: {cpu_ms:.1f} ms")
    if not torch.cuda.is_available():
        print("GPU: SKIPPED")
        return
    gpu_ms = measure_gpu(left, right, repeats)
    print(f"GPU: {gpu_ms:.1f} ms")
    if gpu_ms > 0:
        print(f"Speedup: {cpu_ms / gpu_ms:.2f}x")


if __name__ == "__main__":
    main()

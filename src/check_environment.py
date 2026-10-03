import sys

import torch


def main():
    print("=== Environment ===")
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    cuda = torch.cuda.is_available()
    print("CUDA available:", cuda)

    x = torch.tensor([1.0, 2.0, 3.0])
    y = x * 2
    ok = torch.equal(y, torch.tensor([2.0, 4.0, 6.0]))
    print("Tensor test:", "OK" if ok else "FAIL")

    if not cuda:
        print("GPU count: 0")
        print("GPU test: SKIPPED")
        return

    print("GPU count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        vram_gb = props.total_memory / (1024**3)
        print(f"GPU {index}: {props.name}")
        print(f"VRAM: {vram_gb:.2f} GB")

    probe = torch.ones(4, device="cuda") * 2
    torch.cuda.synchronize()
    gpu_ok = probe.sum().item() == 8
    print("GPU test:", "OK" if gpu_ok else "FAIL")


if __name__ == "__main__":
    main()

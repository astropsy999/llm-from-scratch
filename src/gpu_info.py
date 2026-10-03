import torch


def main():
    print("PyTorch:", torch.__version__)
    cuda = torch.cuda.is_available()
    print("CUDA available:", cuda)

    if not cuda:
        print("Видеокарта NVIDIA для CUDA не видна. Счёт пойдёт на процессоре.")
        return

    print("GPU count:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        vram_gb = props.total_memory / (1024**3)
        print(f"GPU {index}: {props.name}")
        print(f"VRAM: {vram_gb:.2f} GB")


if __name__ == "__main__":
    main()

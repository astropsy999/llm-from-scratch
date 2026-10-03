import torch

x = torch.randn(2000, 2000)
print("cpu:", x.device)

if torch.cuda.is_available():
    x_gpu = x.to("cuda")
    print("gpu:", x_gpu.device)
    y_gpu = x_gpu @ x_gpu
    print("matmul:", tuple(y_gpu.shape))

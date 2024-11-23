import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 간단한 Tensor 연산
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).to(device)
    y = torch.matmul(x, x)
    print("GPU 연산 성공")
else:
    print("GPU가 활성화되지 않았습니다.")

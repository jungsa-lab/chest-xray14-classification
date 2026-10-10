"""현재 환경에서 PyTorch와 CUDA 연산을 확인한다."""
import torch
import torchvision


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(128, 128, device=device)
    y = x @ x.T
    if not torch.isfinite(y).all().item():
        raise ValueError("행렬 연산 결과에 유효하지 않은 값이 있습니다.")
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"계산 장치: {device}")
    print(f"{device.type.upper()} 행렬 연산: 정상")


if __name__ == "__main__":
    main()

"""Grad-CAM: 마지막 Convolution 블록의 특징맵과 기울기로 질환별 히트맵 생성.

    cam = GradCAM(model, target_layer(model_name, model))
    heatmaps, probs = cam(image_tensor, class_indices=[1, 2])   # (K, 224, 224), (14,)

대상 레이어
    DenseNet-121   : model.features  (denseblock4의 마지막 합성곱 → norm5 출력, 7×7×1024)
    EfficientNet-B0: model.features[-1] (마지막 1×1 Conv-BN-SiLU, 7×7×1280)

Grad-CAM 계산
    A_k = 대상 레이어 출력의 채널 k, y_c = 질환 c의 logit
    α_k = mean_{i,j} ∂y_c/∂A_k(i,j)       (채널 중요도)
    CAM = ReLU(Σ_k α_k A_k) → 입력 크기로 확대 → 0~1 정규화

주의
    - torchvision DenseNet은 features 출력에 in-place ReLU를 바로 적용한다. 원본 텐서가 변형되지 않도록
      forward hook에서 복사본을 반환하고, 원본의 기울기를 받는다.
    - TTA(좌우 반전) 이미지로 만든 CAM은 위치 해석에 쓰지 않는다 → 항상 원본 방향 이미지로 계산.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def target_layer(model_name: str, model: nn.Module) -> nn.Module:
    if model_name == "densenet121":
        return model.features
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Grad-CAM 대상 레이어를 모르는 모델: {model_name}")


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model.eval()
        self.layer = layer
        self._activation: torch.Tensor | None = None
        self._gradient: torch.Tensor | None = None
        self._handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self._activation = output
        output.register_hook(self._save_gradient)
        return output.clone()          # 뒤따르는 in-place 연산이 저장한 활성값을 바꾸지 않도록

    def _save_gradient(self, grad):
        self._gradient = grad

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, image: torch.Tensor, class_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """image (3, H, W) 또는 (1, 3, H, W) → (CAM (K, H, W) 0~1, 확률 (14,))"""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        device = next(self.model.parameters()).device
        x = image.to(device).float().requires_grad_(False)
        h, w = x.shape[-2:]

        with torch.enable_grad():
            logits = self.model(x)                      # (1, 14)
            cams = []
            for c in class_indices:
                self.model.zero_grad(set_to_none=True)
                logits[0, c].backward(retain_graph=True)
                act, grad = self._activation[0], self._gradient[0]     # (C, h, w)
                weights = grad.mean(dim=(1, 2))                       # α_k
                cam = F.relu((weights[:, None, None] * act).sum(0))
                cam = F.interpolate(cam[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
                cam = cam - cam.min()
                cam = cam / cam.max() if cam.max() > 0 else cam
                cams.append(cam.detach().cpu().numpy())
        probs = torch.sigmoid(logits[0]).detach().cpu().numpy()
        empty = np.zeros((0, h, w), dtype=np.float32)
        return (np.stack(cams).astype(np.float32) if cams else empty), probs


def overlay(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """흑백 이미지(H, W uint8) 위에 CAM 히트맵을 겹친 RGB uint8"""
    if cam.shape != gray.shape:
        cam = cv2.resize(cam, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return np.uint8((1 - alpha) * base + alpha * heat)


def peak_location(cam: np.ndarray) -> tuple[float, float]:
    """CAM 최댓값 위치 (x, y), 0~1 (이미지 왼쪽 위 = 0, 0; 영상의 왼쪽 = 환자의 오른쪽)"""
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return float(x / cam.shape[1]), float(y / cam.shape[0])

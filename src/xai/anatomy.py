"""흉부 해부학 분할 (폐, 심장, 뼈 등) — Grad-CAM이 폐 영역을 벗어났는지 판정하는 기준.

사전학습 분할 모델: torchxrayvision의 ChestX-Det PSPNet (14개 구조, 입력 512×512로 자동 확대)
    pip install torchxrayvision   (처음 실행 시 가중치 약 200MB 다운로드, 인터넷 필요)

모델이 내놓는 마스크는 근사치다. 분석 그림에 폐 윤곽선을 함께 그려 분할이 맞는지 눈으로 확인한다.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

# 분석에 쓰는 영역 묶음
REGIONS = {
    "lung": ("Left Lung", "Right Lung"),
    "heart": ("Heart",),
    "mediastinum": ("Mediastinum", "Aorta", "Weasand", "Left Hilus Pulmonis", "Right Hilus Pulmonis"),
    "diaphragm": ("Facies Diaphragmatica",),
    "bone": ("Left Clavicle", "Right Clavicle", "Left Scapula", "Right Scapula", "Spine"),
}

# 질환별 '근거가 있어야 할 영역'. 기본은 폐. 폐 밖에 소견이 있는 질환은 해당 구조를 포함한다.
EXPECTED_REGIONS = {
    "Cardiomegaly": ("lung", "heart"),
    "Hernia": ("lung", "heart", "mediastinum", "diaphragm"),     # 횡격막 탈장: 심장 뒤·횡격막 부근
    "Effusion": ("lung", "diaphragm"),                            # 늑골횡격막각
    "Pleural_Thickening": ("lung", "diaphragm"),
    "Edema": ("lung", "heart", "mediastinum"),                    # 폐문 주위 음영, 심비대 동반
}
DILATE_PX = 7      # 224 기준 약 3%: 흉막·폐 경계에 걸친 활성화를 경계 밖으로 세지 않도록


class AnatomySegmenter:
    def __init__(self, device: torch.device | str = "cpu"):
        try:
            import torchxrayvision as xrv
        except ImportError as e:
            raise SystemExit("torchxrayvision이 필요합니다: pip install torchxrayvision") from e
        self.device = torch.device(device)
        self.model = xrv.baseline_models.chestx_det.PSPNet().to(self.device).eval()
        self.targets = list(self.model.targets)

    @torch.no_grad()
    def __call__(self, gray: np.ndarray) -> dict[str, np.ndarray]:
        """흑백 uint8 (H, W) → {영역 이름: bool 마스크 (H, W)}"""
        x = torch.from_numpy(gray.astype(np.float32) / 255.0 * 2048.0 - 1024.0)[None, None].to(self.device)
        out = self.model(x)[0]                                           # (14, 512, 512) logit
        if out.min() < 0 or out.max() > 1:                               # 버전에 따라 확률을 바로 줄 수도 있음
            out = torch.sigmoid(out)
        out = out.cpu().numpy()
        h, w = gray.shape
        structures = {
            name: cv2.resize(out[i], (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
            for i, name in enumerate(self.targets)
        }
        return {region: np.any([structures[s] for s in names], axis=0) for region, names in REGIONS.items()}


def dilate(mask: np.ndarray, px: int = DILATE_PX) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def region_fractions(cam: np.ndarray, masks: dict[str, np.ndarray], disease: str) -> dict[str, float]:
    """CAM 에너지(합) 중 각 영역에 들어간 비율.

    expected : 질환별 기대 영역(팽창 포함) 안 비율 → 0.5 미만이면 '폐(기대 영역) 이탈'
    bone/heart/mediastinum/diaphragm : 기대 영역 밖에서 각 구조에 들어간 비율 (Shortcut 유형 힌트)
    outside_body : 어떤 해부 구조에도 속하지 않은 비율 (배경, 텍스트·마커, 몸 밖 장비 가능성)
    """
    total = float(cam.sum())
    if total <= 0:
        return {"expected": float("nan")}
    expected = dilate(np.any([masks[r] for r in EXPECTED_REGIONS.get(disease, ("lung",))], axis=0))
    any_structure = dilate(np.any(list(masks.values()), axis=0))
    out = {"expected": float(cam[expected].sum() / total)}
    for region in ("bone", "heart", "mediastinum", "diaphragm"):
        out[region] = float(cam[masks[region] & ~expected].sum() / total)
    out["outside_body"] = float(cam[~any_structure].sum() / total)
    return out


def dominant_outside(fractions: dict[str, float]) -> str:
    """기대 영역 밖 에너지가 가장 많이 간 곳 (자동 분류 힌트, 최종 판정은 사람이 그림을 보고)"""
    keys = ("bone", "heart", "mediastinum", "diaphragm", "outside_body")
    best = max(keys, key=lambda k: fractions.get(k, 0.0))
    return best if fractions.get(best, 0.0) > 0 else "-"

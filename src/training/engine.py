import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """모델을 한 Epoch 학습하고 평균 Loss를 반환"""
    
    model.train()   # 학습 모드
    
    total_loss = 0.0
    total_samples = 0
    
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(images)
        loss = criterion(logits, targets)
        
        if not torch.isfinite(loss).item():
            raise ValueError("유호하지 않은 학습 Loss입니다.")
        
        loss.backward() # 손실을 줄이기 위해 각 학습 매개변수를 어느 방향으로 변경해야 하는지 기울기 계산
        optimizer.step()    # 계산한 기울기를 이용해 실제 모델의 가중치를 갱신
        
        batch_size = images.size(0)
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
    if total_samples == 0:
        raise ValueError("학습 데이터가 비어 있습니다.")
    
    return total_loss / total_samples

@torch.no_grad()    # 검증 과정에서 불필요한 기울기 계산을 비활성화
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """가중치 변경 없이 검증 Loss를 계산"""
    
    model.eval()    # 평가 모드
    
    total_loss = 0.0
    total_samples = 0
    
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)
        
        logits = model(images)
        loss = criterion(logits, targets)
        
        if not torch.isfinite(loss).item():
            raise ValueError("유효하지 않은 검증 Loss입니다.")
        
        batch_size = images.size(0)
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
    if total_samples == 0:
        raise ValueError("검증 데이터가 비어 있습니다.")
    
    return total_loss / total_samples
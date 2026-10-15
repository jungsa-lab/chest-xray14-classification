import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


def create_model(pretrained=True):
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    return model  # logits; sigmoid only for metrics/inference


if __name__ == '__main__':
    model = create_model(pretrained=False).eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        print('probabilities shape:', torch.sigmoid(model(x)).shape)

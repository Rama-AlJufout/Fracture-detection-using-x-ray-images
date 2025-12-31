import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def get_model():
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)


    model.fc = nn.Linear(model.fc.in_features, 2)

    return model

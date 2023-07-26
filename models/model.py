"""
A uniform abstract interface for torchvision and Hugging Face models.
"""
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as torchvision_models
from transformers import ViTModel as HuggingFaceViTModel, ViTConfig

__all__ = ["BaselineModel", "AlexNetModel", "ResNetModel", "ViTModel"]


class BaselineModel(nn.Module):
    """
    Just returns the pixel values.
    """

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.flatten(pixel_values, start_dim=1)


class AlexNetModel(nn.Module):
    """
    A wrapper around torchvision's AlexNet.
    """

    def __init__(self, model_name: Optional[str] = "DEFAULT"):
        super(AlexNetModel, self).__init__()
        self.alexnet = torchvision_models.alexnet(weights=model_name).features

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.alexnet(pixel_values), start_dim=1)


class ResNetModel(nn.Module):
    """
    A wrapper around torchvision's ResNet-50.
    """

    def __init__(self, model_name: Optional[str] = "DEFAULT"):
        super(ResNetModel, self).__init__()
        self.resnet = torchvision_models.resnet50(weights=model_name)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = self.resnet.conv1(pixel_values)
        pixel_values = self.resnet.bn1(pixel_values)
        pixel_values = self.resnet.relu(pixel_values)
        pixel_values = self.resnet.maxpool(pixel_values)

        pixel_values = self.resnet.layer1(pixel_values)
        pixel_values = self.resnet.layer2(pixel_values)
        pixel_values = self.resnet.layer3(pixel_values)
        pixel_values = self.resnet.layer4(pixel_values)

        # pixel_values = self.resnet.avgpool(pixel_values)
        return torch.flatten(pixel_values, 1)


class ViTModel(nn.Module):
    """
    A wrapper around Hugging Face's Vision Transformer.
    """

    def __init__(self, model_name, pretrained: bool = True):
        super(ViTModel, self).__init__()
        if pretrained:
            self.vit = HuggingFaceViTModel.from_pretrained(model_name)
        else:
            config = ViTConfig.from_pretrained(model_name)
            self.vit = HuggingFaceViTModel(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=pixel_values).pooler_output

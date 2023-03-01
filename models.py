from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel as HuggingFaceViTModel


class AlexNetModel(nn.Module):
    """
    A wrapper around torchvision's AlexNet.
    """

    def __init__(self, model_name: Optional[str] = None):
        super(AlexNetModel, self).__init__()
        self.alexnet = models.alexnet(weights=model_name).features

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = len(pixel_values)
        return self.alexnet(pixel_values).view(batch_size, -1)


class ViTModel(nn.Module):
    """
    A wrapper around Hugging Face's Vision Transformer.
    """

    def __init__(self, model_name: Optional[str] = None):
        super(ViTModel, self).__init__()
        if model_name is None:
            self.vit = HuggingFaceViTModel()
        else:
            self.vit = HuggingFaceViTModel.from_pretrained(model_name)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=pixel_values).pooler_output

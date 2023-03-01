"""
A uniform abstract interface for image preprocessors.
"""
from abc import ABC, abstractmethod
from typing import List
from typing import Optional

import torch
import torchvision.models as torchvision_models
from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor

__all__ = ["Preprocessor", "BaselinePreprocessor", "AlexNetPreprocessor",
           "ResNetPreprocessor", "ViTPreprocessor"]


class Preprocessor(ABC):
    """
    Wrapper around various image processors
    """

    def __call__(self, images: List[Image.Image]) -> torch.Tensor:
        return self.process_images(images)

    @abstractmethod
    def process_images(self, images: List[Image.Image]) -> torch.Tensor:
        raise NotImplementedError("process_images needs to be implemented")


class BaselinePreprocessor(Preprocessor):
    """
    Just turns an image into pixels and does nothing else
    """

    def __init__(self):
        self.pipeline = transforms.ToTensor()

    def process_images(self, images: List[Image.Image]) -> torch.Tensor:
        return torch.cat([self.pipeline(i).unsqueeze(0) for i in images])


class AlexNetPreprocessor(Preprocessor):
    """
    Preprocessing pipeline for AlexNet
    """

    def __init__(self, model_name: Optional[str] = "DEFAULT"):
        _ = model_name
        self.pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def process_images(self, images: List[Image.Image]) -> torch.Tensor:
        return torch.cat([self.pipeline(i).unsqueeze(0) for i in images])


class ResNetPreprocessor(Preprocessor):
    """
    Preprocessing pipeline for ResNet
    """

    def __init__(self, model_name: Optional[str] = "DEFAULT"):
        if model_name in ["DEFAULT", "IMAGENET1K_V2"]:
            self.pipeline = \
                torchvision_models.ResNet50_Weights.IMAGENET1K_V2.transforms()
        elif model_name == "IMAGENET1K_V1":
            self.pipeline = \
                torchvision_models.ResNet50_Weights.IMAGENET1K_V1.transforms()
        else:
            raise ValueError("ResNet model {} not supported"
                             "".format(model_name))

    def process_images(self, images: List[Image.Image]) -> torch.Tensor:
        return torch.cat([self.pipeline(i).unsqueeze(0) for i in images])


class ViTPreprocessor(Preprocessor):
    """
    Preprocessing pipeline for ViT
    """

    def __init__(self, model_name: Optional[str] = None):
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.processor.do_rescale = False
        self.processor.do_resize = False

    def process_images(self, images: List[Image.Image]) -> torch.Tensor:
        return self.processor(images, return_tensors="pt")["pixel_values"]

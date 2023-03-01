"""
Script to extract image representations from models.
"""
import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from load_data import load_all_data
from models import *
from preprocessors import *


def get_vectors(model: nn.Module, preprocessor: Preprocessor,
                batch_size: int = 5, output_filename: Path = Path("vectors.p"),
                use_cuda: bool = True):
    """
    Extracts image representations from a model and saves them to a
    pickle file.

    :param model: The model to extract image representations from
    :param preprocessor: The preprocessor for the model
    :param batch_size: The batch size to use for the model
    :param output_filename: The file to save the representations to
    :param use_cuda: Whether or not to use GPU
    """
    data = load_all_data(preprocessor=preprocessor)

    if use_cuda:
        model.cuda()
    model.eval()

    # Extract vectors
    all_vectors = []
    all_shapes = []
    all_textures = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            all_shapes += batch["shape"]
            all_textures += batch["texture"]

            if use_cuda:
                input_ = batch["pixel_values"].cuda()
            else:
                input_ = batch["pixel_values"]

            vecs = model(input_).cpu()
            all_vectors.append(vecs)

    # Output to file
    all_vectors = torch.cat(all_vectors).detach().numpy()
    with open(output_filename, "wb") as o:
        pickle.dump({"vectors": all_vectors, "shapes": all_shapes,
                     "textures": all_textures}, o)


if __name__ == "__main__":
    with open("paths.json", "r") as f:
        output_folder = Path(json.load(f)["image_vector_output"])

    # Baseline
    get_vectors(BaselineModel(), BaselinePreprocessor(),
                output_filename=output_folder / "baseline_vectors.p")

    # AlexNet
    alexnet_preprocessor = AlexNetPreprocessor()
    get_vectors(AlexNetModel(), alexnet_preprocessor,
                output_filename=output_folder / "alexnet_vectors.p")
    get_vectors(AlexNetModel(None), alexnet_preprocessor,
                output_filename=output_folder / "alexnet_untrained_vectors.p")

    # ResNet
    resnet_preprocessor = ResNetPreprocessor()
    get_vectors(ResNetModel(), resnet_preprocessor,
                output_filename=output_folder / "resnet_vectors.p")
    get_vectors(ResNetModel(None), resnet_preprocessor,
                output_filename=output_folder / "resnet_untrained_vectors.p")

    # ViT
    vit_preprocessor = ViTPreprocessor("google/vit-base-patch32-224-in21k")
    get_vectors(ViTModel("google/vit-base-patch32-224-in21k"),
                vit_preprocessor,
                output_filename=output_folder / "vit_vectors.p")
    get_vectors(ViTModel("google/vit-base-patch32-224-in21k",
                         pretrained=False),
                vit_preprocessor,
                output_filename=output_folder / "vit_untrained_vectors.p")

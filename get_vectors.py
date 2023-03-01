import pickle

import torch
import torch.nn as nn
from tqdm import tqdm

from load_data import load_all_data
from models import ViTModel, AlexNetModel
from preprocessors import Preprocessor, ViTPreprocessor, AlexNetPreprocessor


def get_vectors(model: nn.Module, preprocessor: Preprocessor,
                batch_size: int = 5, output_filename: str = "vectors.p",
                use_cuda: bool = True):
    """
    Extract representations

    :param model:
    :param preprocessor:
    :param batch_size:
    :param use_cuda:
    :return:
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


use_vit = True

if __name__ == "__main__":
    if use_vit:
        get_vectors(ViTModel("google/vit-base-patch16-224"),
                    ViTPreprocessor("google/vit-base-patch16-224"))
    else:
        #  model = AlexNetModel("AlexNet_Weights.IMAGENET1K_V1")
        get_vectors(AlexNetModel("AlexNet_Weights.DEFAULT"),
                    AlexNetPreprocessor())

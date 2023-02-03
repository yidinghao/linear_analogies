import pickle
from typing import Any, Dict

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import ViTModel, ViTFeatureExtractor

from load_data import load_all_data


def transform(batch: Dict[str, Any], tokenizer: ViTFeatureExtractor):
    inputs = tokenizer(batch["image"], return_tensors="pt")
    inputs["shape"] = batch["shape"]
    inputs["texture"] = batch["texture"]
    return inputs


def get_vectors(model: nn.Module, tokenizer: ViTFeatureExtractor,
                batch_size: int = 5, output_filename: str = "vectors.p",
                use_cuda: bool = True):
    """
    Extract representations

    :param model:
    :param tokenizer:
    :param batch_size:
    :param use_cuda:
    :return:
    """
    data = load_all_data().with_transform(lambda b: transform(b, tokenizer))
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

            vecs = model(input_).pooler_output
            del input_
            vecs = vecs.cpu()
            all_vectors.append(vecs)

    # Output to file
    all_vectors = torch.cat(all_vectors).detach().numpy()
    with open(output_filename, "wb") as o:
        pickle.dump({"vectors": all_vectors, "shapes": all_shapes,
                     "textures": all_textures}, o)


if __name__ == "__main__":
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224")
    get_vectors(vit, feature_extractor)

import csv
import os
from pathlib import Path

from datasets import Dataset, load_dataset, concatenate_datasets

_all_data = None


def _add_metadata_file(folder: Path):
    """
    Adds a metadata.csv file to a folder that lists the shape and
    texture features of each image in the folder.

    :param folder: The folder to add the metadata.csv file to
    """
    with open(folder / "metadata.csv", "w") as o:
        writer = csv.writer(o)
        writer.writerow(["file_name", "shape", "texture"])

        for filename in os.listdir(folder):
            if not filename.endswith(".png"):
                continue
            writer.writerow([filename] + filename[:-4].split("-"))


def _load_folder(folder: str, root: Path = Path("data/")) -> Dataset:
    """
    Loads the images in a folder.

    :param folder: The name of the folder (not the full path)
    :param root: The directory the folder is contained in

    :return: The loaded data
    """
    folder = root / folder

    # Add metadata file if it doesn't exist
    if not os.path.isfile(folder / "metadata.csv"):
        _add_metadata_file(folder)

    return load_dataset("imagefolder", data_dir=folder)["train"]


def load_all_data(root: Path = Path("data/")) -> Dataset:
    """
    Loads all images from the Gheiros dataset.

    :param root: The root directory of the Gheiros dataset
    :return: The loaded data
    """
    global _all_data
    if _all_data is None:
        datasets = [_load_folder(f, root=root)
                    for f in os.listdir(root) if f.isalnum()]
        _all_data = concatenate_datasets(datasets)
    return _all_data


def clear_cache():
    del _all_data

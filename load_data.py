import csv
import os
from pathlib import Path
from typing import Dict

from datasets import Dataset, load_dataset, concatenate_datasets


def _add_metadata_file(folder: Path):
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


def load_all_data(root: Path = Path("data/")) -> Dict[str, Dataset]:
    return concatenate_datasets([_load_folder(f, root=root)
                                 for f in os.listdir(root) if f.isalnum()])

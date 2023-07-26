import csv
import itertools
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np

from shapes.shape import Shape


class ShapeGenerator(ABC):
    """
    A class that generates shapes.
    """

    @abstractmethod
    def generate(self) -> Iterator[Shape]:
        raise NotImplementedError

    def create_dataset(self, folder: Union[Path, str]):
        """
        Creates a dataset of images.

        :param folder: The folder where the dataset should be saved
        """
        folder = Path(folder)
        os.mkdir(folder)
        with open(folder / "metadata.csv", "w") as metadata_file:
            metadata = csv.writer(metadata_file)
            metadata.writerow(["file_name", "shape", "color", "x", "y",
                               "radius", "rotation", "texture"])

            for shape in self.generate():
                fn = shape.generate_filename()
                shape.save(folder / fn)
                metadata.writerow([fn] + list(shape))


class GridSearch(ShapeGenerator):
    """
    Generates all possible shapes in a grid search of feature values.
    """

    def __init__(self, num_x_steps: Optional[int] = 5,
                 num_y_steps: Optional[int] = 5,
                 num_radius_steps: Optional[int] = 10,
                 min_radius_stepsize: int = 5,
                 rotation_step: Optional[int] = 10,
                 no_texture: bool = True):
        """
        Specifies the parameters of the grid search.

        :param num_x_steps: The number of x values to include
        :param num_y_steps: The number of y values to include
        :param num_radius_steps: The number of radius values to include
        :param min_radius_stepsize: The smallest possible interval
            between two radius values
        :param rotation_step: The interval between different rotation
            values, measured in degrees
        :param no_texture: If True, then shapes will not have a texture,
            and texture will not be included in the grid search
        """
        self.num_x_steps = num_x_steps
        self.num_y_steps = num_y_steps
        self.num_radius_steps = num_radius_steps
        self.min_radius_stepsize = min_radius_stepsize
        self.rotation_step = rotation_step
        self.no_texture = no_texture

    def generate(self) -> Iterator[Shape]:
        """
        Generates all possible images.
        """
        textures = [None] if self.no_texture else Shape.textures
        xs = np.linspace(Shape.min_radius, Shape.max_x, self.num_x_steps)
        ys = np.linspace(Shape.min_radius, Shape.max_y, self.num_y_steps)
        grid = itertools.product(Shape.shapes, Shape.colors, textures, xs, ys)
        for s, c, t, x, y in grid:
            x, y = round(x), round(y)
            thetas = np.arange(0, Shape.max_rotation[s], self.rotation_step)

            max_r = Shape.max_radius(x, y)
            max_r_steps = math.ceil(1 + (max_r - Shape.min_radius) /
                                    self.min_radius_stepsize)
            num_r_steps = min(max_r_steps, self.num_radius_steps)
            radii = np.linspace(Shape.min_radius, max_r, num_r_steps)

            for r, theta in itertools.product(radii, thetas):
                r, theta = round(r), round(float(theta), 2)
                theta = int(theta) if theta.is_integer() else theta
                yield Shape(s, c, x, y, r, theta, t)


class RandomShapes(ShapeGenerator):
    def __init__(self, n: int, **shape_kwargs):
        self.n = n
        self.shape_kwargs = shape_kwargs

    def generate(self) -> Iterator[Shape]:
        for _ in range(self.n):
            yield Shape.generate(**self.shape_kwargs)


if __name__ == "__main__":
    # GridSearch().create_dataset("../datasets/shapes/gridsearch_no_texture")
    RandomShapes(1000).create_dataset("../datasets/shapes/random_no_texture")

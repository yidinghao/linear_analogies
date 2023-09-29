import csv
import itertools
import os
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Union

import numpy as np

from shapes.shape import Shape

DatasetRow = namedtuple("DatasetRow", "shape, filename, columns",
                        defaults=[tuple()])


class ShapeGenerator(ABC):
    """
    A class that generates shapes.
    """

    def __init__(self, *column_names: str):
        self.column_names = ("file_name",) + column_names + Shape._fields

    @abstractmethod
    def generate(self) -> Iterator[DatasetRow]:
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
            metadata.writerow(self.column_names)
            for shape, fn, cols in self.generate():
                shape.save(folder / fn)
                metadata.writerow((fn,) + cols + tuple(shape))


class RandomShapes(ShapeGenerator):
    def __init__(self, n: int, **shape_kwargs):
        super(RandomShapes, self).__init__()
        self.n = n
        self.shape_kwargs = shape_kwargs

    def generate(self) -> Iterator[Tuple[Shape, str, ...]]:
        for _ in range(self.n):
            shape = Shape.generate(**self.shape_kwargs)
            yield DatasetRow(shape, shape.generate_filename())


class GridSearch(ShapeGenerator):
    """
    Generates all possible shapes in a grid search of feature values.
    """

    def __init__(self, num_x_steps: Optional[int] = 5,
                 num_y_steps: Optional[int] = 5,
                 radius_step: int = 5,
                 rotation_step: Optional[int] = 10,
                 no_texture: bool = True):
        """
        Specifies the parameters of the grid search.

        :param num_x_steps: The number of x values to include
        :param num_y_steps: The number of y values to include
        :param radius_step: The interval between two radius values
        :param rotation_step: The interval between different rotation
            values, measured in degrees
        :param no_texture: If True, then shapes will not have a texture,
            and texture will not be included in the grid search
        """
        super(GridSearch, self).__init__()
        self.num_x_steps = num_x_steps
        self.num_y_steps = num_y_steps
        self.radius_step = radius_step
        self.rotation_step = rotation_step
        self.no_texture = no_texture

    def generate(self) -> Iterator[Shape]:
        # Set up grid for features that do not depend on other features
        textures = [None] if self.no_texture else Shape.textures
        xs = np.linspace(Shape.min_radius, Shape.max_x, self.num_x_steps)
        ys = np.linspace(Shape.min_radius, Shape.max_y, self.num_y_steps)
        grid = itertools.product(Shape.shapes, Shape.colors, textures, xs, ys)

        for s, c, t, x, y in grid:
            x, y = round(x), round(y)

            # Radius and rotation depend on shape and position
            max_r = Shape.max_radius(x, y)
            radii = np.arange(Shape.min_radius, max_r, self.radius_step)
            thetas = np.arange(0, Shape.max_rotation[s], self.rotation_step)

            for r, theta in itertools.product(radii, thetas):
                r, theta = round(r), round(float(theta), 2)
                theta = int(theta) if theta.is_integer() else theta
                shape = Shape(s, c, x, y, r, theta, t)
                yield DatasetRow(shape, shape.generate_filename())


class AnalogyTest(ShapeGenerator):
    """
        An analogy test is a set of analogies defined by an _arrow_ and a
        _functor_. The arrow and the functor are two shape transformations
        such that every analogy (a, b, c, d) satisfies:
            - b = functor(a)
            - c = arrow(b)
            - d = arrow(a)
        """

    def __init__(self, arrow_features: Union[str, Iterable[str]],
                 functor_features: Union[str, Iterable[str]],
                 no_texture: bool = True, **a_properties):
        """
        Both the arrow and the functor will be functions that perturb
        one or more features of its input.

        :param no_texture: If True, texture will not be used
        :param a_properties: Any properties that a should have, if it is
            generated randomly
        """
        super(AnalogyTest, self).__init__("Analogy ID", "Item")

        if isinstance(arrow_features, str):
            self.arrow_features = [arrow_features]
        else:
            self.arrow_features = list(set(arrow_features))

        if isinstance(functor_features, str):
            self.functor_features = [functor_features]
        else:
            self.functor_features = list(set(functor_features))

        self.no_texture = no_texture
        self.a_properties = a_properties

        self._n = 1000

    def generate_analogy(self) -> Tuple[Shape, ...]:
        """
        Generates a single analogy. The functor and the arrow are
        sampled at generation time.
        """
        a = Shape.generate(no_texture=self.no_texture, **self.a_properties)
        b = a.perturb(self.functor_features)
        c = b.perturb(self.arrow_features)
        d = a._replace(**{k: getattr(c, k) for k in self.arrow_features})
        return a, b, c, d

    def generate(self) -> Iterator[DatasetRow]:
        for i in range(self._n):
            for s, j in zip(self.generate_analogy(), ("a", "b", "c", "d")):
                yield DatasetRow(s, f"{i}{j}.png", (i, j))

    def create_dataset(self, folder: Union[Path, str], n: int = 1000):
        self._n = n
        super(AnalogyTest, self).create_dataset(folder)


if __name__ == "__main__":
    GridSearch().create_dataset("../datasets/grid_search/no_texture")
    # RandomShapes(1000).create_dataset("../datasets/shapes/random_no_texture")

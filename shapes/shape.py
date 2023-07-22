"""
Code for generating pictures of shapes on a gray background.
"""
import math
import random
from collections import namedtuple
from pathlib import Path
from typing import KeysView, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageOps, ImageStat, ImageEnhance

Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]


class Textures:
    """
    A container for textures
    """

    _brightness = 512 / 3 - 1
    textures_dir = Path("shapes/textures")

    def __init__(self, *texture_names: str):
        self._textures = {tn: None for tn in texture_names}

    def __contains__(self, item: str):
        return item in self._textures

    def __iter__(self):
        return self._textures.__iter__()

    def __len__(self):
        return len(self._textures)

    def __getitem__(self, item: str):
        if self._textures[item] is None:
            self._textures[item] = Textures._load_texture(item)
        return self._textures[item]

    @classmethod
    def _load_texture(cls, texture_name: str) -> Image:
        """ Loads a texture from a jpeg file """
        fn = cls.textures_dir / "{}.jpeg".format(texture_name)
        img = ImageOps.grayscale(Image.open(fn).resize((224, 224)))
        factor = cls._brightness / ImageStat.Stat(img).mean[0]
        return ImageEnhance.Brightness(img).enhance(factor).convert("RGB")

    def keys(self) -> KeysView:
        return self._textures.keys()


class Shape(NamedTuple):
    """
    A shape inside a (225, 225) square canvas
    """
    shape: str
    color: Color
    center_x: int
    center_y: int
    radius: int
    rotation: int
    texture: Optional[str] = None

    """ Shape properties """

    shapes = ("circle", "square", "triangle")
    colors = {"red": (225, 0, 0), "green": (0, 225, 0), "blue": (0, 0, 225)}
    max_x = 225
    max_y = 225
    min_radius = 20
    textures = Textures("blotchy", "knitted", "lacelike", "marbled", "porous")

    """ Generate a shape """

    @classmethod
    def _generate_shape(cls, exclude: Optional[str] = None):
        if exclude is None:
            return random.choice(cls.shapes)
        return random.choice([s for s in cls.shapes if s != exclude])

    @classmethod
    def _generate_color(cls, exclude: Optional[Color] = None) -> Color:
        color_names = [cn for cn, c in cls.colors.items() if c != exclude]
        return cls.colors[random.choice(color_names)]

    @classmethod
    def _generate_x(cls, radius: Optional[int] = None) -> int:
        if radius is None:
            radius = cls.min_radius
        return random.randint(radius, cls.max_x - radius)

    @classmethod
    def _generate_y(cls, radius: Optional[int] = None) -> int:
        if radius is None:
            radius = cls.min_radius
        return random.randint(radius, cls.max_y - radius)

    @classmethod
    def _generate_radius(cls, max_radius: Optional[int] = None) -> int:
        if max_radius is None:
            max_radius = int(min(cls.max_x, cls.max_y) / 2)
        return random.randint(cls.min_radius, max_radius)

    @staticmethod
    def _generate_rotation():
        return random.randint(0, 359)

    @classmethod
    def _generate_texture(cls, exclude: Optional[str] = None):
        return random.choice([k for k in cls.textures if k != exclude])

    @classmethod
    def generate(cls, shape: Optional[str] = None,
                 color: Optional[Union[Color, str]] = None,
                 center_x: Optional[int] = None,
                 center_y: Optional[int] = None,
                 radius: Optional[int] = None,
                 rotation: Optional[int] = None,
                 texture: Optional[str] = None,
                 no_texture: bool = True) -> "Shape":
        """
        Generate a random shape with desired properties.
        """
        if shape is None:
            shape = cls._generate_shape()

        if color is None:
            color = cls._generate_color()
        elif isinstance(color, str):
            color = colors[color]

        if center_x is None:
            center_x = cls._generate_x()

        if center_y is None:
            center_y = cls._generate_y()

        if radius is None:
            max_radius = min(center_x, center_y, cls.max_x - center_x,
                             cls.max_y - center_y)
            radius = cls._generate_radius(max_radius=max_radius)

        if rotation is None:
            rotation = cls._generate_rotation()

        if not no_texture and texture is None:
            texture = cls._generate_texture()

        return cls(shape, color, center_x, center_y, radius, rotation,
                   texture=texture)

    def perturb(self, features: List[str]) -> "Shape":
        kwargs = {}
        if "shape" in features:
            kwargs["shape"] = self._generate_shape(exclude=self.shape)
        if "color" in features:
            kwargs["color"] = self._generate_color(exclude=self.color)
        if "center_x" in features:
            center_x = self._generate_x(radius=self.radius)
            kwargs["center_x"] = center_x
        else:
            center_x = self.center_x
        if "center_y" in features:
            center_y = self._generate_y(radius=self.radius)
            kwargs["center_y"] = self._generate_y(radius=self.radius)
        else:
            center_y = self.center_y
        if "radius" in features:
            max_radius = min(center_x, center_y, Shape.max_x - center_x,
                             Shape.max_y - center_y)
            kwargs["radius"] = self._generate_radius(max_radius)
        if "rotation" in features:
            kwargs["rotation"] = self._generate_rotation()
        if "texture" in features:
            kwargs["texture"] = self._generate_texture(exclude=self.texture)

        return self._replace(**kwargs)

    """ Draw the shape """

    def _draw_circle(self, drawer: ImageDraw):
        drawer.ellipse((self.center_x - self.radius,
                        self.center_y - self.radius,
                        self.center_x + self.radius,
                        self.center_y + self.radius), fill=self.color)

    def _draw_polygon(self, drawer: ImageDraw, n: int):
        drawer.regular_polygon((self.center_x, self.center_y, self.radius),
                               n, rotation=self.rotation, fill=self.color)

    def _draw_shape(self, drawer: ImageDraw):
        if self.shape == "circle":
            self._draw_circle(drawer)
        elif self.shape == "square":
            self._draw_polygon(drawer, 4)
        elif self.shape == "triangle":
            self._draw_polygon(drawer, 3)
        else:
            raise NotImplementedError("Drawing shape {} is currently not "
                                      "supported".format(self.shape))

    def draw(self, bg_color: Color = (128, 128, 128)) -> Image:
        """
        Draws this shape.

        :param bg_color: The background color
        :return: The image with the shape in it
        """
        # Draw the basic shape
        shape_image = Image.new("RGB", (224, 224), bg_color)
        self._draw_shape(ImageDraw.Draw(shape_image))

        if self.texture is None:
            return shape_image

        # Draw texture
        mask = Image.new("RGBA", (224, 224), (0, 0, 0, 255))
        self._replace(color=(225, 225, 225, 86))._draw_shape(
            ImageDraw.Draw(mask))

        return Image.composite(shape_image, Shape.textures[self.texture],
                               mask).convert("RGB")


""" Calculate the overlap between two shapes """

# pixels: Overlap in pixels
# percentage: Overlap in percentage of combined area
# percentage1: Overlap in percentage of img1's area
# percentage2: Overlap in percentage of img2's area
Overlap = namedtuple("Overlap", "pixels percentage percentage1 percentage2")


def _get_area(img: Image) -> int:
    return np.asarray(img, dtype="int8").sum()


def calculate_overlap(s1: Shape, s2: Shape) -> Overlap:
    """
    Calculates how much overlap there is between two shapes.

    :param s1: A shape
    :param s2: Another shape image
    :return: The amount of overlap between s1 and s2
    """
    # First check for zero overlap
    d = math.dist((s1.center_x, s1.center_y), (s2.center_x, s2.center_y))
    if d > s1.radius + s2.radius + .01:
        return Overlap(0, 0., 0., 0.)

    # Now draw the two pictures and calculate the overlap
    white = (255, 255, 255)
    black = (0, 0, 0)
    img1 = s1._replace(color=white).draw(bg_color=black).convert("1")
    img2 = s2._replace(color=white).draw(bg_color=black).convert("1")

    area1 = _get_area(img1)
    area2 = _get_area(img2)
    and_ = _get_area(ImageChops.logical_and(img1, img2))
    or_ = _get_area(ImageChops.logical_or(img1, img2))

    return Overlap(and_, and_ / or_, and_ / area1, and_ / area2)

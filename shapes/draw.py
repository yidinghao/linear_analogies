"""
Code for generating pictures of shapes on a gray background.
"""
import math
import random
from collections import namedtuple
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageOps, ImageStat, ImageEnhance

""" Color and Shape Definitions """

colors = {"red": (225, 0, 0),
          "green": (0, 225, 0),
          "blue": (0, 0, 225),
          "cyan": (0, 225, 225),
          "magenta": (225, 0, 225),
          "yellow": (225, 225, 0),
          "gray": (128, 128, 128),
          "black": (0, 0, 0),
          "white": (225, 225, 225)}

shapes = ["circle", "square", "triangle"]

""" Texture Definitions """

texture_names = ["blotchy", "knitted", "lacelike", "marbled", "porous"]
textures_dir = Path("textures")
textures = {}


def _load_texture(texture_name: str, brightness: float = 512 / 3 - 1) -> Image:
    """ Loads textures from files """
    global textures_dir
    texture_img = Image.open(textures_dir / "{}.jpeg".format(texture_name))
    texture_img = ImageOps.grayscale(texture_img.resize((224, 224)))
    factor = brightness / ImageStat.Stat(texture_img).mean[0]
    return ImageEnhance.Brightness(texture_img).enhance(factor).convert("RGB")


def get_texture(texture_name: str) -> Image:
    """ Retrieves a texture based on its name. """
    global textures
    if texture_name not in textures:
        textures[texture_name] = _load_texture(texture_name)
    return textures[texture_name]


""" Code for Drawing Shapes """
1
Shape = namedtuple("Shape", "shape color center_x center_y radius rotation")


def generate_shape(shape_name: str, color: Tuple[int, ...]) -> Shape:
    """
    Generates a shape with a given color and optional texture.
    Number of unique blue circles: 1,089,836
    Number of unique blue squares/triangles: 392,340,960
    Total number of unique shapes: 2,357,315,268

    :param shape_name: One of the options in the shapes list
    :param color: An RGB or RGBA tuple
    :return: A randomly generated shape
    """
    global colors
    if isinstance(color, str):
        color = colors[color]

    x = random.randint(20, 205)
    y = random.randint(20, 205)
    max_radius = min(x, y, 225 - x, 225 - y)
    radius = random.randint(20, max_radius)
    rotation = random.randint(0, 359)

    return Shape(shape_name, color, x, y, radius, rotation)


def _draw_shape(drawer: ImageDraw, shape: Shape):
    if shape.shape == "circle":
        drawer.ellipse((shape.center_x - shape.radius,
                        shape.center_y - shape.radius,
                        shape.center_x + shape.radius,
                        shape.center_y + shape.radius), fill=shape.color)
    elif shape.shape == "square":
        drawer.regular_polygon((shape.center_x, shape.center_y, shape.radius),
                               4, rotation=shape.rotation, fill=shape.color)
    elif shape.shape == "triangle":
        drawer.regular_polygon((shape.center_x, shape.center_y, shape.radius),
                               3, rotation=shape.rotation, fill=shape.color)


def draw_shape(shape: Shape, texture_name: Optional[str] = None,
               bg_color: Tuple[int, ...] = colors["gray"]) -> Image:
    """
    Draws a colored shape with a solid background and an optional
    texture.

    :param shape: The shape to be drawn
    :param texture_name: An optional texture to apply to the shape
    :param bg_color: The background color
    :return: The image with the shape in it
    """
    global colors

    # Draw the basic shape
    shape_image = Image.new("RGB", (224, 224), bg_color)
    _draw_shape(ImageDraw.Draw(shape_image), shape)

    if texture_name is None:
        return shape_image

    # Draw texture
    mask = Image.new("RGBA", (224, 224), (0, 0, 0, 255))
    _draw_shape(ImageDraw.Draw(mask),
                shape._replace(color=(225, 225, 225, 86)))

    return Image.composite(shape_image, get_texture(texture_name),
                           mask).convert("RGB")


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
    img1 = draw_shape(s1._replace(color=(255, 255, 255)),
                      bg_color=(0, 0, 0)).convert("1")
    img2 = draw_shape(s2._replace(color=(255, 255, 255)),
                      bg_color=(0, 0, 0)).convert("1")

    area1 = _get_area(img1)
    area2 = _get_area(img2)
    intersection = _get_area(ImageChops.logical_and(img1, img2))
    union = _get_area(ImageChops.logical_or(img1, img2))

    return Overlap(intersection, intersection / union, intersection / area1,
                   intersection / area2)


if __name__ == "__main__":
    shape = Shape("circle", colors["green"], 112, 112, 50, 0)
    for t in texture_names:
        draw_shape(shape, texture_name=t).show()

"""
Code for generating pictures of shapes on a gray background.
"""
import random
from collections import namedtuple
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw

""" Color and Shape Definitions """

colors = {"red": (225, 0, 0),
          "green": (0, 225, 0),
          "blue": (0, 0, 225),
          "gray": (128, 128, 128)}

shapes = ["circle", "square", "triangle"]

""" Texture Definitions """

texture_names = ["blotchy", "knitted", "lacelike", "marbled", "porous"]
textures_dir = Path("data/textures/")
textures = {}


def _load_texture(texture_name: str) -> Image:
    """ Loads textures from files """
    global textures_dir
    texture_image = Image.open(textures_dir / "{}.jpeg".format(texture_name))
    return texture_image.resize((224, 224)).convert("L").convert("RGBA")


def get_texture(texture_name: str) -> Image:
    """ Retrieves a texture based on its name. """
    global textures
    if texture_name not in textures:
        textures[texture_name] = _load_texture(texture_name)
    return textures[texture_name]


""" Code for Drawing Shapes """

Shape = namedtuple("Shape", "shape color center_x center_y radius rotation")


def generate_shape(shape_name: str, color: Tuple[int, ...]) -> Shape:
    """
    Generates a shape with a given color and optional texture.

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


def draw_shape(shape: Shape, texture_name: Optional[str] = None) -> Image:
    """
    Draws a colored shape with a gray background and an optional
    texture.

    :param shape: The shape to be drawn
    :param texture_name: An optional texture to apply to the shape
    :return: The image with the shape in it
    """
    global colors

    # Draw the basic shape
    shape_image = Image.new("RGB", (224, 224), colors["gray"])
    _draw_shape(ImageDraw.Draw(shape_image), shape)

    if texture_name is None:
        return shape_image

    # Draw texture
    mask = Image.new("RGBA", (224, 224), (0, 0, 0))
    _draw_shape(ImageDraw.Draw(mask),
                shape._replace(color=(256, 256, 256, 50)))

    return Image.composite(shape_image, get_texture(texture_name),
                           mask).convert("RGB")

"""
Code for generating pictures of shapes on a gray background.
"""
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw

""" Color Definitions """

colors = {"red": (225, 0, 0),
          "green": (0, 225, 0),
          "blue": (0, 0, 225),
          "gray": (128, 128, 128)}

""" Code for Loading Textures """

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


def _draw_shape(drawer: ImageDraw, shape: str, color: Tuple[int, ...],
                center_x: int, center_y: int, radius: int, rotation: int,
                texture_name: Optional[str] = None):
    """ Helper function for drawing a shape """
    if shape == "circle":
        drawer.ellipse((center_x - radius, center_y - radius, center_x +
                        radius, center_y + radius), fill=color)
    elif shape == "square":
        drawer.regular_polygon((center_x, center_y, radius), 4,
                               rotation=rotation, fill=color)
    elif shape == "triangle":
        drawer.regular_polygon((center_x, center_y, radius), 3,
                               rotation=rotation, fill=color)


def draw_shape(shape: str, color: Tuple[int, ...], center_x: int,
               center_y: int, radius: int, rotation: int,
               texture_name: Optional[str] = None) -> Image:
    """
    Draws a colored shape with a gray background and an optional
    texture.

    :param shape:
    :param color:
    :param center_x:
    :param center_y:
    :param radius:
    :param rotation:
    :param texture_name:
    :return:
    """
    global colors

    # Draw the basic shape
    shape_image = Image.new("RGB", (224, 224), colors["gray"])
    _draw_shape(ImageDraw.Draw(shape_image), shape, color, center_x, center_y,
                radius, rotation)

    if texture_name is None:
        return shape_image

    # Draw texture
    mask = Image.new("RGBA", (224, 224), (0, 0, 0))
    _draw_shape(ImageDraw.Draw(mask), shape, (256, 256, 256, 50), center_x,
                center_y, radius, rotation)

    return Image.composite(shape_image, get_texture(texture_name),
                           mask).convert("RGB")

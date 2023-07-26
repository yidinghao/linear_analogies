"""
Code for generating pictures of shapes on a gray background.
"""
import random
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageOps, ImageStat, ImageEnhance

canvas_size = (225, 225)


class Color:
    """
    Data structure for colors
    """

    names = {"red": (255, 0, 0),
             "green": (0, 255, 0),
             "blue": (0, 0, 255),
             "black": (0, 0, 0),
             "white": (255, 255, 255),
             "gray": (127, 127, 127)}

    def __init__(self, *args, alpha: Optional[int] = None):
        """
        Creates a color.

        :param args: The color's name (str) or RGB value (int). For RGB,
            put the 3 numbers as separate arguments. For the color name,
            use the keys of Color.names
        :param alpha: The color's alpha value
        """
        self.alpha = alpha
        self.name = None

        # If the user entered the color name...
        if len(args) == 1 and args[0] in self.names:
            self.rgb = self.names[args[0]]
            self.name = args[0]
        # If the user entered the RGB...
        elif all(isinstance(a, int) for a in args) and len(args) == 3:
            self.rgb = tuple(args)
        # If the user entered an RGBA...
        elif all(isinstance(a, int) for a in args) and len(args) == 4:
            self.rgb = tuple(args[:3])
            if alpha is None:
                self.alpha = args[3]
        else:
            raise ValueError("{} is not a valid color".format(",".join(args)))

        # Figure out the name of the color
        if self.name is None:
            for k, v in self.names.items():
                if v == self.rgb:
                    self.name = k
                    break
        if self.name is None:
            self.name = "%02x%02x%02x" % self.rgb

    def __call__(self) -> Tuple[int, ...]:
        return self.rgb if self.alpha is None else self.rgb + (self.alpha,)

    def __repr__(self):
        if self.alpha is None:
            return "Color({})".format(self.name)
        return "Color({}, alpha={})".format(*self.name, self.alpha)

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.rgb + (self.alpha,))

    def __eq__(self, other: "Color") -> bool:
        return self.rgb == other.rgb and self.alpha == other.alpha


class Texture:
    """
    Data structure for textures
    """
    _brightness = 512. / 3. - 1.
    _cache = dict()
    dir = Path("shapes/textures")

    def __init__(self, texture_name: str):
        self.name = texture_name

    def __repr__(self):
        return "Texture({})".format(self.name)

    def image(self):
        if self.name in self._cache:
            return self._cache[self.name]
        else:
            fn = self.dir / "{}.jpeg".format(self.name)
            img = ImageOps.grayscale(Image.open(fn).resize((224, 224)))
            factor = self._brightness / ImageStat.Stat(img).mean[0]
            img = ImageEnhance.Brightness(img).enhance(factor).convert("RGB")
            self._cache[self.name] = img
            return img


class Shape(NamedTuple):
    """
    A shape inside a (225, 225) square canvas
    """
    shape: str
    color: Color
    x: int
    y: int
    radius: int
    rotation: int
    texture: Optional[Texture] = None

    """ Shape properties """

    shapes = ("circle", "square", "triangle")
    colors = (Color("red"), Color("green"), Color("blue"))
    min_radius = 20
    max_x = canvas_size[0] - min_radius - 1
    max_y = canvas_size[1] - min_radius - 1

    max_rotation = {"circle": 1, "square": 90, "triangle": 120}
    textures = (Texture("blotchy"), Texture("knitted"), Texture("lacelike"),
                Texture("marbled"), Texture("porous"))

    @classmethod
    def max_radius(cls, x: int, y: int) -> int:
        return min(x, y, canvas_size[0] - x - 1, canvas_size[1] - y - 1)

    """ Generate a shape """

    @staticmethod
    def _sample(options, exclude=None):
        if exclude is None:
            return random.choice(options)
        return random.choice([i for i in options if i != exclude])

    @classmethod
    def _generate_xy(cls, max_xy: int, radius: Optional[int] = None) -> int:
        if radius is None:
            radius = cls.min_radius
        return random.randint(radius, max_xy - radius)

    @classmethod
    def _generate_radius(cls, max_radius: Optional[int] = None) -> int:
        if max_radius is None:
            max_radius = int(min(cls.max_x, cls.max_y) / 2)
        return random.randint(cls.min_radius, max_radius)

    @classmethod
    def _generate_rotation(cls, shape: str):
        return random.randint(0, cls.max_rotation[shape])

    @classmethod
    def generate(cls, shape: Optional[str] = None,
                 color: Optional[Color] = None,
                 x: Optional[int] = None,
                 y: Optional[int] = None,
                 radius: Optional[int] = None,
                 rotation: Optional[int] = None,
                 texture: Optional[str] = None,
                 no_texture: bool = True) -> "Shape":
        """
        Generate a random shape with desired properties.
        """
        shape = cls._sample(cls.shapes) if shape is None else shape
        color = cls._sample(cls.colors) if color is None else color
        x = cls._generate_xy(cls.max_x) if x is None else x
        y = cls._generate_xy(cls.max_y) if y is None else y
        rotation = cls._generate_rotation("shape") if rotation is None else \
            rotation
        if radius is None:
            radius = cls._generate_radius(max_radius=cls.max_radius(x, y))
        if not no_texture and texture is None:
            texture = cls._sample(cls.textures)

        return cls(shape, color, x, y, radius, rotation, texture=texture)

    def perturb(self, features: List[str]) -> "Shape":
        kwargs = {}
        if "shape" in features:
            kwargs["shape"] = self._sample(self.shapes, exclude=self.shape)
        if "color" in features:
            kwargs["color"] = self._sample(self.colors, exclude=self.color)
        if "x" in features:
            kwargs["x"] = self._generate_xy(self.max_x, radius=self.radius)
        if "y" in features:
            kwargs["y"] = self._generate_xy(self.max_y, radius=self.radius)
        if "radius" in features:
            x = self.x if "x" not in kwargs else kwargs["x"]
            y = self.y if "y" not in kwargs else kwargs["y"]
            kwargs["radius"] = self._generate_radius(self.max_radius(x, y))
        if "rotation" in features:
            shape = self.shape if "shape" not in kwargs else kwargs["shape"]
            kwargs["rotation"] = self._generate_rotation(shape)
        if "texture" in features:
            kwargs["texture"] = self._sample(self.textures,
                                             exclude=self.texture)

        return self._replace(**kwargs)

    """ Draw the shape """

    def _draw_circle(self, drawer: ImageDraw):
        drawer.ellipse((self.x - self.radius, self.y - self.radius,
                        self.x + self.radius, self.y + self.radius),
                       fill=self.color())

    def _draw_polygon(self, drawer: ImageDraw, n: int):
        drawer.regular_polygon((self.x, self.y, self.radius),
                               n, rotation=self.rotation, fill=self.color())

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

    def draw(self, bg_color: Color = Color("gray")) -> Image:
        """
        Draws this shape.

        :param bg_color: The background color
        :return: The image with the shape in it
        """
        # Draw the basic shape
        shape_image = Image.new("RGB", canvas_size, bg_color())
        self._draw_shape(ImageDraw.Draw(shape_image))

        if self.texture is None:
            return shape_image

        # Draw texture
        mask = Image.new("RGBA", canvas_size, (0, 0, 0, 255))
        self._replace(color=Color(225, 225, 225, 86))._draw_shape(
            ImageDraw.Draw(mask))

        return Image.composite(shape_image, self.texture.image(), \
                               mask).convert("RGB")

    """ Save the shape to a file """

    def generate_filename(self) -> str:
        return "{}_{}_{}_x{}_y{}_r{}_theta{}.png".format(
            self.color.name, "" if self.texture is None else self.texture.name,
            self.shape, self.x, self.y, self.radius, self.rotation)

    def save(self, filename: Optional[Union[Path, str]] = None):
        filename = self.generate_filename() if filename is None else filename
        self.draw().save(filename)

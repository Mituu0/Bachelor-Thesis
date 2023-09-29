from webcolors import name_to_rgb
import numpy as np
import re
from PIL import Image, ImageDraw
import os
from os import listdir
import hashlib


def image_from_icon(image, shape_name, size, color, xy):
    # Open the icon.
    icon = Image.open("icons/{}.png".format(shape_name)).convert("RGBA")

    # Color the icon.
    data = np.array(icon)  # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability
    shape_areas = (red >= 0) & (blue >= 0) & (green >= 0)  # every pixel of the icon
    data[..., :-1][shape_areas.T] = name_to_rgb(color)
    icon = Image.fromarray(data)

    # Paste a resized version of the icon on the image.
    shape_size = shape_width, shape_height = size
    icon = icon.resize(shape_size, resample=Image.Resampling.NEAREST)
    image.paste(icon, xy, mask=icon)


def create_icon(shape_name):
    """ Takes a given image and turns it into an icon, meaning resizing it."""
    # Open the icon.
    icon = Image.open("icons/{}.png".format(shape_name)).convert("RGBA")

    # Color the icon red.
    icon = color_image(icon)

    # Save a resized version of the icon.
    shape_size = shape_width, shape_height = (10, 10)
    icon = icon.resize(shape_size, resample=Image.Resampling.NEAREST)

    # Make a specific pixel transparent.
    """if shape_name == "heart":
        pixels = icon.load()
        pixels[4, 1] = (0, 0, 0, 0)
        pixels[5, 1] = (0, 0, 0, 0)
        pixels[2, 6] = (0, 0, 0, 0)
        pixels[0, 8] = (0, 0, 0, 0)
        pixels[9, 8] = (0, 0, 0, 0)
        pixels[8, 9] = (0, 0, 0, 0)"""

    # Save the altered icon.
    icon.save("icons/{}.png".format(shape_name), "PNG")

    # Save the icon on a bigger background, for testing.
    image = Image.new('RGB', image_pixel, BACKGROUND_COLOR)
    shape_size = shape_width, shape_height = (16, 16)
    icon = icon.resize(shape_size, resample=Image.Resampling.NEAREST)
    x_pos, y_pos = int(IMAGE_WIDTH/2 - shape_width/2), int(IMAGE_HEIGHT/2 - shape_height/2)
    image.paste(icon, (x_pos, y_pos), icon)
    image.save("icons/{}_test.png".format(shape_name), "PNG")

    ## current smallest: for star: (8, 8)
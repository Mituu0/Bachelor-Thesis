"""Creates images with name being in the form
<shape>_<color>_<shape_height>_<shape_width>_<x_pos>_<y_pos>_<rotation_in_angles>.png"""
import numpy as np
from PIL import Image, ImageDraw
import os

import duplicate_handler
from icon_handler import image_from_icon
from settings import (IMAGE_PX, IMAGE_HEIGHT, IMAGE_WIDTH, SHAPES, COLORS, ROTATIONS, POSSIBLE_X_POS, POSSIBLE_Y_POS,
                      TEST_SHAPES, SIZES)

OUTPUT_DIR = 'new_dataset'

BACKGROUND_COLOR = 'black'

DELETE_DUPLICATES = False

NUM_POSITIONS = 1


def coordinates_by_shape(x_middle, y_middle, shape_height, shape_width):
    """
    Gives back a dictionary, containing the xy-coordinates for a shape given where the center of the shape on the
    image is and the size of the shape. Since changing the way images are created (now with pasting icons instead of
    using PIL.ImageDraw), the value for each shape is the same, namely the coordinates of the upper left corner of
    the shape.

    Args:
        x_middle (int): The x-coordinate of the center of the shape on the image.
        y_middle (int): The y-coordinate of the center of the shape on the image.
        shape_height (int): The height of the shape.
        shape_width (int): The width of the shape.

    Returns:
        dict[str: tuple[int]]: A dictionary with the shape name as key and the respective xy-coordinates in a tuple.
    """
    return {'rectangle': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'square': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'circle': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'ellipse': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'triangle': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'line': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'star': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'heart': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'menu': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'dots': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'x_sign': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'infinity': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'arrow': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'wifi': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'moon': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'x_circle': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'hashtag': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'cactus': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'sword': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1),
            'alien': (int(x_middle - shape_width / 2) + 1, int(y_middle - shape_height / 2) + 1)}


def all_sizes(random_size=False):
    """
    Returns a random size out of the sizes defined in the settings.py file.

    Args:
        random_size (bool, optional): If set, instead of a list with all sizes, a list with just one random size will be
            returned.

    Returns:
        list[tuple[int]]: A list containing all the sizes in tuples (width, height). If random_size was set, instead
            a list containing just one randomly chosen size will be returned.
    """

    if random_size:
        size = np.random.choice(SIZES)
        return [size]
    else:
        return SIZES


def random_pos():
    """
    Returns a random position out of the positions defined in the settings.py file.

    Returns:
        tuple[int]: A tuple containing two ints, one for the x and one for the y position.
    """
    pos = (np.random.choice(POSSIBLE_X_POS), np.random.choice(POSSIBLE_Y_POS))
    return pos


def create_symmetrical_images(random_size=False):
    """
    Creates symmetrical images and saves them in the folder 'symmetric' in the directory defined in OUTPUT_DIR. The
    shapes will always be positioned in the middle of the image, namely position (15, 15) if the image size is (32, 32).

    Args:
        random_size (bool, optional): If set, every shape will be created in just one random size, not in every possible
            one.
    """
    for shape in SHAPES:
        for color in COLORS:
            for i in range(NUM_POSITIONS):

                for rotation in ROTATIONS:

                    sizes = all_sizes(random_size=random_size)
                    for size in sizes:
                        create_image(shape, color, int(IMAGE_HEIGHT / 2 - 1), int(IMAGE_WIDTH / 2 - 1), size, rotation,
                                     'symmetric')


def create_asymmetrical_images(random_size=False, one_position=False):
    """
    Creates asymmetrical images and saves them in the folder 'asymmetric' in the directory defined in OUTPUT_DIR. The
    shape's position will be random for every individual image, expect argument one_position is set.

    Args:
        random_size (bool, optional): If set, every shape will be created in just a random size, not in every possible
            one.
        one_position (bool, optional): If set, every image will have the shape on the same position, here position
            (13, 11).
    """

    label = 'asymmetric'

    x_pos, y_pos = IMAGE_WIDTH / 2 - 1, IMAGE_HEIGHT / 2 - 1
    if one_position:
        x_pos, y_pos = 13, 11
        label = label[:-6] + "_oneposition"

    for shape in SHAPES:
        for color in COLORS:
            for i in range(NUM_POSITIONS):

                for rotation in ROTATIONS:

                    sizes = all_sizes(random_size=random_size)
                    for size in sizes:

                        if not one_position:
                            while x_pos == IMAGE_WIDTH / 2 - 1 or y_pos == IMAGE_HEIGHT / 2 - 1:
                                x_pos, y_pos = random_pos()

                        create_image(shape, color, x_pos, y_pos, size, rotation, label)


def create_image(shape, color, x_pos, y_pos, size, rotation, label, path=OUTPUT_DIR):
    """
    Creates and saves an image according to the given arguments.

    Args:
        shape (str): The name of the shape to paste on the image.
        color (str): The color of the shape.
        x_pos (int): The x-coordinate of the position of the shape (center of shape) on the image.
        y_pos (int): The y-coordinate of the position of the shape (center of shape) on the image.
        size (tuple[int]): A tuple containing two coordinates for the width and height of the image.
        rotation (int): The rotation of the shape.
        label (str): The folder the image will be saved in.
        path (str, optional): The directory, in which the folders with the images will be saved in.
    """

    shape_height, shape_width = size

    image = Image.new('RGB', IMAGE_PX, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)

    create_shape(shape, image, draw, coordinates_by_shape(x_pos, y_pos, shape_height, shape_width)[shape], size, color,
                 rotation)

    if not DELETE_DUPLICATES or DELETE_DUPLICATES and duplicate_handler.is_unique(image):
        image.save('{}/{}/{}_{}_{}°_{}px_at({},{}).png'.format(path, label, shape, color, rotation, shape_height,
                                                               x_pos, y_pos), 'PNG')


def create_shape(shape, image, draw, xy, size, color, rotation):
    """
    Pastes a shape on the given image.

    For the images drawn with ImageDraw, xy is the pixels to draw the shape (deprecated). For the shapes drawn from the
    icons, it is the left upper pixel, the shape should be placed on, on the image.

    Args:
        shape (str): The name of the shape to paste on the image.
        image (PIL.Image.Image): The Image instance the shape will be pasted on.
        draw (PIL.ImageDraw.ImageDraw, deprecated): The ImageDraw-instance to draw the shape on. Not used anymore, since
            the shapes get pasted on the image now with icons.
        xy (tuple[int]): A tuple containing the xy-coordinates of the shape (upper left corner) on the image.
        size (tuple[int]): A tuple containing two coordinates for the width and height of the image.
        color (str): The color of the shape.
        rotation (int): The rotation of the shape.
    """

    image_from_icon(image, '{}_{}px_{}'.format(shape, size[0], rotation), size, color, xy)


def create_sequence(path='sequences', shapes=SHAPES):
    """
    Creates and saves an image for every combination of shape, color, size, rotation and position on the image. Every
    individual shape gets a folder, with len(POSSIBLE_X_POS) * len(POSSIBLE_Y_POS) images, containing the shape on
    very possible position.

    Args:
        path (str, optional): The directory, in which the sub-folders with the images are saved.
        shapes (list[str]): The list of shape names to be created. Used mainly to differentiate between the 16 default
            shapes used for training and the 4 test shapes used for testing.
    """
    for color in COLORS:
        for shape in shapes:

            sizes = all_sizes()
            for size in sizes:

                for rotation in ROTATIONS:

                    folder_name = "{}_{}_{}px_{}°".format(shape, color, size[1], rotation)
                    # Create a folder for this shape with that color and this size and this rotation.
                    os.makedirs(os.path.join(path, folder_name))

                    for x in POSSIBLE_X_POS:
                        for y in POSSIBLE_Y_POS:
                            create_image(shape, color, x, y, size, rotation, folder_name, path=path)


if __name__ == '__main__':
    create_sequence(path="test_data/", shapes=TEST_SHAPES)
    # create_sequence()
    """
    # To prepare the icons
    for shape in SHAPES:
        create_icon(shape)
    rotate_images(path="icons")
    
    # Create the sequence
    create_sequence()
    """

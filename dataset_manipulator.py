from webcolors import name_to_rgb
import numpy as np
import re
from PIL import Image
import os
import duplicate_handler

from settings import ROTATIONS


def rotate_images(path, rotations, make_edges_transparent=True, background_color='black', delete_duplicates=False,
                  replace_img=False):
    """
    Rotates every image in every subdirectory found in given path. If there are no subdirectories, just rotate every
    image found in path.

    Args:
        path (str): The directory, in which subdirectories with images are stored. Optionally the path, in which images
            are stored.
        rotations (list[int]): The angles in which the images will be rotated.
        make_edges_transparent (bool, optional): If the rotation results in the images having cut-outs on the corners
            (like with 45°), those areas are turned transparent, if set.
        background_color (str, optional): The color to fill the corners with, if new areas occur when rotating (like
            with 45°). Is ignored, it make_edges_transparent is set.
        delete_duplicates (bool, optional): To delete the image, if an identical one has already been seen.
        replace_img (bool, optional): If set, the initial file will be replaced by the rotation. If more than one
            angle is given, only the last one will be saved as all the others will be overwritten at every step.
    """
    # Get every subdirectory, here probably ['symmetric', 'asymmetric']
    # If there are no subdirectories, just use given path to find images.
    subdirectories = []
    try:
        subdirectories = next(os.walk(path))[1]
    except:
        subdirectories.append(os.path.split(path)[1])

    if not subdirectories:
        subdirectories.append(os.path.split(path)[1])

    for subdirectory in subdirectories:
        if subdirectory not in path:
            sub_path = os.path.join(path, subdirectory)
        else:
            sub_path = path

        for image in os.listdir(sub_path):

            for angle in rotations:

                open_image = Image.open(os.path.join(sub_path, str(image)))

                rotated_image = open_image.rotate(angle, expand=False, fillcolor=background_color)

                if make_edges_transparent:
                    rotated_image = turn_transparent(rotated_image)

                if not delete_duplicates or delete_duplicates and duplicate_handler.is_unique(rotated_image):

                    if replace_img:
                        rotated_image.save(os.path.join(sub_path, image), "PNG")
                    else:
                        rotated_image.save(sub_path + '/' + str(image)[:-4] + '_' + str(angle) + '.png')


def color_every_image(path):
    """
    Colors every image in a path (or in the subdirectories of same path) with the color defined in color_image.

    Args:
        path (str): The directory, in which subdirectories with images are stored. Optionally the path, in which images
            are stored.
    """
    # Get every subdirectory, here probably ['symmetric', 'asymmetric']
    # If there are no subdirectories, just use given path to find images.
    subdirectories = []
    try:
        subdirectories = next(os.walk(path))[1]
    except:
        subdirectories.append(os.path.split(path)[1])

    if not subdirectories:
        subdirectories.append(os.path.split(path)[1])

    for subdirectory in subdirectories:
        if subdirectory not in path:
            sub_path = os.path.join(path, subdirectory)
        else:
            sub_path = path

        for image in os.listdir(sub_path):
            open_image = Image.open(os.path.join(sub_path, str(image))).convert("RGBA")
            colored_image = color_image(open_image)
            colored_image.save(os.path.join(sub_path, str(image)), 'PNG')


def every_size(path, sizes):
    """
    Saves all images in the given path in all given sizes.

    Args:
        path (str): The directory, in which subdirectories with images are stored. Optionally the path, in which images
            are stored.
        sizes (list[tuple[int]]): The sizes, the images are resized to.
    """
    # Get every subdirectory, here probably ['symmetric', 'asymmetric']
    # If there are no subdirectories, just use given path to find images.
    subdirectories = []
    try:
        subdirectories = next(os.walk(path))[1]
    except:
        subdirectories.append(os.path.split(path)[1])

    if not subdirectories:
        subdirectories.append(os.path.split(path)[1])

    for subdirectory in subdirectories:
        if subdirectory not in path:
            sub_path = os.path.join(path, subdirectory)
        else:
            sub_path = path

        for image in os.listdir(sub_path):

            for size in sizes:
                resized_image = Image.open(os.path.join(sub_path, str(image))).convert("RGBA") \
                    .resize(size, resample=Image.Resampling.NEAREST)
                resized_image.save(sub_path + '/' + str(image)[:-4] + '_' + str(size[0]) + 'px.png')


def turn_transparent(image):
    """
    Turns every black pixel in an image transparent.

    Args:
        image (PIL.Image.Image): The opened image.

    Returns:
        PIL.Image.Image: The image with all initially black pixel turned transparent.
    """
    color = name_to_rgb('black')  # color to make transparent

    datas = image.getdata()
    new_data = []
    for item in datas:
        if item[0] == color[0] and item[1] == color[1] and item[2] == color[2]:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)

    image.putdata(new_data)

    return image


def color_image(icon):
    """
    Colors every pixel of the given image red. (Red was an arbitrary choice.)

    Args:
        icon (PIL.Image.Image): The opened image.

    Returns:
        PIL.Image.Image: The image with all pixels colored red.
    """
    color = name_to_rgb("red")

    # Color the icon red.
    data = np.array(icon)  # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability
    shape_areas = (red >= 0) & (blue >= 0) & (green >= 0)  # every pixel
    data[..., :-1][shape_areas.T] = color  # change RGB values and transpose back
    icon = Image.fromarray(data)

    return icon


def change_random_pixel(image):
    """
    Finds a random pixel in the border of the image and colors it black. Mainly to make images asymmetrical.

    Args:
        image (PIL.Image.Image): The opened image.

    Returns:
        PIL.Image.Image: The image with a random pixel colored black
    """
    color = name_to_rgb("black")  # the color, the random pixel is 

    data = np.array(image)  # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability
    shape_areas = (red > 0) | (blue > 0) | (green > 0)  # every pixel of the shape

    # find one pixel, belonging to the background (black), that is right next to the shape
    random_index = [np.random.choice(np.arange(shape_areas.shape[0])),
                    np.random.choice(np.arange(shape_areas.shape[1]))]
    while not (shape_areas[random_index[0], random_index[1]] and
               (not shape_areas[random_index[0] + 1, random_index[1]] or
                not shape_areas[random_index[0] - 1, random_index[1]] or
                not shape_areas[random_index[0], random_index[1] + 1] or
                not shape_areas[random_index[0], random_index[1] - 1])):
        random_index = [np.random.choice(np.arange(shape_areas.shape[0])),
                        np.random.choice(np.arange(shape_areas.shape[1]))]

    random_pixel = np.full(shape_areas.shape, False)
    random_pixel[random_index[0], random_index[1]] = True
    data[..., :-1][random_pixel.T] = color
    return Image.fromarray(data)


def make_shapes_asymmetrical(path, nr_px=1):
    """
    Makes every shape itself (found in path) in the image asymmetrical, by coloring one of the pixels on the border of
    the image black. The new image replaces the initial one.

    Args:
        path (str): The path in which the images are stored.
        nr_px (int, optionally): The number of pixels to color.
    """
    for image in os.listdir(path):
        open_image = Image.open(os.path.join(path, str(image))).convert("RGBA")

        for i in range(nr_px):
            open_image = change_random_pixel(open_image)

        open_image.save(os.path.join(path, str(image)), 'PNG')


def check_uniform_color(image):
    """
    Checks if all pixel, except for the black ones, have the same color in an image. Mainly to detect irregularities
    in the icons that would cause an asymmetry on a pixel level.

    Args:
        image (PIL.Image.Image): The opened image.

    Returns:
        bool: True, if all pixels found on the image (expect for black) have the same RGB values. False, otherwise.
    """
    data = np.array(image)  # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability
    shape_areas = (red > 0) | (blue > 0) | (green > 0)  # every pixel of the shape
    colored_pixel = data[shape_areas.T]
    same_color = np.all(colored_pixel == colored_pixel[0])

    return same_color


def check_all_images_uniform_color(path):
    """
    Checks every image in given path for having only one color, except for black. Will print out, how many images have
    been checked and which ones did not have a uniform color.

    Args:
        path (str): The path in which the images are stored. May contain subdirectories are image files directly.

    Returns:
        bool: True, if all pixels found on every image (expect for black) have the same RGB values in itself. False,
            otherwise.
    """
    uniform_color = True
    nr_images = 0

    # Get every subdirectory, here probably ['symmetric', 'asymmetric']
    # If there are no subdirectories, just use given path to find images.
    subdirectories = []
    try:
        subdirectories = next(os.walk(path))[1]
    except:
        subdirectories.append(os.path.split(path)[1])

    if not subdirectories:
        subdirectories.append(os.path.split(path)[1])

    for subdirectory in subdirectories:
        if subdirectory not in path:
            sub_path = os.path.join(path, subdirectory)
        else:
            sub_path = path

        for image in os.listdir(sub_path):

            image_path = os.path.join(sub_path, str(image))
            open_image = Image.open(image_path).convert("RGBA")

            if not check_uniform_color(open_image):
                print("Image {} does not have a uniform color.".format(image_path))
                uniform_color = False

            nr_images += 1

    print("Checked {} images.".format(nr_images))
    return uniform_color


def rename_file_180_degrees(file_path):
    """
    Renames a file by changing the (x, y) substring to the corresponding new coordinates, if the image was rotated 180°.
    Example: (9, 13) --> (21, 17). Caution: Hard coded.

    Args:
        file_path (str): The path of the file to rename.
    """
    file_name = os.path.split(file_path)[-1]

    x, y = extract_xy(file_name)

    cor_x = 15 + (15 - x)
    cor_y = 15 + (15 - y)

    renamed_file = file_path.replace("({},{})".format(x, y), "({},{})_rotated".format(cor_x, cor_y))
    os.rename(file_path, renamed_file)


def rename_files_180_degrees(path):
    """
    Renames all files found in path by changing the (x, y) substring to the corresponding new coordinates,
    if the image was rotated 180°. Example: (9, 13) --> (21, 17). Caution: Hard coded.

    Args:
        path (str): The path in which the files to rename are stored.
    """
    subdirectories = []
    try:
        subdirectories = next(os.walk(path))[1]
    except:
        subdirectories.append(os.path.split(path)[1])

    if not subdirectories:
        subdirectories.append(os.path.split(path)[1])

    for subdirectory in subdirectories:
        if subdirectory not in path:
            sub_path = os.path.join(path, subdirectory)
        else:
            sub_path = path

        for file in os.listdir(sub_path):
            rename_file_180_degrees(os.path.join(sub_path, file))


def extract_xy(file_name):
    """
    Extracts the xy-coordinates of a file, if the file name contains them in the structure '(x,y)'.

    Args:
        file_name (str): The file name, to extract the xy-values from.

    Returns:
        tuple[int]: A tuple containing the xy-coordinates in the form (x,y).
    """
    split = os.path.split(file_name)

    if not split[0]:
        file_name = split[-1]

    x = int(re.search('\((.+?),', file_name).group(1))
    y = int(re.search(',(.+?)\)', file_name).group(1))

    return x, y


if __name__ == "__main__":
    # make_shapes_asymmetrical("new_dataset/asym_oneposition_pxremoved", nr_px=4)

    # rotate_images("sequences_rotated_180", [180], make_edges_transparent=True, background_color='black', delete_duplicates=False, replace_img=True, rename_files=True)
    # rename_files_180_rotation("sequences_rotated_180")
    # print(check_all_images_uniform_color("icons/"))
    rotate_images("test/", ROTATIONS, make_edges_transparent=True, background_color='black', delete_duplicates=False,
                  replace_img=False, rename_files=False)

    """open_image = Image.open("/home/mhani/Documents/thesis/Prework/icons/moon_14px_135.png").convert("RGBA")
    #print(check_uniform_color(open_image))
    #rotate_images("icons/", [0, 45, 90, 135, 180, 225, 270, 315])
    #sizes = [(i, i) for i in [8, 10, 12, 14, 16]]
    #every_size(path="icons", sizes=sizes)
    #print(check_all_images_uniform_color("icons/"))
    #print(check_all_images_uniform_color("icons"))
    # make_shapes_asymmetrical("test_asym_dataset")

    remove_alpha_values

    rotate_images("sequences_rotated_180", [180], replace_img=True)
    x = int(re.search('\((.+?),', file.name).group(1))
    y = int(re.search(',(.+?)\)', file.name).group(1))
    angle = int(re.search('px_(.+?)°', entry.name).group(1))

    cor_x = 15 + (15 - x)
    cor_y = 15 + (15 - y)
    cor_angle = angle + 180 if angle < 180 else angle + 180 - 360 

    orig_file_path = os.path.join("sequences", entry.name, file.name)
    compare_file_path = os.path.join("sequences", entry.name.replace("_{}°".format(angle), "_{}°".format(cor_angle)), file.name.replace("({},{})".format(x, y), "({},{})".format(cor_x, cor_y)))
    """

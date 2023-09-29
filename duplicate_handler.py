# TODO: needs proper docstrings and maybe debugging (testing for sure needed)
import re
import os
from os import listdir
import hashlib
import io
from PIL import Image, ImageDraw


def is_unique(image, hashes):
    """
    Computes hash of given image and compares it with the hashes already collected in hashes list.
    Returns True, if the hash isn't found and appends it to the list.
    """
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    unique = image_hash not in hashes
    if unique:
        hashes.append(image_hash)
    return unique


def delete_duplicates(path, hashes, delete=True):
    """ Reads every image in the specified folder and deletes duplicates. """

    deleted = []

    # Get every subdirectory, here probably ['symmetric', 'asymmetric']
    subdirectories = next(os.walk(path))[1]

    if not subdirectories: subdirectories.append("")

    for subdirectory in subdirectories:
        subpath = os.path.join(path, subdirectory)

        for filename in os.listdir(subpath):
            file_path = os.path.join(subpath, filename)

            with open(file_path, 'rb') as image:
                image_hash = hashlib.md5(image.read()).hexdigest()
                if image_hash in hashes:
                    deleted.append(filename)
                    if delete: os.remove(file_path)
                else:
                    hashes.append(image_hash)

    print("Deleted {} files, because they were duplicates.".format(len(deleted)))
    return deleted


def delete_images_with_re(directory, reg_exp_str):
    """ Reads every image in the specified folder and deletes duplicates. """

    deleted = []

    # Get every subdirectory, here probably ['symmetric', 'asymmetric']
    subdirectories = next(os.walk(directory))[1]

    for subdirectory in subdirectories:
        path = os.path.join(directory, subdirectory)

        for filename in os.listdir(path):

            reg_exp = re.compile(reg_exp_str)

            if reg_exp.search(filename):
                file_path = os.path.join(path, filename)
                deleted.append(filename)
                os.remove(file_path)

    print("Deleted {} files, because they matched the regular expression.".format(len(deleted)))
    return deleted


def equal(img_one, img_two):
    """Compares two 'bytes' objects by calculating a hashvalue and returns True, if they're equal."""
    return hashlib.md5(img_one).hexdigest() == hashlib.md5(img_two).hexdigest()
    # Could've also simply used: return img_one == img_two


def symmetrically_equal(img_one, img_two):
    # Open the image and rotate it 180°.
    open_img_one = Image.open(img_one)
    rotated_img_one = open_img_one.rotate(180)

    # Saves the rotated image in a BytesIO.
    buffer = io.BytesIO()
    rotated_img_one.save(buffer, format="PNG")

    # See if their bytes are equal.
    with open(img_two, 'rb') as open_img_two:
        equality = equal(buffer.getvalue(), open_img_two.read())

    return equality


def check_sequences_for_symmetrical_equality(path="sequences"):
    for entry in os.scandir(path):
        for file in os.scandir(os.path.join(path, entry.name)):
            x = int(re.search('\((.+?),', file.name).group(1))
            y = int(re.search(',(.+?)\)', file.name).group(1))
            angle = int(re.search('px_(.+?)°', entry.name).group(1))

            cor_x = 15 + (15 - x)
            cor_y = 15 + (15 - y)
            cor_angle = angle + 180 if angle < 180 else angle + 180 - 360

            orig_file_path = os.path.join(path, entry.name, file.name)
            compare_file_path = os.path.join(path, entry.name.replace("_{}°".format(angle), "_{}°".format(cor_angle)),
                                             file.name.replace("_{}°".format(angle), "_{}°".format(cor_angle)).
                                             replace("({},{})".format(x, y), "({},{})".format(cor_x, cor_y)))

            result = symmetrically_equal(orig_file_path, compare_file_path)
            if not result:
                print(orig_file_path)
                print(compare_file_path)
                # break

    print('Done')


def check_icons_for_symmetrical_equality(path="icons"):
    for file in os.scandir(path):

        try:
            angle = int(re.search('px_(.+?).png', file.name).group(1))
        except:
            print("Found a file that doesn't match given description: {}".format(file.name))
            continue

        cor_angle = angle + 180 if angle < 180 else angle + 180 - 360

        orig_file_path = os.path.join(path, file.name)
        compare_file_path = os.path.join(path,
                                         file.name.replace("px_{}.png".format(angle), "px_{}.png".format(cor_angle)))

        result = symmetrically_equal(orig_file_path, compare_file_path)
        if not result:
            result = symmetrically_equal(orig_file_path, compare_file_path)
            print(orig_file_path)
            print(compare_file_path)

    print('Done')


if __name__ == "__main__":
    # print(symmetrically_equal("icons/ellipse_10px_45.png", "icons/ellipse_10px_45.png"))
    delete_duplicates("testdata", [], delete=False)
    #check_sequences_for_symmetrical_equality(path="test_sequences")

    """one = open("icons/dots_right.png", "rb").read()
    two = open("icons/dots_315.png", "rb").read()
    print(equal(one, two))"""

    # print(symmetrical("sequences/x_sign_yellow_16px_315°/x_sign_yellow_16px_at(15,21).png", "sequences/x_sign_yellow_16px_315°/x_sign_yellow_16px_at(15,15).png"))

import torch.nn as nn
from collections import namedtuple

IMAGE_HEIGHT, IMAGE_WIDTH = 32, 32
IMAGE_PX = (IMAGE_HEIGHT, IMAGE_WIDTH)

# 16 shapes, 8 rotations, 8 colors, 5 sizes, 49 possible positions of the shape on the image
SHAPES = ['rectangle', 'circle', 'square', 'ellipse', 'line', 'triangle', 'star', 'heart', 'menu', 'dots', 'x_sign',
          'infinity', 'arrow', 'wifi', 'moon', 'x_circle']
COLORS = ['red', 'blue', 'yellow', 'purple', 'grey', 'green', 'orange', 'white']
ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]
SIZES = [(i, i) for i in [8, 10, 12, 14, 16]]
POSSIBLE_X_POS = POSSIBLE_Y_POS = [9, 11, 13, 15, 17, 19, 21]
NUM_X_POS, NUM_Y_POS = len(POSSIBLE_X_POS), len(POSSIBLE_Y_POS)

# Some new shapes, to create a complete new dataset for testing.
TEST_SHAPES = ['hashtag', 'cactus', 'sword', 'alien']

LR = 1e-3
CRITERION = nn.MSELoss()

TRANSITION = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
IMAGE_STORAGE = namedtuple('ImageStorage', 'path')

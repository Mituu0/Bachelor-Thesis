import os
import gymnasium as gym
from gym import spaces
import numpy as np
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import autoencoder
from checkpoints_handler import load_checkpoint, save_checkpoint, save_heatmap, load_heatmap, save_loss, load_loss
from image_show import show_images, open_images, show_heatmap
from cuda import get_device
from settings import IMAGE_PX, IMAGE_HEIGHT, IMAGE_WIDTH, SHAPES, COLORS, ROTATIONS, CRITERION
from dataset_manipulator import extract_xy


device = get_device()


class Environment(gym.Env):

    def __init__(self,
                 autoencoder, directory, possible_pos, image_center,
                 steps_taken=0,
                 num_steps=15,
                 steps_reached_center=None,
                 path=None,
                 image_name=None,
                 image_history=None,
                 action_history=None,
                 total_image_history=None,
                 total_action_history=None,
                 loss_history=None,
                 delta_history=None,
                 ideal_delta = None,
                 latent_code=None,
                 beginning_position=None,
                 shape_center=None,
                 criterion=CRITERION
                 ):
        self.ae = autoencoder
        self.loss_history = loss_history
        self.delta_history = delta_history
        self.ideal_delta = ideal_delta
        self.latent_code = latent_code

        self.directory = directory
        self.path = path
        self.image_name = image_name
        self.image_history = image_history
        self.action_history = action_history
        self.total_image_history = total_image_history
        self.total_action_history = total_action_history

        self.possible_pos = possible_pos
        self._beginning_position = beginning_position
        self._shape_center = shape_center
        self._image_center = image_center
        # Observations are dictionaries with the shapes location.
        self.num_steps = num_steps
        self.steps_taken = steps_taken
        self.steps_reached_center = steps_reached_center
        self.observation_space = spaces.Dict({"shape": spaces.Box(0, 31, shape=(2,), dtype=int)})

        # We have 5 actions, corresponding to "right", "up", "left", "down", "stay"
        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([2, 0]),
            1: np.array([0, 2]),
            2: np.array([-2, 0]),
            3: np.array([0, -2]),
            4: np.array([0, 0])
        }

        self.criterion = criterion

    def _get_obs(self):
        """ Returns the latent code of the current image state."""
        return self.latent_code

    def _get_info(self):
        """ Return the manhattan distance between shape's position and center. """
        return {"distance": np.linalg.norm(self._shape_center - self._image_center, ord=1)}

    def reset(self, seed=None, options=None):
        """ Resets the episode by clearing all relevant attributes and creating a new random image. """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Add the last episode (if it was a full episode) to the total history of the env.
        if self.image_history and self.action_history:
            assert len(self.image_history) - 1 == len(self.action_history), "Number of action does not fit the number of images stored."
            if len(self.image_history) == self.num_steps + 1:  # when a full episode has been done
                self.set_total_image_history(self.image_history[:-1])
                self.set_total_action_history(self.action_history)

        # Clear all attributes for the new episode.
        self.loss_history = []
        self.image_history = []
        self.action_history = []
        self.delta_history = []
        self.steps_taken = 0
        self.ideal_delta = None
        self.steps_reached_center = None

        path = self.np_random.choice(os.listdir(self.directory))
        self.path = os.path.join(self.directory, path)

        # Choose the beginning shape's position randomly
        self._beginning_position = np.array(self.np_random.choice(self.possible_pos, size=2))
        #while np.array_equal(self._beginning_position, self._image_center):
        #    self._beginning_position = np.array(self.np_random.choice(self.possible_pos, size=2))
        self._shape_center = self._beginning_position

        # find the right image file and save the name
        for image in os.listdir(self.path):
            positional_str = "({},{})".format(self._shape_center[0], self._shape_center[1])
            if positional_str in image:
                self.image_name = image
        # TODO: optimize by choosing a file randomly and extracting the positional info from the file name

        assert self.image_name is not None, "Was not able to find a fitting image file."
        self.process_image(self.image_name)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """ A step the agent can take to change the state of the environment and generate a reward. """

        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid!"
        assert self._shape_center is not None, "Call reset before using step method."

        self.steps_taken += 1

        # Add action to action_history
        self.set_action_history(action)

        direction = self._action_to_direction[action]
        # To make sure, the shape stays on the possible positions
        new_shape_center = np.clip(self._shape_center + direction, np.min(self.possible_pos), np.max(self.possible_pos))
        old_positional_str = "({},{})".format(self._shape_center[0], self._shape_center[1])
        new_positional_str = "({},{})".format(new_shape_center[0], new_shape_center[1])
        self.image_name = self.image_name.replace(old_positional_str, new_positional_str)
        self._shape_center = new_shape_center

        self.process_image(self.image_name)

        # Save the number of steps it took to reach the center of the image the first time
        if np.array_equal(self._shape_center, self._image_center) and not self.steps_reached_center:
            self.steps_reached_center = self.steps_taken

        terminated = self.steps_taken >= self.num_steps  # an episode is done after the defined number of steps

        reward = 1 if self.delta_history[-1] > self.delta_history[-2] else 0
        #ideal_reward = self.ideal_delta # for normalization TODO: delete self.ideal_reward
        #reward = (self.delta_history[-1] - self.delta_history[-2]) / 10  # TODO: magic numbers

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def process_image(self, image_name):
        """ Process the image information by extracting them and embedding them into the environments attributes. """
        img_tensor, recon_loss, latent_code, delta, ideal_delta = self.extract_image_info(image_name)
        self.add_image_info(self.image_name, recon_loss, latent_code, delta, ideal_delta)

    def extract_image_info(self, image):
        """ Extract the relevant information of a given image. """

        # Convert the image to a tensor
        img_tensor = open_images([os.path.join(self.path, image)]).to(device)

        reconstructed_im = self.ae(img_tensor)
        loss = self.criterion(reconstructed_im, img_tensor)

        ideal_delta = energy(img_tensor)
        delta = 100 * ideal_delta - 2000 * loss  # TODO: magic numbers

        return img_tensor, loss, self.ae.latent_code, delta, ideal_delta

    def add_image_info(self, image_name, loss, latent_code, delta, ideal_delta):
        """ Adds given image information into respective attributes. """

        # Add image name to the image_history
        self.set_image_history(image_name)
        # Add reconstruction loss of image to the loss_history
        self.set_loss_history(loss)
        # Add latent_code
        self.set_latent_code(latent_code)
        # Add delta value to the delta_history
        self.set_delta_history(delta)
        # Add ideal_reward to ideal_reward.
        self.set_ideal_delta(ideal_delta)

    def set_delta_history(self, delta):
        self.delta_history.append(delta)
    
    def set_ideal_delta(self, ideal_delta):
        self.ideal_delta = ideal_delta

    def set_loss_history(self, loss):
        self.loss_history.append(loss)

    def set_latent_code(self, latent_code):
        self.latent_code = latent_code

    def set_image_history(self, image_name):
        self.image_history.append(os.path.join(self.path, image_name))
    
    def set_action_history(self, action):
        self.action_history.append(action)

    def set_total_image_history(self, image_history):
        if self.total_image_history is None:
            self.total_image_history = []
        self.total_image_history += image_history

    def set_total_action_history(self, action_history):
        if self.total_action_history is None:
            self.total_action_history = []
        self.total_action_history += action_history


def energy(image_tensor):
    """ Implements the energy term as the standard deviation, hence the contrast, of the image. """
    return torch.std(image_tensor)

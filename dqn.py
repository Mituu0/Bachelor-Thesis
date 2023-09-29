#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

import torch.nn as nn
import torch.nn.functional as F

from settings import IMAGE_STORAGE, TRANSITION


plt.ion()


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.indices = None  # mainly used to be able to extract the corresbonding images in ImageReplayMemory

    def push(self, *args):
        """Save a transition"""
        self.memory.append(TRANSITION(*args))

    def sample(self, batch_size):
        # return random.sample(self.memory, batch_size)
        self.set_indices(np.random.choice(len(self), size=batch_size, replace=False))
        return [self.memory[i] for i in self.indices]

    def set_indices(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.memory)


class ImageReplayMemory(ReplayMemory):

    def __init__(self, capacity):
        super().__init__(capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(IMAGE_STORAGE(*args))

    def sample(self, batch_size, indices):
        if indices is None:  # the case, when only the ae is trained
            return random.sample(self.memory, batch_size)
        else:
            return [self.memory[i] for i in indices]


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

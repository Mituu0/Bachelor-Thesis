#!/usr/bin/env python

import os
import numpy as np
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import autoencoder
from checkpoints_handler import load_checkpoint, save_checkpoint, save_heatmap, load_heatmap, save_loss, load_loss, \
    save_image_history, save_action_history, load_image_history, load_action_history, save_durations, load_durations
import dataset_loader
from dqn import DQN, ReplayMemory, ImageReplayMemory
from image_show import show_images, open_images, show_heatmap
from cuda import get_device
from settings import IMAGE_PX, IMAGE_HEIGHT, IMAGE_WIDTH, SHAPES, COLORS, ROTATIONS, IMAGE_STORAGE, TRANSITION
from dataset_manipulator import extract_xy
from environment import Environment

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# epsilon is the probability of choosing a random action
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


def joint_training(num_epochs, extension, round_nr=1):
    dqn_loss = []
    ae_loss = []

    print("Joint Training")
    for i_epoch in tqdm(range(num_epochs)):

        # Do one episode
        episode_step()

        # Perform one step of the optimization (on the policy network), only if memory has enough data
        if len(memory) < BATCH_SIZE:
            continue
        loss = dqn_optimization_step()
        dqn_loss.append(loss)

        # Optimize the autoencoder
        loss = ae_optimization_step()
        ae_loss.append(loss)

    ae_name = "ae_{}".format(extension) if round_nr == 1 else "ae_{}_round{}".format(extension, round_nr)
    dqn_name = "dqn_{}".format(extension) if round_nr == 1 else "dqn_{}_round{}".format(extension, round_nr)

    save_loss(torch.tensor(ae_loss), ae_name)
    save_loss(torch.tensor(dqn_loss), dqn_name)
    save_checkpoint(env.ae, ae_optimizer, ae_loss[-1], num_epochs, ae_name)
    save_checkpoint(policy_net, optimizer, dqn_loss[-1], num_epochs, dqn_name)

    save_histories(extension, round_nr)


def sequential_training(num_epochs, extension, round_nr=1):

    solo_ae_extension = "{}_soloaetraining".format(extension) if round_nr == 1 else "{}_soloaetraining_round{}".format(extension, round_nr)
    solo_dqn_extension = "{}_sequentialtraining".format(extension) if round_nr == 1 else "{}_sequentialtraining_round{}".format(extension, round_nr)

    print("Sequential Training")
    sole_ae_training(num_epochs=num_epochs, extension=solo_ae_extension)
    save_histories("{}_soloaetraining".format(extension), round_nr=round_nr)

    # "Reset" all the relevant data, to have seperate data for each solo training.
    global episode_durations
    episode_durations = []
    env.total_image_history = None
    env.total_action_history = None

    sole_dqn_training(num_epochs=num_epochs, extension=solo_dqn_extension)
    save_histories("{}_sequentialtraining".format(extension), round_nr=round_nr)


def sole_dqn_training(num_epochs, extension):
    dqn_loss = []

    print("Sole DQN Training")
    for i_epoch in tqdm(range(num_epochs)):

        # Do one episode
        episode_step()

        # Perform one step of the optimization (on the policy network), only if memory has enough data
        if len(memory) < BATCH_SIZE:
            continue

        loss = dqn_optimization_step()
        dqn_loss.append(loss)

    ae_name = "ae_{}".format(extension)
    dqn_name = "dqn_{}".format(extension)

    save_loss(torch.tensor(dqn_loss), dqn_name)
    save_checkpoint(env.ae, ae_optimizer, None, 0, ae_name)
    save_checkpoint(policy_net, optimizer, dqn_loss[-1], num_epochs, dqn_name)


def sole_ae_training(num_epochs, extension):
    ae_loss = []

    print("Sole Autoencoder Training")
    for i_epoch in tqdm(range(num_epochs)):

        # Do one episode
        episode_step()

        # Perform one step of the optimization, only if memory has enough data
        if len(memory) < BATCH_SIZE:
            continue

        # Optimize the autoencoder
        loss = ae_optimization_step()
        ae_loss.append(loss)

    ae_name = "ae_{}".format(extension)
    dqn_name = "dqn_{}".format(extension)

    save_loss(torch.tensor(ae_loss), ae_name)
    save_checkpoint(env.ae, ae_optimizer, ae_loss[-1], num_epochs, ae_name)
    save_checkpoint(policy_net, optimizer, None, 0, dqn_name)


def create_image_action_history(dqn_model, ae_model, test_data, device):
    # Only works properly if a SequentialSampler for the Datalaoder test_data is used!
    assert isinstance(test_data.sampler, torch.utils.data.sampler.SequentialSampler), "There has to be SequentialSampler (shuffle=False) for test_data, so that this function can work properly."
    image_history, latent_codes = autoencoder.test_for_latent_code(ae_model, test_data, device)
    action_history = torch.argmax(dqn_model(latent_codes), dim=1).tolist()

    return image_history, action_history


def heatmap_by_action(image_history, action_history, extension, n_actions=5):
    """
    Creates and saves one heatmap per action, in which the number of the respective action taken for every position
    is shown.

    Args:
        image_history (list[str]): A list of image file names, from where the position will be extracted.
        action_history (np.ndarray): An array, in which the actions are stored. The image_history and action_history
            have to be in the same order, meaning that the i-th image in image_history is connected to the i-th action
            in action_history.
        extension (str): The string that gets appended to the file name. The file name will have the form
            "action_matrix_<action_idx>_<extension>".
    """
    assert len(image_history) == len(action_history), "Number of images is not equal to the number of actions."

    action_matrix = np.zeros((n_actions, IMAGE_WIDTH, IMAGE_HEIGHT))

    for i in range(len(image_history) - 1):
        x, y = extract_xy(os.path.split(image_history[i])[-1])
        action = action_history[i]
        action_matrix[action, x, y] += 1

    for action_idx in range(action_matrix.shape[0]):
        save_heatmap(action_matrix[action_idx], "action_matrix_{}_{}".format(action_idx, extension))


def dqn_optimization_step():

    # Perform one step of the optimization (on the policy network)
    loss = optimize_model()

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)

    return loss


def ae_optimization_step():
    image_storage = image_memory.sample(BATCH_SIZE, memory.indices)
    train_data = open_images(IMAGE_STORAGE(*zip(*image_storage)).path).to(device)
    loss = autoencoder.train_step(env.ae, ae_optimizer, train_data, device)
    return loss


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False, extension=None):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
        plt.savefig("plots/plot_{}".format(extension))
        save_durations(durations_t.numpy(), "duration_{}".format(extension))
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration [num_steps]')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def save_histories(extension, round_nr):

    if round_nr == 1:
        image_history_file_name = 'image_history_{}'.format(extension)
        action_history_file_name = 'action_history_{}'.format(extension)
        heatmap_extension = extension
        plot_extension = extension
    else:
        image_history_file_name = 'image_history_{}_round{}'.format(extension, round_nr)
        action_history_file_name = 'action_history_{}_round{}'.format(extension, round_nr)
        heatmap_extension = '{}_round{}'.format(extension, round_nr)
        plot_extension = '{}_round{}'.format(extension, round_nr)

    save_image_history(env.total_image_history, image_history_file_name)
    save_action_history(env.total_action_history, action_history_file_name)

    image_history = load_image_history(image_history_file_name)
    action_history = load_action_history(action_history_file_name)

    heatmap_by_action(image_history, action_history, heatmap_extension)
    print('Complete')
    
    plot_durations(show_result=True, extension=plot_extension)
    plt.ioff()
    plt.show()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = TRANSITION(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss


def episode_step():
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = state.clone().detach().unsqueeze(0).to(device)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        # print(reward)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = observation.clone().detach().unsqueeze(0).to(device)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Store the images used
        image_memory.push(env.image_history[-1])

        # Move to the next state
        state = next_state

        if done:
            episode_durations.append(env.steps_reached_center if env.steps_reached_center else env.num_steps)
            plot_durations()
            # show the images
            # print(env.image_history)
            # print('loss_history: {}'.format(env.loss_history))

            # show every 20th episode
            """if (i_episode + 1) % (num_episodes // 10) == 0:
                image_tensors = open_images(env.image_history)
                show_images(torch.stack([image_tensors]))"""
            break


def test_dqn(extension="binary_reward_50kepochs_sequentialtraining", test_data_dir="testdata", round_nr=1):

    device = get_device()
    
    model_name = "dqn_{}_round{}".format(extension, round_nr) if round_nr > 1 else "dqn_{}".format(extension)

    dqn = DQN(64, 5).to(device)
    optimizer = torch.optim.AdamW(dqn.parameters(), lr=1e-4, amsgrad=True)
    dqn_model, _, _ = load_checkpoint(dqn, optimizer, model_name)
    ae = autoencoder.FranciscoAutoencoderWithTwoPools1x1x64FranVersion().to(device)
    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    ae_model, _, _ = load_checkpoint(ae, ae_optimizer, "ae_{}".format(extension))
    test_data = dataset_loader.load_dataset(path=test_data_dir, batch_size=1, shuffle=False)
    image_history, action_history = create_image_action_history(dqn_model, ae_model, test_data, device)

    if round_nr == 1:
        file_name = "{}_{}".format(extension, test_data_dir)
    else:
        file_name = "{}_{}_round{}".format(extension, test_data_dir, round_nr)

    save_image_history(image_history, "image_history_{}".format(file_name))
    save_action_history(action_history, "action_history_{}".format(file_name))

    heatmap_by_action(image_history, action_history, file_name)


if __name__ == "__main__":
    # if GPU is to be used
    device = get_device()

    model = autoencoder.FranciscoAutoencoderWithTwoPools1x1x64FranVersion().to(device)
    ae_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # ae, _, _ = load_checkpoint(model, ae_optimizer,
    #                           "ae_binary_reward_50kepochs_soloaetraining_round2.pt")
    # ae, _, _ = load_checkpoint(model, torch.optim.Adam(model.parameters(), lr=1e-3), "sequences_round3.pt")
    #env = Environment(ae, "sequences", np.array([9, 11, 13, 15, 17, 19, 21]), np.array([15, 15]))
    env = Environment(model, "sequences", np.array([9, 11, 13, 15, 17, 19, 21]), np.array([15, 15]))

    # Get number of actions from gym action spacebi
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(1000)
    image_memory = ImageReplayMemory(1000)

    #policy_net, _, _ = load_checkpoint(DQN(n_observations, n_actions).to(device), optimizer, "dqn_continous_reward_10kepochs")
    #target_net.load_state_dict(policy_net.state_dict())

    steps_done = 0

    extension = "binary_reward_50kepochs"

    #sole_dqn_training(2000, extension)
    #joint_training(num_epochs=100000, extension=extension, round_nr=1)
    sequential_training(num_epochs=50000, extension=extension, round_nr=5)

    save_histories(extension, round_nr=5)

   

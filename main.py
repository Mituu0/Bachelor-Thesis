#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

import dataset_loader
import image_show
from image_show import show_images, show_heatmap, show_loss_history, annotate_heatmap_with_arrows, annotate_heatmap_with_labels, show_surface_plot, show_durations, avg_loss_history, avg_durations
from checkpoints_handler import save_checkpoint, load_checkpoint, save_loss_history, load_loss_history, save_heatmap, \
    load_heatmap, load_np, load_durations, save_image_history, save_action_history, save_loss
from autoencoder import *
from autoencoder import EPOCHS
from cuda import get_device
from dqn import DQN
from joint_training import create_image_action_history, heatmap_by_action, test_dqn
from settings import CRITERION, LR, IMAGE_WIDTH, IMAGE_HEIGHT


def train_autoencoder_old(ae_names, ae_models, train_models=True, split=[0.8, 0.2], seed=None): #, limit_data=0):
    if len(ae_names) != len(ae_models):
        raise Exception("The number of given model names does not equal the number of models.")

    device = get_device()

    # Load the data and split it into a train and test set.
    # This is done here and not inside the loop, because of reproducability reasons with torch.manual_seed(0).
    dataloaders = []  # every entry of the shape (train_data, test_data) for the respective autoencoder
    for ae_name in ae_names:

        dataloaders.append(dataset_loader.dataset_by_name(ae_name, split=split, seed=seed))

    loaded_models = []
    test_dataloaders = []
    for i, ae_name in enumerate(ae_names):

        #torch.manual_seed(0)

        model = ae_models[i]
        train = dataloaders[i][0] if len(dataloaders[i]) == 2 else dataloaders[i]
        test = dataloaders[i][1] if len(dataloaders[i]) == 2 else dataloaders[i]

        test_dataloaders.append(test)

        # Initialize the optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        if device:
            model.to(device)

        if train_models:
            # Train and save the model(s).
            loss_history = train_one(model, optimizer, train, device)
            save_checkpoint(model, optimizer, loss_history[-1], EPOCHS, ae_name + ".pt")
            save_loss(loss_history, ae_name)

        # Load the model:
        loaded_models.append(
            load_checkpoint(model, optimizer, ae_name + ".pt"))  # appends a tuple (model, model_loss, model_epoch)

    #test_autoencoder(test_dataloaders, device, loaded_models=loaded_models, limit_data=limit_data)

    show_loss_history(ae_names, ae_models)


def train_autoencoder(ae_name, ae_model, train_data):

    device = get_device()

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=LR)

    ae_model.to(device)

    # Train and save the model(s).
    loss_history = train_one(ae_model, optimizer, train_data, device)
    save_checkpoint(ae_model, optimizer, loss_history[-1], EPOCHS, ae_name + ".pt")
    save_loss(loss_history, ae_name)


def test_autoencoder_old(test_data, device, ae_names=[], ae_models=[], loaded_models=[], limit_data=0):

    if not (ae_models or ae_names or loaded_models):
        print("No information of which models to use was given.")

    # load the models with the information given by ae_names, ae_models, if no loaded_models are given
    if not loaded_models:
        for i, ae_name in enumerate(ae_names):

            #torch.manual_seed(0)

            model = ae_models[i]

            # Initialize the optimizer.
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            if device:
                model.to(device)

            loaded_models.append(
                load_checkpoint(model, optimizer, ae_name + ".pt"))  # appends a tuple (model, model_loss, model_epoch)

    for i, model in enumerate(loaded_models):
        predict(model[0], test_data[i], device, limit_data=limit_data)


def test_autoencoder(ae_model, test_data, device, image_show=True, limit_data=0):

    ae_model.to(device)
    _, loss = predict(ae_model, test_data, device, image_show=image_show, limit_data=limit_data)

    print("The average loss of the tested autoencoder is: {}".format(loss))


def test_on_sequence(model_name, model, test_model=False, filter=None, data_path="sequences"):
    # To make results reproducable.
    # torch.manual_seed(0)

    # Initialize the optimizer(s).
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Load the model.
    model, _, _ = load_checkpoint(model, optimizer, model_name + ".pt")

    heatmap_name = "loss_{}_model_{}_{}".format(filter, model_name, data_path)

    # Put it on the GPU if possible.
    device = get_device()
    if device: model.to(device)

    if test_model:
        # Load the dataset with the sequences.
        if filter:
            data = dataset_loader.load_dataset(path=data_path, filter=str(filter))
        else:
            data = dataset_loader.load_dataset(path=data_path)

        targets = np.array(data.sampler.data_source.targets)

        num_classes = len(data.dataset.class_to_idx)
        loss = np.zeros((1, IMAGE_WIDTH, IMAGE_HEIGHT))

        # Get the loss array from every class and append them together
        print("Testing Autoencoder.")
        for i in tqdm(range(num_classes)):
            # TODO: make a assert, that checks, that a class really only contains one image per position, if more, the information would get overwritten

            indices = np.arange(len(targets))[targets == i].tolist()
            subset = Subset(data.dataset, indices)
            dataloader = DataLoader(subset)
            if loss.any():
                loss = np.append(loss, predict_by_pos(model, dataloader, device)[np.newaxis, ...], axis=0)
            else:
                loss = predict_by_pos(model, dataloader, device)[np.newaxis, ...]

        # Plot the results, once with the average loss, once with the summed loss.
        avg_loss = np.average(loss, axis=0)

        save_heatmap(avg_loss, heatmap_name)

    show_heatmap(heatmap_name)


def test_every_filter():

    SHAPES = ['rectangle', 'circle', 'square', 'ellipse', 'line', 'triangle', 'star', 'heart', \
              'menu', 'dots', 'x_sign', 'infinity', 'arrow', 'wifi', 'moon', 'x_circle']
    SHAPES = ['circle', 'square', 'heart']
    COLORS = ['red', 'blue', 'yellow', 'purple', 'grey', 'green', 'orange', 'white']
    ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]

    # test_on_sequence('mixed', FranciscoAutoencoderWithTwoPools1x1x64(), test_model=True)
    for shape in SHAPES:
        test_on_sequence('mixed_new_dataset_5px_removed_1x1x64', FranciscoAutoencoderWithTwoPools1x1x64(), test_model=True, filter="(?=^{})(?=.*(red|blue))".format(shape))
    return None
    for rotation in ROTATIONS:
        test_on_sequence('mixed_new_dataset_5px_removed_1x1x64', FranciscoAutoencoderWithTwoPools1x1x64(), test_model=True,
                         filter="{}°".format(rotation))
    return None
    for color in COLORS:
        test_on_sequence('sequences', FranciscoAutoencoderWithTwoPools1x1x64(), test_model=True, filter=color)    

    all_sizes = [(i, i) for i in [10, 12, 14, 16]]
    for size in all_sizes:
        test_on_sequence('sequences', FranciscoAutoencoderWithTwoPools1x1x64(), test_model=True,
                         filter="{}px".format(size[0]))


def main():

    #torch.manual_seed(2)

    device = get_device()

    ae_name = "ae_binary_reward_50kepochs_sequentialtraining_round5"
    model = FranciscoAutoencoderWithTwoPools1x1x64FranVersion()
    opiimizer = optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model, loss, epoch = load_checkpoint(model, optimizer, ae_name + ".pt")  # appends a tuple (model, model_loss, model_epoch)

    #train_data = dataset_loader.load_dataset(path="sequences")
    #test_data = dataset_loader.load_dataset(path="testdata")

    #ae_names = ["asym", "asym_round2", "asym_round3"]
    #show_loss_history(ae_names)
    #save_loss(avg_loss_history("asym", 3), "asym_avg")
    #show_loss_history(["sym_avg", "asym_avg"], abs_difference=False)

    #train_autoencoder(ae_name, model, train_data)
    #test_on_sequence('dqn_binary_reward_50kepochs_sequentialtraining_round3', DQN(64, 5), test_model=True, data_path="testdata")
    test_on_sequence(ae_name, model, test_model=True, filter=None, data_path="testdata")
    #test_dqn(extension="binary_reward_50kepochs_sequentialtraining", test_data_dir="testdata", round_nr=5)
    #test_autoencoder(model, test_data, device, image_show=False, limit_data=0)

    return None
    #show_heatmap("loss_rectangle(.*)_0°")
    #test_model = False

    print("\nAutoencoder with binary reward (5k epochs)")
    test_on_sequence('dqn_binary_reward_50kepochs', DQN(64, 5), test_model=True, data_path="testdata")
    test_on_sequence('dqn_binary_reward_50kepochs_2ndround', DQN(64, 5), test_model=True, data_path="testdata")
    test_on_sequence('dqn_binary_reward_50kepochs_3ndround', DQN(64, 5)(), test_model=True, data_path="testdata")
    #test_on_sequence('mixed_new_dataset_asymonepos_pxremoved_1x1x64FranVersion', FranciscoAutoencoderWithTwoPools1x1x64FranVersion(), test_model=True)
    return None
    test_on_sequence('sequences_1x1x64FranVersion', FranciscoAutoencoderWithTwoPools1x1x64FranVersion(), test_model=True)
    test_on_sequence('sequences_redblue_1x1x64FranVersion', FranciscoAutoencoderWithTwoPools1x1x64FranVersion(), test_model=True, filter="red|blue")
    return None
    test_on_sequence('mixed_new_dataset_5px_removed_1x1x64FranVersion', FranciscoAutoencoderWithTwoPools1x1x64FranVersion(), test_model=test_model, filter="arrow")
    test_on_sequence('mixed_new_dataset_5px_removed_1x1x64FranVersion', FranciscoAutoencoderWithTwoPools1x1x64FranVersion(), test_model=test_model, filter="^circle")
    test_on_sequence('mixed_new_dataset_5px_removed_1x1x64FranVersion', FranciscoAutoencoderWithTwoPools1x1x64FranVersion(), test_model=test_model, filter="x_circle")
    test_on_sequence('mixed_new_dataset_5px_removed_1x1x64FranVersion', FranciscoAutoencoderWithTwoPools1x1x64FranVersion(), test_model=test_model, filter="0°")
    return None


if __name__ == '__main__':
    plt.ioff()
    
    # Code for plots in Chapter 3.1 Initial Proof of Concept.
    """
    show_loss_history(["sym_avg", "asym_avg"], abs_difference=False)
    model = FranciscoAutoencoderWithTwoPools1x1x64FranVersion()
    sym_ae, _, _ = load_checkpoint(model, torch.optim.Adam(model.parameters(), lr=LR), "sym")
    test_autoencoder(sym_ae, dataset_loader.load_sym_dataset(split=[0.8, 0.2])[-1], get_device(), limit_data=1)

    model = FranciscoAutoencoderWithTwoPools1x1x64FranVersion()
    asym_ae, _, _ = load_checkpoint(model, torch.optim.Adam(model.parameters(), lr=LR), "asym")
    test_autoencoder(asym_ae, dataset_loader.load_asym_dataset(split=[0.8, 0.2])[-1], get_device(), limit_data=1)
    """
    

    # Code for plots in Chapter 3.2 Loss Respective To Position.
    """
    annotate_heatmap_with_labels("loss_None_model_mixed_nosplit_testdata")
    annotate_heatmap_with_labels("loss_None_model_mixed_nosplit_round2_testdata")
    annotate_heatmap_with_labels("loss_None_model_mixed_nosplit_round3_testdata")
    avg_loss_heatmap = image_show.avg_loss_heatmap("mixed_nosplit_testdata", 3)
    annotate_heatmap_with_labels("", loaded_heatmap=avg_loss_heatmap)
    show_surface_plot("", loaded_heatmap=avg_loss_heatmap)

    annotate_heatmap_with_labels("loss_None_model_sequences_testdata")
    annotate_heatmap_with_labels("loss_None_model_sequences_round2_testdata")
    annotate_heatmap_with_labels("loss_None_model_sequences_round3_testdata")
    avg_loss_heatmap = image_show.avg_loss_heatmap("sequences_testdata", 3)
    annotate_heatmap_with_labels("", loaded_heatmap=avg_loss_heatmap)
    show_surface_plot("", loaded_heatmap=avg_loss_heatmap)
    """

    # Code for plots in Chapter 3.3 Reinforcement Learner.
    """
    show_durations("soledqntrained_pretrainedae_sequences")
    show_durations("soledqntrained_pretrainedae_sequences_round2")
    show_durations("soledqntrained_pretrainedae_sequences_round3")
    avg_du, std_du = avg_durations("soledqntrained_pretrainedae_sequences", 3)
    show_durations("", loaded_durations=avg_du, num_to_avg_over=100, std=std_du)

    show_durations("soledqntrained_pretrainedae_mixedae")
    show_durations("soledqntrained_pretrainedae_mixedae_round2")
    show_durations("soledqntrained_pretrainedae_mixedae_round3")
    avg_du, std_du = avg_durations("soledqntrained_pretrainedae_mixedae", 3)
    show_durations("", loaded_durations=avg_du, num_to_avg_over=100, std=std_du)
    """

    # Code for plots in Chapter 3.4 Joint Training.
    """
    show_durations("binary_reward_50kepochs", num_to_avg_over=1000)
    show_durations("binary_reward_50kepochs_round2", num_to_avg_over=1000)
    show_durations("binary_reward_50kepochs_round3", num_to_avg_over=1000)
    avg_du, std_du = avg_durations("binary_reward_50kepochs", 3)
    show_durations("", loaded_durations=avg_du, num_to_avg_over=1000, std=std_du)
    print(avg_du)
    print(std_du)

    annotate_heatmap_with_labels("loss_None_model_ae_binary_reward_50kepochs_testdata", vmin=0.01, vmax=0.025)
    annotate_heatmap_with_labels("loss_None_model_ae_binary_reward_50kepochs_testdata_round2", vmin=0.01, vmax=0.025)
    annotate_heatmap_with_labels("loss_None_model_ae_binary_reward_50kepochs_testdata_round3", vmin=0.01, vmax=0.025)
    avg_heatmap = image_show.avg_loss_heatmap("ae_binary_reward_50kepochs_testdata", 3)
    annotate_heatmap_with_labels("", loaded_heatmap=avg_heatmap, vmin=0.01, vmax=0.025)

    annotate_heatmap_with_arrows("binary_reward_50kepochs_testdata", scaling= 0.5, vmin=0.01, vmax=0.025)
    annotate_heatmap_with_arrows("binary_reward_50kepochs_testdata_round2", scaling= 0.5, vmin=0.01, vmax=0.025)
    annotate_heatmap_with_arrows("binary_reward_50kepochs_testdata_round3", scaling= 0.5, vmin=0.01, vmax=0.025)
    avg_action_heatmaps = image_show.avg_action_matrices("binary_reward_50kepochs_testdata", 3)
    annotate_heatmap_with_arrows("", num_heatmaps=5, scaling= 0.5, loaded_loss_heatmap=avg_heatmap, loaded_action_heatmaps=avg_action_heatmaps, vmin=0.01, vmax=0.025)
    """
    
    # Code for plots in Chapter 3.5 Sequential Training.
    """
    show_durations("binary_reward_50kepochs_soloaetraining", num_to_avg_over=1000)
    show_durations("binary_reward_50kepochs_soloaetraining_round2", num_to_avg_over=1000)
    show_durations("binary_reward_50kepochs_soloaetraining_round3", num_to_avg_over=1000)
    avg_du, std_du = avg_durations("binary_reward_50kepochs_soloaetraining", 3)
    show_durations("", loaded_durations=avg_du, num_to_avg_over=1000, std=std_du)

    show_durations("binary_reward_50kepochs_sequentialtraining", num_to_avg_over=1000)
    show_durations("binary_reward_50kepochs_sequentialtraining_round2", num_to_avg_over=1000)
    show_durations("binary_reward_50kepochs_sequentialtraining_round3", num_to_avg_over=1000)
    avg_du, std_du = avg_durations("binary_reward_50kepochs_sequentialtraining", 3)
    show_durations("", loaded_durations=avg_du, num_to_avg_over=1000, std=std_du)

    annotate_heatmap_with_labels("loss_None_model_ae_binary_reward_50kepochs_sequentialtraining_testdata", vmin=0.01, vmax=0.025)
    annotate_heatmap_with_labels("loss_None_model_ae_binary_reward_50kepochs_sequentialtraining_testdata_round2", vmin=0.01, vmax=0.025)
    annotate_heatmap_with_labels("loss_None_model_ae_binary_reward_50kepochs_sequentialtraining_testdata_round3", vmin=0.01, vmax=0.025)
    avg_heatmap = image_show.avg_loss_heatmap("ae_binary_reward_50kepochs_sequentialtraining_testdata", 3)
    annotate_heatmap_with_labels("", loaded_heatmap=avg_heatmap, vmin=0.01, vmax=0.025)

    annotate_heatmap_with_arrows("binary_reward_50kepochs_sequentialtraining_testdata", scaling= 0.5, vmin=0.01, vmax=0.025)
    annotate_heatmap_with_arrows("binary_reward_50kepochs_sequentialtraining_testdata_round2", scaling= 0.5, vmin=0.01, vmax=0.025)
    annotate_heatmap_with_arrows("binary_reward_50kepochs_sequentialtraining_testdata_round3", scaling= 0.5, vmin=0.01, vmax=0.025)
    avg_action_heatmaps = image_show.avg_action_matrices("binary_reward_50kepochs_sequentialtraining_testdata", 3)
    annotate_heatmap_with_arrows("", scaling= 0.5, loaded_loss_heatmap=avg_heatmap, loaded_action_heatmaps=avg_action_heatmaps, vmin=0.01, vmax=0.025)
    """

    # Code for plots in Appendix: Optimal Number of Steps.
    """
    zero_array = np.zeros((7, 7))
    optimal_num_steps = np.array([
        [6, 5, 4, 3, 4, 5, 6],
        [5, 4, 3, 2, 3, 4, 5],
        [4, 3, 2, 1, 2, 3, 4],
        [3, 2, 1, 0, 1, 2, 3],
        [4, 3, 2, 1, 2, 3, 4],
        [5, 4, 3, 2, 3, 4, 5],
        [6, 5, 4, 3, 4, 5, 6]])

    fig, ax = plt.subplots()
    im = ax.imshow(zero_array)
    ax = plt.gca()

    ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True)

    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    labels = optimal_num_steps

    for x in range(optimal_num_steps.shape[0]):
        for y in range(optimal_num_steps.shape[1]):
            text = ax.text(y, x, np.round(labels[x, y], 2), ha="center", va="center", color="w")

    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    
    fig.tight_layout()
    plt.show()
    """

    # Code for plots in Appendix: Individual Seeds for Sequential Training
    """
    annotate_heatmap_with_labels("loss_None_model_ae_binary_reward_50kepochs_sequentialtraining_testdata", vmin=0.01,
                                 vmax=0.025, patch=True)
    annotate_heatmap_with_labels("loss_None_model_ae_binary_reward_50kepochs_sequentialtraining_testdata_round2",
                                 vmin=0.01, vmax=0.025, patch=True)
    annotate_heatmap_with_labels("loss_None_model_ae_binary_reward_50kepochs_sequentialtraining_testdata_round3",
                                 vmin=0.01, vmax=0.025, patch=True)
    annotate_heatmap_with_labels("loss_None_model_ae_binary_reward_50kepochs_sequentialtraining_testdata_round5",
                                 vmin=0.01, vmax=0.025, patch=True)
    annotate_heatmap_with_labels("loss_None_model_ae_binary_reward_50kepochs_sequentialtraining_testdata_round4",
                                 vmin=0.01, vmax=0.025, patch=True)
    avg_heatmap = image_show.avg_loss_heatmap("ae_binary_reward_50kepochs_sequentialtraining_testdata", 5)
    annotate_heatmap_with_labels("", loaded_heatmap=avg_heatmap, vmin=0.01, vmax=0.025, patch=True)

    annotate_heatmap_with_arrows("binary_reward_50kepochs_sequentialtraining_testdata", scaling=0.5, vmin=0.01,
                                 vmax=0.025, patch=True)
    annotate_heatmap_with_arrows("binary_reward_50kepochs_sequentialtraining_testdata_round2", scaling=0.5, vmin=0.01,
                                 vmax=0.025, patch=True)
    annotate_heatmap_with_arrows("binary_reward_50kepochs_sequentialtraining_testdata_round3", scaling=0.5, vmin=0.01,
                                 vmax=0.025, patch=True)
    annotate_heatmap_with_arrows("binary_reward_50kepochs_sequentialtraining_testdata_round4", scaling=0.5, vmin=0.01,
                                 vmax=0.025, patch=True)
    annotate_heatmap_with_arrows("binary_reward_50kepochs_sequentialtraining_testdata_round5", scaling=0.5, vmin=0.01,
                                 vmax=0.025, patch=True)
    avg_action_heatmaps = image_show.avg_action_matrices("binary_reward_50kepochs_sequentialtraining_testdata", 5)
    annotate_heatmap_with_arrows("", scaling=0.5, loaded_loss_heatmap=avg_heatmap,
                                 loaded_action_heatmaps=avg_action_heatmaps, vmin=0.01, vmax=0.025, patch=True)
    """
    

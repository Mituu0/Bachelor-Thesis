import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
from checkpoints_handler import load_heatmap, load_checkpoint, load_loss, load_durations
import matplotlib.patches as patches

from settings import LR, IMAGE_WIDTH, IMAGE_HEIGHT, POSSIBLE_X_POS, POSSIBLE_Y_POS, NUM_X_POS, NUM_Y_POS

V_MIN = 0.005
V_MAX = 0.01


def show_images(image_data, compare_with=None, device=None):

    figsize = (20, 7)
    fig = plt.figure(figsize=figsize)

    nrows = 4
    ncols = 10
    for batch in image_data: 

        if isinstance(image_data, torch.utils.data.DataLoader) and len(batch) == 2:
            batch, _ = batch  # cancels out the labels

        pos_index = 0
        for image_idx in range(len(batch)):

            pos_index += 1

            # Add an Axes to the figure as part of a subplot arrangement, params: (nrows, ncols, index, **kwargs)
            ax = fig.add_subplot(nrows, ncols, pos_index, xticks=[], yticks=[])

            if device:
                    image = batch[image_idx].cpu().permute(1, 2, 0).detach().numpy()
            else:
                image = batch[image_idx].permute(1, 2, 0).detach().numpy()

            plt.imshow(image)

            if compare_with is not None:
                pos_index += 1
            
            if pos_index + 1 >= nrows * ncols:
                break  # break out of inner loop

        if pos_index + 1 >= nrows * ncols:
            break  # break out of outer loop
    
    if compare_with is not None:

        for batch in compare_with:

            pos_index = 1
            for image_idx in range(len(batch)):

                pos_index += 1

                # Add an Axes to the figure as part of a subplot arrangement, params: (nrows, ncols, index, **kwargs)
                ax = fig.add_subplot(nrows, ncols, pos_index, xticks=[], yticks=[])
                
                if device:
                    image = batch[image_idx].cpu().permute(1, 2, 0).detach().numpy()
                else:
                    image = batch[image_idx].permute(1, 2, 0).detach().numpy()
                    
                plt.imshow(image)

                pos_index += 1
                
                if pos_index >= nrows * ncols:
                    break

    plt.show()


def show_images_in_two_plots(image_data, compare_with=None, device=None):

    figsize = (10, 10)
    fig = plt.figure(figsize=figsize)

    nrows = 4
    ncols = 5
    for batch in image_data:

        if isinstance(image_data, torch.utils.data.DataLoader) and len(batch) == 2:
            batch, _ = batch  # cancels out the labels

        pos_index = 0
        for image_idx in range(len(batch)):

            pos_index += 1

            # Add an Axes to the figure as part of a subplot arrangement, params: (nrows, ncols, index, **kwargs)
            ax = fig.add_subplot(nrows, ncols, pos_index, xticks=[], yticks=[])

            if device:
                image = batch[image_idx].cpu().permute(1, 2, 0).detach().numpy()
            else:
                image = batch[image_idx].permute(1, 2, 0).detach().numpy()

            plt.imshow(image)

            if pos_index >= nrows * ncols:
                break  # break out of inner loop

        if pos_index >= nrows * ncols:
            break  # break out of outer loop
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=figsize)

    if compare_with is not None:

        for batch in compare_with:

            pos_index = 0
            for image_idx in range(len(batch)):

                pos_index += 1

                # Add an Axes to the figure as part of a subplot arrangement, params: (nrows, ncols, index, **kwargs)
                ax = fig.add_subplot(nrows, ncols, pos_index, xticks=[], yticks=[])

                if device:
                    image = batch[image_idx].cpu().permute(1, 2, 0).detach().numpy()
                else:
                    image = batch[image_idx].permute(1, 2, 0).detach().numpy()

                plt.imshow(image)

                if pos_index >= nrows * ncols:
                    break
    plt.tight_layout()
    plt.show()


def open_images(image_paths):
    """Converts image paths into tensors. Needed for show_images and to put images into ANN."""
    tensors = []
    to_tensor = transforms.ToTensor()

    for image_path in image_paths:
        open_image = Image.open(image_path)
        tensors.append(to_tensor(open_image))

    # dataset = torch.utils.data.TensorDataset(tensors)

    # return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return torch.stack(tensors)


def show_heatmap(heatmap_name):

    loss = load_heatmap(heatmap_name)

    im = plt.imshow(loss, cmap='hot', interpolation='nearest')
    plt.title(heatmap_name)
    plt.colorbar(im)
    plt.show()


def annotate_heatmap_with_arrows(heatmap_extension, scaling=1, num_heatmaps=5, loaded_loss_heatmap=None, loaded_action_heatmaps=None, vmin=V_MIN, vmax=V_MAX, patch=False):

    if loaded_action_heatmaps is None:
        # Load all heatmaps fitting the given extension and append them into one matrix, where the idx corresponds to
        # the action in env.action_space.
        action_data = np.zeros((num_heatmaps, IMAGE_WIDTH, IMAGE_HEIGHT))
        for action_idx in range(num_heatmaps):
            action_data[action_idx] = load_heatmap("action_matrix_{}_{}".format(str(action_idx), heatmap_extension))
    else:
        assert len(loaded_action_heatmaps) == num_heatmaps, ("The number of action heatmaps given in "
                                                             "loaded_action_heatmaps does not equal num_heatmaps.")
        action_data = np.array([loaded_action_heatmaps[action_idx] for action_idx in range(len(loaded_action_heatmaps))])

    # Resize the matrix by deleting all the zeros and save it in variable non_zero_action_data.
    non_zero_action_data = np.zeros((num_heatmaps, len(POSSIBLE_X_POS), len(POSSIBLE_Y_POS)))
    for action_idx in range(num_heatmaps):
        non_zero_action_data[action_idx] = resize_to_possible_pos(action_data[action_idx])

    # Get the sum of actions for each position.
    sum_matrix = np.sum(non_zero_action_data, axis=0)

    # Load the loss heatmap and resize it to the possible positions.
    if loaded_loss_heatmap is None:
        loss_heatmap = resize_to_possible_pos(load_heatmap("loss_None_model_ae_{}".format(heatmap_extension)))
    else:
        loss_heatmap = resize_to_possible_pos(loaded_loss_heatmap)

    # Show the loss heatmap.
    fig, ax = plt.subplots()
    im = plt.imshow(loss_heatmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))

    # Set the ticks, such the middle is (0, 0).
    start_tick, end_tick = - (NUM_X_POS // 2), NUM_X_POS // 2
    coordinates = np.arange(NUM_X_POS)
    ticks = np.arange(start_tick, end_tick + 1, 1)
    ax.set_xticks(coordinates)
    ax.set_yticks(coordinates)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    # If patch is True, draw a patch around the square with the minimal value
    if patch:
        ind = np.unravel_index(np.argmin(loss_heatmap, axis=None), loss_heatmap.shape)
        ind = (ind[1] - 0.5, ind[0] - 0.5)
        rect = patches.Rectangle(ind, 1, 1, linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    # Add in the arrows, which depend on actions taken in each position.
    for x in range(non_zero_action_data.shape[1]):
        for y in range(non_zero_action_data.shape[2]):

            xd = (non_zero_action_data[0, x, y] - non_zero_action_data[2, x, y]) / sum_matrix[x, y]
            yd = (non_zero_action_data[1, x, y] - non_zero_action_data[3, x, y]) / sum_matrix[x, y]

            plt.arrow(x, y, xd * scaling, yd * scaling, head_width=0.2, facecolor="w")

    fig.tight_layout()
    plt.show()


def annotate_heatmap_with_labels(heatmap_name, loaded_heatmap=None, vmin=V_MIN, vmax=V_MAX, patch=False):

    # Load the loss heatmap and resize it to the possible positions.
    if loaded_heatmap is not None:
        heatmap = loaded_heatmap
    else:
        heatmap = load_heatmap(heatmap_name)
    resized_heatmap = resize_to_possible_pos(heatmap)

    # Create the plot.
    fig, ax = plt.subplots()
    im = ax.imshow(resized_heatmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))

    # Arrange the labels.
    labels = resized_heatmap

    # A scaling variable, to control the explainability of the displayed values/ labels.
    scaling = 100

    for x in range(resized_heatmap.shape[0]):
        for y in range(resized_heatmap.shape[1]):
            text = ax.text(y, x, np.round(labels[x, y] * scaling, 2), ha="center", va="center", color="w")

    # Set the ticks, such the middle is (0, 0).
    start_tick, end_tick = - (NUM_X_POS // 2), NUM_X_POS // 2
    coordinates = np.arange(NUM_X_POS)
    ticks = np.arange(start_tick, end_tick + 1, 1)
    ax.set_xticks(coordinates)
    ax.set_yticks(coordinates)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    # If patch is True, draw a patch around the square with the minimal value
    if patch:
        ind = np.unravel_index(np.argmin(resized_heatmap, axis=None), resized_heatmap.shape)
        ind = (ind[1] - 0.5, ind[0] - 0.5)
        rect = patches.Rectangle(ind, 1, 1, linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    fig.tight_layout()
    plt.show()


def avg_action_matrices(extension, num_rounds, num_actions=5):
    """
    This function returns the action_matrices averages over every round/ seed.
    """
    # this function has to load 5 heatmaps, corresponding to the 5 actions,
    avg_action_heatmaps = []

    for action_idx in range(num_actions):
        loaded_heatmaps = []  # the loaded heatmaps per action
        for i_round in range(1, num_rounds + 1):
            # For every action, append the action matrices of every seed/ round
            if i_round == 1:
                loaded_heatmaps.append(load_heatmap("action_matrix_{}_{}".format(str(action_idx), extension)))
            else:
                try:
                    loaded_heatmaps.append(load_heatmap("action_matrix_{}_{}_round{}".format(str(action_idx), extension, i_round)))
                except FileNotFoundError:
                    splitted = extension.rsplit("_", 1)
                    loaded_heatmaps.append(load_heatmap("action_matrix_{}_{}_round{}_{}".format(str(action_idx), splitted[0], i_round, splitted[-1])))

        avg_heatmap = np.mean(np.array(loaded_heatmaps), axis=0)
        avg_action_heatmaps.append(avg_heatmap)

    return avg_action_heatmaps


def avg_loss_heatmap(extension, num_rounds):
    """
    Return a loss heatmap averages over every round/ seed.
    """
    # we assume the filter was None, and it was an autoencoder model
    loaded_heatmaps = []
    for i_round in range(1, num_rounds + 1):
        if i_round == 1:
            loaded_heatmaps.append(load_heatmap("loss_None_model_{}".format(extension)))
        else:
            try:
                loaded_heatmaps.append(load_heatmap("loss_None_model_{}_round{}".format(extension, i_round)))
            except FileNotFoundError:
                splitted = extension.rsplit("_", 1)
                loaded_heatmaps.append(load_heatmap("loss_None_model_{}_round{}_{}".format(splitted[0], i_round, splitted[-1])))

    avg_heatmap = np.mean(np.array(loaded_heatmaps), axis=0)
    return avg_heatmap


def avg_loss_history(name, num_rounds):
    loss_list = []
    for round in range(1, num_rounds + 1):
        if round == 1:
            loss_list.append(load_loss(name))
        else:
            loss_list.append(load_loss("{}_round{}".format(name, round)))
    avg_loss = np.mean(np.stack(loss_list), axis=0)
    return avg_loss


def avg_durations(duration_extension, num_rounds):
    durations_list = []
    for round in range(1, num_rounds + 1):
        if round == 1:
            durations_list.append(load_durations("duration_{}".format(duration_extension)))
        else:
            durations_list.append(load_durations("duration_{}_round{}".format(duration_extension, round)))
    durations = np.array(durations_list)
    avg_durations = np.mean(durations, axis=0)
    std = np.std(durations, axis=0)
    return avg_durations, std


def resize_to_possible_pos(array):

    resized_array = np.zeros((len(POSSIBLE_X_POS), len(POSSIBLE_Y_POS)))

    for i_x, x in enumerate(POSSIBLE_X_POS):
        for i_y, y in enumerate(POSSIBLE_Y_POS):
            resized_array[i_x, i_y] = array[x, y]

    return resized_array


def avg_over_array(array, num_to_avg_over):
    zero_fill = np.zeros(num_to_avg_over)
    means = np.array([np.mean(array[i:i + num_to_avg_over]) for i in range(len(array) - num_to_avg_over)])
    return np.concatenate((zero_fill, means))


def show_surface_plot(heatmap_name, loaded_heatmap=None):

    if loaded_heatmap is None:
        loss = load_heatmap(heatmap_name)
    else:
        loss = loaded_heatmap

    fig = plt.figure(figsize=(NUM_X_POS, NUM_Y_POS))
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(range(NUM_X_POS), range(NUM_Y_POS))
    loss_nozero = resize_to_possible_pos(loss)
    surf = ax.plot_surface(X, Y, loss_nozero, cmap=plt.cm.cividis)

    # Set the ticks, such the middle is (0, 0).
    start_tick, end_tick = - (NUM_X_POS // 2), NUM_X_POS // 2
    coordinates = np.arange(NUM_X_POS)
    ticks = np.arange(start_tick, end_tick + 1, 1)
    ax.set_xticks(coordinates)
    ax.set_yticks(coordinates)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    ax.axes.set_zlim3d(bottom=V_MIN, top=V_MAX)
    fig.tight_layout()
    plt.show()


def show_durations(duration_extension, loaded_durations=None, num_to_avg_over=100, std=None):

    if loaded_durations is None:
        # Load the numpy array containing the durations.
        durations = load_durations("duration_{}".format(duration_extension))
    else:
        durations = loaded_durations

    # Take the episode averages.
    assert len(durations) >= num_to_avg_over, "The number of values given for the durations is smaller than {}.".format(num_to_avg_over)
    means = avg_over_array(durations, num_to_avg_over)
    plt.plot(means, label="avg over {} epochs".format(num_to_avg_over))
    if std is not None:
        std = avg_over_array(std, num_to_avg_over)
        plt.fill_between(range(means.size), means - std, means + std, alpha=0.3)

    plt.title("Number of steps to find the center")
    plt.xlabel("Epoch")
    plt.ylabel("Duration [num_steps]")

    plt.tight_layout()
    plt.legend()
    plt.show()


def show_loss_history(names, abs_difference=True, avg_over=0):

    GRAPH_COLORS = ["b", "g", "r", "c"]  # to control the colors, also to make graph transluscent if avg_over parameter is given

    loss_histories = []

    for i, name in enumerate(names):

        # Load the loss history and plot it
        loss_history = load_loss(name)
        # TODO: loss_history = load_loss_history(name + ".txt")
        loss_histories.append(loss_history)

        if avg_over:
            plt.plot(loss_history, label=name, color=GRAPH_COLORS[i], alpha=0.2)
            means = avg_over_array(loss_history, avg_over)
            plt.plot(means, label="{}, avg over {} epochs".format(name, avg_over), color=GRAPH_COLORS[i])
        else:
            plt.plot(loss_history, label=name, color=GRAPH_COLORS[i])

    if len(loss_histories) == 2 and abs_difference:
        difference = np.abs(loss_histories[0] - loss_histories[1])
        max_difference_idx = np.argmax(difference) + 1
        print("The max difference between the two losses is at epoche {} with a difference of {}.".format(max_difference_idx, difference[max_difference_idx]))

        plt.plot(range(1, len(loss_history) + 1), difference, label="abs_difference")

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    #plt.yscale('log')

    plt.tight_layout()
    plt.legend()
    plt.show()

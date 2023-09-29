import torch
import os
import numpy as np
import typing_extensions

CHECKPOINT_PATH = "checkpoints/"
LOSS_HISTORY_PATH = "checkpoints/loss_history"
HEATMAP_PATH = "heatmaps/"
IMAGE_HISTORY_PATH = "action_history/image_history"
ACTION_HISTORY_PATH = "action_history/action_history"
DURATIONS_PATH = "durations/"

EXTENSION = ".pt"


def save_checkpoint(model, optimizer, loss, epochs, model_name, path=CHECKPOINT_PATH):
    """
    Saves a nn-model as a checkpoint in a .pt-file.

    Args:
        model (nn.Module): The neural network.
        optimizer (torch.optim.optimizer.Optimizer): The optimizer of the model.
        loss (int): The last loss recorded during training.
        epochs (int): The number of epochs the model was trained in.
        model_name (str): The name of the model. Used for the file name.
        path (str, optional): The directory, the checkpoint will be saved in.
    """
    model_name = append_ending(model_name)

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, os.path.join(path, model_name))

    print("Finished saving the model checkpoint '{}'.".format(model_name))


def load_checkpoint(model, optimizer, model_name, path=CHECKPOINT_PATH):
    """
    Loads a nn-model from a checkpoint in a .pt-file.

    Args:
        model (nn.Module): The neural network to load the checkpoint into.
        optimizer (torch.optim.optimizer.Optimizer): The optimizer of the model.
        model_name (str): The name of the model. Used for the file name.
        path (str, optional): The directory, the checkpoint is loaded from.

    Returns:
        nn.Module: The model, loaded with the weights and biases from the checkpoint.
        int: The last loss of the checkpoint.
        int: The number of epochs the checkpoint was trained in.
    """
    model_name = append_ending(model_name)

    checkpoint = torch.load(os.path.join(path, model_name), map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()

    return model, loss, epoch


def append_ending(model_name):
    """
    Appends the file extension defined in the constant EXTENSION to the given name, if not already there.

    Args:
        model_name(str): The name, the file extension should be appended to, if not already.

    Returns:
        str: The name with the extension.
    """
    if model_name[-3:] != EXTENSION:
        model_name += EXTENSION
    
    return model_name


@typing_extensions.deprecated("Saves loss_history as txt-file. Use save_loss(), to save as a numpy array instead.")
def save_loss_history(loss_history, file_name):
    """
    Saves a loss history as a txt-file.

    Args:
        loss_history (list[int]): The loss history to save.
        file_name (str): The name of the file the loss history should be saved in.
    """
    path = os.path.join(LOSS_HISTORY_PATH, file_name)

    with open(path, "w") as file:
        for loss in loss_history:
            file.write("{}\n".format(loss))
    print("Finished writing the loss_history into the file '{}'.".format(file_name))


@typing_extensions.deprecated("Saves loss_history as txt-file. Use load_loss(), to save as a numpy array instead.")
def load_loss_history(file_name):
    """
    Loads a loss history from a txt-file.

    Args:
        file_name (str): The name of the file the loss_history is saved in.

    Returns:
        np.ndarray: The loss history, converted to a numpy array.
    """
    path = os.path.join(LOSS_HISTORY_PATH, file_name)

    loss_history = []
    with open(path, "r") as file:
        for line in file:
            loss_history.append(float(line[:-1]))

    return np.array(loss_history)


def save_np(array, path):
    """
    Saves a numpy array in a .npy format.

    Args:
        array (np.ndarray): The numpy array to save.
        path (str): The path, the numpy array will be saved in.
    """
    with open(path, "wb") as file:
        np.save(file, array)


def load_np(path):
    """
    Loads a numpy array form a .npy file.

    Args:
        path (str): The path the .npy is loaded from.

    Returns:
        np.ndarray: The loaded array.
    """
    with open(path, "rb") as file:
        data = np.load(file)

    return data


def save_loss(loss_history, file_name, path=LOSS_HISTORY_PATH):
    """
    Saves a loss history as a .npy file.

    Args:
        loss_history (list[int]): The loss history to save. Can optionally be a numpy array.
        file_name (str): The name of the file the loss history should be saved in.
        path (str, optional): The directory, the loss is loaded from.
    """
    path = os.path.join(path, file_name)

    if torch.is_tensor(loss_history):
        if loss_history.is_cuda:
            loss_history = loss_history.cpu()
        if loss_history.requires_grad:
            loss_history = loss_history.detach().numpy()

    save_np(np.array(loss_history), path)
    print("Finished writing the loss information into a file '{}'.".format(file_name))


def load_loss(file_name, path=LOSS_HISTORY_PATH):
    """
    Loads a loss history from a .npy file.

    Args:
        file_name (str): The name of the file the loss history is saved in.
        path (str, optional): The directory the loss history is loaded from.

    Returns:
        np.ndarray: The loss history as a numpy array.
    """
    path = os.path.join(path, file_name)
    return load_np(path)


def save_heatmap(data, file_name, path=HEATMAP_PATH):
    """
    Saves the numpy array of a heatmap into a .npy file.

    Args:
        data (np.ndarray): The array with the data of the heatmap.
        file_name (str): The name of the file the heatmap should be saved in.
        path (str, optional): The directory, the heatmap is saved in.
    """
    path = os.path.join(path, file_name)
    save_np(np.array(data), path)
    print("Finished writing the heatmap into a file '{}'.".format(file_name))


def load_heatmap(file_name, path=HEATMAP_PATH):
    """
    Loads the array of a heatmap.

    Args:
        file_name (str): The name of the file the heatmap is saved in.
        path (str, optional): The directory the heatmap is loaded from.

    Returns:
        np.ndarray: The data of the heatmap as a numpy array.
    """
    path = os.path.join(path, file_name)
    return load_np(path)


def save_image_history(image_history, file_name, path=IMAGE_HISTORY_PATH):
    """
    Saves an image history as a txt-file.

    Args:
        image_history (list[str]): The image history to save.
        file_name (str): The name of the file the image history should be saved in.
        path (str, optional): The directory, the image history is saved in.
    """
    path = os.path.join(path, file_name)
    
    with open(path, 'w') as file:
        for item in image_history:
            file.write("%s\n" % item)
    
    print('Finished writing the image_history {} into a file.'.format(file_name))


def load_image_history(file_name, path=IMAGE_HISTORY_PATH):
    """
    Loads an image history from a txt-file into a list.

    Args:
        file_name (str): The name of the file the image history is saved in.
        path (str, optional): The directory, the image history is loaded from.

    Returns:
        list[str]: The image history as a list, where every image is a element.
    """
    path = os.path.join(path, file_name)

    image_history = []
    with open(path, 'r') as file:
        for line in file:
            x = line[:-1]  # remove linebreak /n
            image_history.append(x)

    return image_history


def save_action_history(action_history, file_name, path=ACTION_HISTORY_PATH):
    """
    Saves an action history as a npy file.

    Args:
        action_history (np.ndarray): The action history as a numpy array. Optionally a list with ints.
        file_name (str): The name of the file the action history should be saved in.
        path (str, optional): The directory, the action history is saved in.
    """
    path = os.path.join(path, file_name)
    save_np(np.array(action_history), path)
    print('Finished writing the action_history {} into a npy-file.'.format(file_name))


def load_action_history(file_name, path=ACTION_HISTORY_PATH):
    """
    Loads the action history from a npy file.

    Args:
        file_name (str): The name of the file the action history is saved in.
        path (str, optional): The directory the action history is loaded from.

    Returns:
        np.ndarray: The action history as a numpy array.
    """
    path = os.path.join(path, file_name)
    return load_np(path)


def save_durations(durations, file_name, path=DURATIONS_PATH):
    """
    Saves the durations history as a npy file.

    Args:
        durations (np.ndarray): The durations history as a numpy array. Optionally a list with ints.
        file_name (str): The name of the file the durations should be saved in.
        path (str, optional): The directory, the durations history is saved in.
    """
    path = os.path.join(path, file_name)
    save_np(np.array(durations), path)
    print('Finished writing the durations {} into a npy-file.'.format(file_name))


def load_durations(file_name, path=DURATIONS_PATH):
    """
    Loads the durations history from a npy file.

    Args:
        file_name (str): The name of the file the durations history is saved in.
        path (str, optional): The directory the durations history is loaded from.

    Returns:
        np.ndarray: The durations as a numpy array.
    """
    path = os.path.join(path, file_name)
    return load_np(path)

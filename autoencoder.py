import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm

from image_show import show_images, show_images_in_two_plots
from settings import LR, CRITERION, IMAGE_HEIGHT, IMAGE_WIDTH
from dataset_manipulator import extract_xy

EPOCHS = 200


def train_step(model, optimizer, train_data, device):
    """
    Performs a train step (one epoch) for the model with the given data.

    Args:
        model (nn.Module): The model to train.
        optimizer (torch.optim.optimizer.Optimizer): The optimizer of the model.
        train_data (torch.utils.data.DataLoader): The dataloader containing the data to train the model with.
        device (torch.Device): The device the model and train_data should be moved to.

    Returns:
        int: The training loss of this step.
    """
    train_loss = 0.0  # to monitor the training loss

    for data in train_data:

        if len(data) == 2:
            images, _ = data
        else:
            images = data

        # Move images to GPU if there is one present.
        if device:
            images = images.to(device)

        # print(images.shape)
        # print(label)
        optimizer.zero_grad()  # clear the gradients of all optimized variables
        output = model(images)

        loss = CRITERION(output, images)  # calculate the loss

        loss.backward()  # compute gradient of the loss with respect to model parameters
        optimizer.step()  # perform a singe optimization step (parameter update)
        train_loss += loss.item() * images.size(0)  # update running training loss

    train_loss = train_loss / len(train_data)

    return train_loss


def train_one(model, optimizer, train_data, device):
    """
    Trains the model with the given data. This function will call train_step() for every epoch.

    Args:
        model (nn.Module): The model to train.
        optimizer (torch.optim.optimizer.Optimizer): The optimizer of the model.
        train_data (torch.utils.data.DataLoader): The dataloader containing the data to train the model with.
        device (torch.Device): The device the model and train_data should be moved to.

    Returns:
        list[int]: The training history, contains the training loss of every step/ epoch.
    """
    outputs = []
    loss_history = []
    print("Training Autoencoder.")
    for epoch in tqdm(range(EPOCHS)):

        train_loss = train_step(model, optimizer, train_data, device)

        loss_history.append(train_loss)
        print('Epoch: {} \tReconstruction Loss: {:.6f}'.format(epoch, train_loss))

    return loss_history


def predict(model, test_data, device, image_show=True, limit_data=0):
    """
    Tests the model with the given data.

    Args:
        model (nn.Module): The model to test.
        test_data (torch.utils.data.DataLoader): The dataloader containing the data to test the model with.
        device (torch.Device): The device the model and test_data should be moved to.
        image_show (bool): If the reconstructions should be plotted for the user at the end.
        limit_data (int): The number of batches to test. If 0, there is no limit and every batch will be used for testing.

    Returns:
        list[nn.Tensor]: The reconstructed images.
        np.Array: The average loss over every tested batch. One dimensional array.
    """
    images_list = []  # necessary to do, because the Dataloader object test_data gives a different object everytime
    # when iterated through when shuffle=True
    reconstructed_im_list = []
    losses = np.empty(len(test_data))

    count = 0

    for i, batch in enumerate(test_data):
        images, _ = batch
        images_list.append(images)

        # Move images to GPU if there is one present.
        if device:
            images = images.to(device)

        reconstructed_im = model(images)
        reconstructed_im_list.append(reconstructed_im)

        # Calculate the loss.
        losses[i] = CRITERION(reconstructed_im, images)

        count += 1
        if limit_data and count >= limit_data:
            break

    # Make the list a tensor with 5 dimensions: (batch_idx, inside_batch_idx, color_channel, width, height)
    reconstructed_tensor = torch.stack(reconstructed_im_list)
    images_tensor = torch.stack(images_list)

    average_loss = np.average(losses)

    if image_show: show_images_in_two_plots(images_tensor, compare_with=reconstructed_tensor, device=device)

    return reconstructed_im_list, average_loss


def predict_by_pos(model, test_data, device):
    """
    Tests the model with the given data, respectively for every position. The batch size here must be equal to one for
    the function to give valid results. This function uses predict().

    Args:
        model (nn.Module): The model to test.
        test_data (torch.utils.data.DataLoader): The dataloader containing the data to test the model with. The batch
            size must be one.
        device (torch.Device): The device the model and test_data should be moved to.

    Returns:
        np.array: A matrix with the loss for every position. The row is equivalent to the x-coordinate for the image and
            the column for the y-coordinate. Hence, the array is of size (IMAGE_WIDTH, IMAGE_HEIGHT).

    Raises:
        AssertionError: If the batch size of the test_data is not equal to 1.
    """

    loss = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT))

    for i, batch in enumerate(test_data):
        image, _ = batch  # here, batch size is 1, so image only contains one image and label contains one label  # TODO: make assert
        file_name = os.path.split(test_data.dataset.dataset.samples[i][0])[-1]
        x_pos, y_pos = extract_xy(file_name)
        loss[x_pos, y_pos] = predict(model, [batch], device, image_show=False)[1]
    return loss


def test_for_latent_code(ae_model, test_data, device):
    """
    Gets test_data and returns the latent_code for every data point. TODO

    Args:
        ae_model (nn.Module): The model to test.
        test_data (torch.utils.data.DataLoader): The dataloader containing the data to test the model with. The batch
            size must be one.
        device (torch.Device): The device the model and test_data should be moved to.

    Returns:
        list[str]: The file names of the images that have been tested.
        list[np.Array]: The latent codes of the images tested.

    Raises:
        AssertionError: If the batch size of the test_data is not equal to 1.
    """
    images = []
    latent_codes = []

    print("Testing autoencoder for latent code.")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_data)):
            image, _ = batch  # here, batch size is 1, so image only contains one image and label contains one label
            assert len(image) == 1, "For this function to work properly, the batch_size must be 1."

            # Move image to GPU if there is one present.
            if device:
                image = image.to(device)

            file_name = os.path.split(test_data.dataset.samples[i][0])[-1]
            ae_model(image)  # ae reconstructs the image and saves the latent code as a attribute
            images.append(file_name)
            latent_codes.append(ae_model.latent_code)
            torch.cuda.empty_cache()

    return images, torch.stack(latent_codes)


class FranciscoAutoencoderWithTwoPools1x1x64(nn.Module):
    """
    An autoencoder class created to show a significant difference between 
    the reconstruction of symmetrical and asymmetrical images.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 8, stride=4),  # 32x32x3 -> 7x7x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 7x7x96 -> 3x3x96
            nn.Conv2d(96, 64, 1, stride=1),  # 3x3x96 -> 3x3x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 3x3x64 -> 1x1x64
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 96, 3, stride=2),  # 1x1x64 -> 3x3x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 3x3x96 -> 7x7x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 7x7x96 -> 15x15x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2),  # 15x15x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = torch.flatten(encoded)

        return decoded


class FranciscoAutoencoderWithTwoPools1x1x64WithPadding(nn.Module):
    """
    An autoencoder class created to show a significant difference between 
    the reconstruction of symmetrical and asymmetrical images.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 8, stride=4, padding=2),  # 32x32x3 -> 8x8x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 8x8x96 -> 4x4x96
            nn.Conv2d(96, 64, 4, stride=2, padding=1),  # 4x4x96 -> 2x2x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 2x2x64 -> 1x1x64
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 96, 3, stride=2),  # 1x1x64 -> 3x3x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 3x3x96 -> 7x7x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 7x7x96 -> 15x15x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2),  # 15x15x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = encoded

        return decoded


class FranciscoAutoencoderWithTwoPools1x1x32WithPadding(nn.Module):
    """
    An autoencoder class created to show a significant difference between 
    the reconstruction of symmetrical and asymmetrical images.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 8, stride=4, padding=2),  # 32x32x3 -> 8x8x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 8x8x96 -> 4x4x96
            nn.Conv2d(96, 32, 4, stride=2, padding=1),  # 4x4x32 -> 2x2x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 2x2x32 -> 1x1x32
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 96, 3, stride=2),  # 1x1x32 -> 3x3x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 3x3x96 -> 7x7x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 7x7x96 -> 15x15x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2),  # 15x15x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = encoded

        return decoded


class FranciscoAutoencoderWithTwoPools1x1x16WithPadding(nn.Module):
    """
    An autoencoder class created to show a significant difference between 
    the reconstruction of symmetrical and asymmetrical images.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 8, stride=4, padding=2),  # 32x32x3 -> 8x8x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 8x8x96 -> 4x4x96
            nn.Conv2d(96, 16, 4, stride=2, padding=1),  # 4x4x96 -> 2x2x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 2x2x16 -> 1x1x16
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 96, 3, stride=2),  # 1x1x16 -> 3x3x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 3x3x96 -> 7x7x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 7x7x96 -> 15x15x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2),  # 15x15x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = encoded

        return decoded


class FranciscoAutoencoderWithTwoPools1x1x128WithPadding(nn.Module):
    """
    An autoencoder class created to show a significant difference between 
    the reconstruction of symmetrical and asymmetrical images.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 8, stride=4, padding=2),  # 32x32x3 -> 8x8x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 8x8x96 -> 4x4x96
            nn.Conv2d(96, 128, 4, stride=2, padding=1),  # 4x4x96 -> 2x2x128
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 2x2x128 -> 1x1x128
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 3, stride=2),  # 1x1x128 -> 3x3x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 3x3x96 -> 7x7x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 7x7x96 -> 15x15x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2),  # 15x15x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = encoded

        return decoded


class FranciscoAutoencoderWithTwoPools1x1x64WithPaddingChangedDecoder(nn.Module):
    """
    An autoencoder class created to show a significant difference between 
    the reconstruction of symmetrical and asymmetrical images.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 8, stride=4, padding=2),  # 32x32x3 -> 8x8x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 8x8x96 -> 4x4x96
            nn.Conv2d(96, 64, 4, stride=2, padding=1),  # 4x4x96 -> 2x2x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 2x2x64 -> 1x1x64
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 96, 2, stride=2),  # 1x1x64 -> 2x2x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 4, stride=2),  # 2x2x96 -> 6x6x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 6, stride=2),  # 6x6x96 -> 16x16x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2, padding=1),  # 16x16x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = encoded

        return decoded


class FranciscoAutoencoderWithTwoPools1x1x64FranVersion(nn.Module):
    """
    An autoencoder class created to show a significant difference between the reconstruction of symmetrical and
    asymmetrical images.

    Attributes:
        encoder (nn.Sequential): The encoder part of the autoencoder.
        decoder (nn.Sequential): The decoder part of the autoencoder.
        latent_code (torch.Tensor): The latent code of the last forwarded image as a flattened array.
    """

    def __init__(self) -> None:
        """Initializes an autoencoder instance."""
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 10, stride=2),  # 32x32x3 -> 12x12x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 12x12x96 -> 6x6x96
            nn.Conv2d(96, 64, 4, stride=2),  # 6x6x96 -> 2x2x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 2x2x64 -> 1x1x64
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 96, 3, stride=2),  # 1x1x64 -> 3x3x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 3x3x96 -> 7x7x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 7x7x96 -> 15x15x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2),  # 15x15x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        """
        Forwards the input image through the network and saves the latent code as an attribute of the class.

        Args:
            input_image (torch.Tensor): The image as a 3x32x32 tensor to en- and decode.

        Returns:
            torch.Tensor: The reconstructed image as a 3x32x32 tensor.
        """
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = torch.flatten(encoded)

        return decoded


class FranciscoAutoencoderWithTwoPools1x1x32FranVersion(nn.Module):
    """
    An autoencoder class created to show a significant difference between 
    the reconstruction of symmetrical and asymmetrical images.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 10, stride=2),  # 32x32x3 -> 12x12x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 12x12x96 -> 6x6x96
            nn.Conv2d(96, 32, 4, stride=2),  # 6x6x96 -> 2x2x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 2x2x32 -> 1x1x32
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 96, 3, stride=2),  # 1x1x32 -> 3x3x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 3x3x96 -> 7x7x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 7x7x96 -> 15x15x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2),  # 15x15x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = encoded

        return decoded


class FranciscoAutoencoderWithTwoPools1x1x16FranVersion(nn.Module):
    """
    An autoencoder class created to show a significant difference between 
    the reconstruction of symmetrical and asymmetrical images.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 10, stride=2),  # 32x32x3 -> 12x12x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 12x12x96 -> 6x6x96
            nn.Conv2d(96, 16, 4, stride=2),  # 6x6x96 -> 2x2x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 2x2x16 -> 1x1x16
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 96, 3, stride=2),  # 1x1x16 -> 3x3x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 3x3x96 -> 7x7x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 7x7x96 -> 15x15x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2),  # 15x15x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = encoded

        return decoded


class FranciscoAutoencoderWithTwoPools1x1x128FranVersion(nn.Module):
    """
    An autoencoder class created to show a significant difference between 
    the reconstruction of symmetrical and asymmetrical images.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 10, stride=2),  # 32x32x3 -> 12x12x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 12x12x96 -> 6x6x96
            nn.Conv2d(96, 128, 4, stride=2),  # 6x6x96 -> 2x2x128
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 2x2x128 -> 1x1x128
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 3, stride=2),  # 1x1x128 -> 3x3x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 3x3x96 -> 7x7x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 7x7x96 -> 15x15x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2),  # 15x15x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = encoded

        return decoded


class FranciscoAutoencoderWithTwoPools1x1x64FranVersionChangedDecoder(nn.Module):
    """
    An autoencoder class created to show a significant difference between 
    the reconstruction of symmetrical and asymmetrical images.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 10, stride=2),  # 32x32x3 -> 12x12x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 12x12x96 -> 6x6x96
            nn.Conv2d(96, 64, 4, stride=2),  # 6x6x96 -> 2x2x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 2x2x64 -> 1x1x64
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 96, 2, stride=2),  # 1x1x64 -> 2x2x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 4, stride=2),  # 2x2x96 -> 6x6x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 6, stride=2),  # 6x6x96 -> 16x16x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2, padding=1),  # 16x16x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = encoded

        return decoded


class FranciscoAutoencoderWithTwoPools1x1x64FC(nn.Module):
    """
    An autoencoder class created to show a significant difference between 
    the reconstruction of symmetrical and asymmetrical images.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 8, stride=4, padding=1),  # 32x32x3 -> 7x7x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 7x7x96 -> 3x3x96
            nn.Conv2d(96, 64, 1, stride=1),  # 3x3x96 -> 3x3x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 3x3x64 -> 1x1x64
            nn.Flatten(start_dim=-3),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Unflatten(-1, (64, 1, 1)),
            nn.ConvTranspose2d(64, 96, 3, stride=2),  # 1x1x64 -> 3x3x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 3x3x96 -> 7x7x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 7x7x96 -> 15x15x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2),  # 15x15x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = encoded

        return decoded


class FranciscoAutoencoderWithTwoPools1x1x32(nn.Module):
    """
    An autoencoder class created to show a significant difference between 
    the reconstruction of symmetrical and asymmetrical images.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 8, stride=4),  # 32x32x3 -> 7x7x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 7x7x96 -> 3x3x96
            nn.Conv2d(96, 32, 1, stride=1),  # 3x3x96 -> 3x3x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 3x3x32 -> 1x1x32
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 96, 3, stride=2),  # 1x1x32 -> 3x3x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 3x3x96 -> 7x7x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 7x7x96 -> 15x15x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2),  # 15x15x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = encoded

        return decoded


class FranciscoAutoencoderWithTwoPools1x1x16(nn.Module):
    """
    An autoencoder class created to show a significant difference between 
    the reconstruction of symmetrical and asymmetrical images.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, 8, stride=4),  # 32x32x3 -> 7x7x96
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 7x7x96 -> 3x3x96
            nn.Conv2d(96, 16, 1, stride=1),  # 3x3x96 -> 3x3x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 3x3x16 -> 1x1x16
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 96, 3, stride=2),  # 1x1x16 -> 3x3x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 3x3x96 -> 7x7x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 96, 3, stride=2),  # 7x7x96 -> 15x15x96
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 4, stride=2),  # 15x15x96 -> 32x32x3
            nn.Tanh()
        )

        self.latent_code = None

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        self.latent_code = encoded

        return decoded

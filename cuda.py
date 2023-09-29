import torch


def get_device(print_device=False):
    """
    Gets the device, namely the CPU or GPU, depending on accessibility, preferably the GPU.

    Args:
        print_device (bool, optional): If set, the chosen device will be printed out on the console.

    Returns:
        torch.Device: The device found.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if print_device:
        print(device)
    return device

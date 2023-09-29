import torch
import re
import os
from typing import Any, Callable, Optional
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader

from random_split import random_split

DATA_PATH = 'dataset'
BATCH_SIZE = 256


def load_dataset(path=DATA_PATH, batch_size=BATCH_SIZE, shuffle=True, split=None, filter=None):
    """
    Returns an iterable and shuffled dataloader.

    Args:
        path (str): The directory from where the dataset is loaded.
        batch_size (int, optional): The batch size.
        shuffle (bool, optional): To use a RandomSamples or not.
        split (list[int], optional): The random train-test split of the dataset. If set, the function will return two
            dataloaders one for each sub-dataset. If none, the function will return only one dataloader with the whole
            dataset. Is implemented with random_split form the torch library.
        filter (str, optional): The filter the directories/ classes will be applied to. If set, the function will
            call a CustomImageFOlder, which is able to filter the classes by regular expression. If None, the
            ImageFolder class by torchvision will be used.

    Returns:
        Dataloader: An iterable dataloader containing the found images. If split was set, instead a tuple of two
            iterable dataloader will be returned.
    """
    transform = transforms.ToTensor()

    if filter:  # only get the classes, whose directory name fits the given filter, for customizing
        dataset = CustomImageFolder(path, filter, transform=transform)
    else:
        dataset = datasets.ImageFolder(path, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    if split:
        train_subset, test_subset = random_split(dataloader.dataset, split)
        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        return train_dataloader, test_dataloader

    return dataloader


def load_mixed_dataset(path=DATA_PATH, batch_size=BATCH_SIZE, split=None, filter=None):
    """
    Returns an iterable dataloader of all the data. This functions sole purpose is to call load_dataset with set path
    argument.

    Args:
        path (str, optional): The directory from where the dataset is loaded.
        batch_size (int, optional): The batch size.
        split (list[int], optional): The random train-test split of the dataset. If set, the function will return two
            dataloaders one for each sub-dataset. If none, the function will return only one dataloader with the whole
            dataset. Is implemented with random_split form the torch library.
        filter (str, optional): The filter the directories/ classes will be applied to. If set, the function will
            call a CustomImageFOlder, which is able to filter the classes by regular expression. If None, the
            ImageFolder class by torchvision will be used.

    Returns:
        Dataloader: An iterable dataloader containing the found images. If split was set, instead a tuple of two
            iterable dataloader will be returned.
    """
    return load_dataset(path=path, batch_size=batch_size, split=split, filter=filter)


def load_sym_dataset(path=DATA_PATH, batch_size=BATCH_SIZE, split=None, filter="^symmetric"):
    """
    Returns an iterable dataloader of all the sym dataset. This functions sole purpose is to call load_dataset with set
    path argument and set filter.

    Args:
        path (str, optional): The directory from where the dataset is loaded.
        batch_size (int, optional): The batch size.
        split (list[int], optional): The random train-test split of the dataset. If set, the function will return two
            dataloaders one for each sub-dataset. If none, the function will return only one dataloader with the whole
            dataset. Is implemented with random_split form the torch library.
        filter (str, optional): The filter the directories/ classes will be applied to. If set, the function will
            call a CustomImageFOlder, which is able to filter the classes by regular expression. If None, the
            ImageFolder class by torchvision will be used.

    Returns:
        Dataloader: An iterable dataloader containing the found images. If split was set, instead a tuple of two
            iterable dataloader will be returned.
    """
    return load_dataset(path=path, batch_size=batch_size, split=split, filter=filter)


def load_asym_dataset(path=DATA_PATH, batch_size=BATCH_SIZE, split=None, filter="asymmetric"):
    """
    Returns an iterable dataloader of all the asym dataset. This functions sole purpose is to call load_dataset with set
    path argument and set filter.

    Args:
        path (str, optional): The directory from where the dataset is loaded.
        batch_size (int, optional): The batch size.
        split (list[int], optional): The random train-test split of the dataset. If set, the function will return two
            dataloaders one for each sub-dataset. If none, the function will return only one dataloader with the whole
            dataset. Is implemented with random_split form the torch library.
        filter (str, optional): The filter the directories/ classes will be applied to. If set, the function will
            call a CustomImageFOlder, which is able to filter the classes by regular expression. If None, the
            ImageFolder class by torchvision will be used.

    Returns:
        Dataloader: An iterable dataloader containing the found images. If split was set, instead a tuple of two
            iterable dataloader will be returned.
    """
    return load_dataset(path=path, batch_size=batch_size, split=split, filter=filter)


def load_cifar10():
    """
    Loads the CIFAR10 dataset provided by torchvision.

    Returns:
        tuple[Dataloader]: Two dataloaders containing the train and test data respectively.
    """
    transform = transforms.ToTensor()
    training_data = datasets.CIFAR10(
        root="dataset/cifar10",
        train=True,
        download=True,
        transform=transform
    )

    testing_data = datasets.CIFAR10(
        root="dataset/cifar10",
        train=False,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader


def dataset_by_name(name, split=None, filter=None, seed=None):

    # To make results reproducable.
    if seed is not None: torch.manual_seed(seed)

    if 'mixed' in name:
        return load_mixed_dataset(split=split, filter=filter)
    elif 'asym' in name:
        return load_asym_dataset(split=split)
    elif 'sym' in name:
        return load_sym_dataset(split=split)
    elif 'sequences' in name:
        return load_dataset(path="sequences")
    elif 'cifar' in name:
        return load_cifar10()


class CustomImageFolder(datasets.ImageFolder):
    """
    A class to implement the filtering of classes to load with a regular expression.

    Attributes:
        root (str): The directory, in which the data is stored (in directories per class).
        filter (str): The regular expression the class names (directory names) will be filtered by.
    """

    def __init__(
            self,
            root: str,
            filter: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None
    ):
        """
        The constructor of the class.

        Args:
            root (str): The directory the dataset with folders is stored in.
            filter (str): The filter that is applied to the class names (folder names).
        """
        self.filter = filter
        super().__init__(root, transform, target_transform, loader, is_valid_file)

    def find_classes(self, directory):
        """
        The overridden function to implement the filter.

        Args:
            directory (str): The root directory in which the data with the classes are stored.

        Returns:
            list[str]: A list containing all classes fitting the filter
            directory[str: int]: A directory with the class names as keys and an ascending index as value.
        """
        regexp = re.compile(self.filter)

        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir() and regexp.search(entry.name))
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder matching the filter {self.filter}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

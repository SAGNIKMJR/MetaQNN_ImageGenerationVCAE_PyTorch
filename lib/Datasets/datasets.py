import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from lib.Datasets.preprocessing import Preprocessing
import preprocessing as Preprocessing
import math
import os
import struct
import gzip
import errno
import numpy as np

class CIFAR10:
    """
    CIFAR-10 dataset featuring tiny 32x32 color images of
    objects belonging to hundred different classes.
    Dataloader adapted from CIFAR10.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        train_transforms (torchvision.transforms): to Tensor
        val_transforms (torchvision.transforms): to Tensor
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        # Need to define the class dictionary by hand as the default
        # torchvision CIFAR10 data loader does not provide class_to_idx
        self.val_loader.dataset.class_to_idx = {'airplane': 0,
                                                'automobile': 1,
                                                'bird': 2,
                                                'cat': 3,
                                                'deer': 4,
                                                'dog': 5,
                                                'frog': 6,
                                                'horse': 7,
                                                'ship': 8,
                                                'truck': 9}

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            ])

        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.CIFAR10 to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.CIFAR10('datasets/CIFAR10/train/', train=True, transform=self.train_transforms,
                                    target_transform=None, download=True)
        valset = datasets.CIFAR10('datasets/CIFAR10/test/', train=False, transform=self.val_transforms,
                                  target_transform=None, download=True)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)


        return train_loader, val_loader

class MNIST:
    """
    MNIST dataset featuring gray-scale 28x28 images of
    hand-written characters belonging to ten different classes.
    Dataset implemented with torchvision.datasets.MNIST.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        train_transforms (torchvision.transforms): to Tensor
        val_transforms (torchvision.transforms): to Tensor
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        # Need to define the class dictionary by hand as the default
        # torchvision MNIST data loader does not provide class_to_idx
        self.val_loader.dataset.class_to_idx = {'0': 0,
                                                '1': 1,
                                                '2': 2,
                                                '3': 3,
                                                '4': 4,
                                                '5': 5,
                                                '6': 6,
                                                '7': 7,
                                                '8': 8,
                                                '9': 9}

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            ])

        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.MNIST to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.MNIST('datasets/MNIST/train', train=True, transform=self.train_transforms,
                                  target_transform=None, download=True)
        valset = datasets.MNIST('datasets/MNIST/test', train=False, transform=self.val_transforms,
                                target_transform=None, download=True)
        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader

class STL10:
    """
    STL10 dataset featuring tiny 96x96 color images of
    objects belonging to hundred different classes.
    Dataloader adapted from STL10.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        normalize (dict): Contains per-channel means and stds of the dataset.
        train_transforms (torchvision.transforms): to Tensor
        val_transforms (torchvision.transforms): to Tensor
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        # Need to define the class dictionary by hand as the default
        # torchvision STL10 data loader does not provide class_to_idx
        self.val_loader.dataset.class_to_idx = {'airplane': 0,
                                                'car': 1,
                                                'bird': 2,
                                                'cat': 3,
                                                'deer': 4,
                                                'dog': 5,
                                                'horse': 6,
                                                'monkey': 7,
                                                'ship': 8,
                                                'truck': 9}

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            ])

        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.STL10 to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.STL10('datasets/STL10/train/', split='train', transform=self.train_transforms,
                                    target_transform=None, download=True)
        valset = datasets.STL10('datasets/CIFAR10/test/', split = 'test', transform=self.val_transforms,
                                  target_transform=None, download=True)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)


        return train_loader, val_loader
import torch
import torchvision
import torchvision.transforms as transforms
from abc import ABC
import numpy as np
import math
from torch.utils.data import ConcatDataset


class Data(ABC):
    """
    Abstract base class to manage and partition datasets for federated learning experiments.
    Handles train/test loaders, data splits across clients, and class-based sample extraction.

    Attributes:
        num_classes (int): Number of output classes.
        size (tuple): Shape of each image.
        trainset (Dataset): PyTorch Dataset object for training.
        testset (Dataset): PyTorch Dataset object for testing.
        args (Namespace): Experiment arguments/configuration.
        client_train_idx (dict): Mapping from client ID to list of assigned sample indices.
    """

    def __init__(self, num_classes, size, trainset, testset, args):
        """
        Initialize the Data class with training and test sets and configuration.

        Args:
            num_classes (int): Number of output classes.
            size (tuple): Shape of the images (e.g., (32, 32, 3)).
            trainset (Dataset): Training dataset.
            testset (Dataset): Testing dataset.
            args (Namespace): Parsed arguments including batch_size, clients, etc.
        """
        self.args = args
        self.num_classes = num_classes
        self.image_shape = size
        self.trainset = trainset
        self.testset = testset
        self.client_train_idx = {}

    def init_train_split(self, client_ids):
        """
        Initialize a uniform data split for training across clients.

        Args:
            client_ids (list): List of client identifiers.
        """
        train_loader = self.get_trainloader(shuffle=False)
        class_idx = [[] for _ in range(self.num_classes)]
        self.client_train_idx = {c: [] for c in client_ids}
        idx = 0

        # Index all samples by class
        for data, labels in train_loader:
            for label in labels:
                class_idx[label].append(idx)
                idx += 1

        # Assign class samples to clients
        for i, c in enumerate(class_idx):
            np.random.shuffle(c)
            local_sizes = np.repeat(math.floor(len(c) // self.args.clients), self.args.clients)

            start_indices = [0] + [sum(local_sizes[:i]) for i in range(1, self.args.clients)]
            end_indices = [sum(local_sizes[:i + 1]) for i in range(self.args.clients)]

            for j, client_id in enumerate(client_ids):
                self.client_train_idx[client_id].extend(c[start_indices[j]: end_indices[j]])


    def get_testloader(self):
        """
        Returns a DataLoader for the test set.

        Returns:
            DataLoader: Batched test set loader.
        """
        batch_size = self.args.batch_size if "batch_size" in self.args else self.args.unlearn_batch_size
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        return testloader

    def get_client_set(self, clients):
        """
        Get a dataset for one or more clients based on their assigned training indices.

        Args:
            clients (list or object): A list of client objects or a single client.

        Returns:
            Subset: PyTorch dataset containing the union of selected clients' data.
        """
        if not isinstance(clients, list):
            clients = [clients]

        indices = []
        for client in clients:
            client.train_samples_idx = self.client_train_idx[client.cid]
            indices.extend(client.train_samples_idx)

        client_dataset = torch.utils.data.Subset(self.trainset, indices)
        return client_dataset

    def get_trainloader(self, clients=None, shuffle=True, add_datatset=None):
        """
        Returns a DataLoader for the training set. If clients are provided, filters data by their assigned indices.

        Args:
            clients (list or object): List of client objects or a single client. If None, use entire trainset.
            shuffle (bool): Whether to shuffle the data.
            add_datatset (Dataset, optional): An additional dataset to concatenate with the main set.

        Returns:
            DataLoader: DataLoader for training.
        """
        batch_size = self.args.batch_size if "batch_size" in self.args else self.args.unlearn_batch_size

        if not isinstance(clients, list) and clients is not None:
            clients = [clients]

        if clients:
            ds = self.get_client_set(clients)
            if add_datatset:
                ds = ConcatDataset([ds, add_datatset])
            return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        else:
            ds = self.trainset
            if add_datatset:
                ds = ConcatDataset([ds, add_datatset])
            trainloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=2)
            return trainloader

    def get_class_sample(self, label, size, batch_size=None):
        """
        Get a fixed number of samples of a given class label from the trainset.

        Args:
            label (int): Class label to sample from.
            size (int): Number of samples to retrieve.
            batch_size (int, optional): Batch size for returned DataLoader.

        Returns:
            DataLoader: Batched DataLoader of samples for the given label.
        """
        subset_indices = []
        for index in range(len(self.trainset)):
            img, lbl = self.trainset[index]
            if lbl == label:
                subset_indices.append(index)
                if len(subset_indices) == size:
                    break

        batch_size = batch_size or self.args.batch_size
        subset = torch.utils.data.Subset(self.trainset, subset_indices)
        return torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                           shuffle=False, num_workers=1)


class Cifar10(Data):
    """
    CIFAR-10 dataset loader and manager.

    Inherits from the abstract `Data` class and loads CIFAR-10 with optional normalization.
    """

    def __init__(self, args, normalize=True):
        transform_list = [transforms.ToTensor()]
        if normalize:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        super().__init__(num_classes=10, size=(32, 32, 3), trainset=trainset, testset=testset, args=args)


class MNIST(Data):
    """
    MNIST dataset loader and manager.

    Inherits from the abstract `Data` class and loads MNIST with optional normalization.
    """

    def __init__(self, args, normalize=True):
        transform_list = [transforms.ToTensor()]
        if normalize:
            transform_list.append(transforms.Normalize(0.5, 0.5))
        transform = transforms.Compose(transform_list)

        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        super().__init__(num_classes=10, size=(28, 28, 1), trainset=trainset, testset=testset, args=args)

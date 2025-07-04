import random
import numpy as np
import torch

from src.dataloader.data import Cifar10, MNIST
from src.dataloader.watermark_data import WatermarkDataset
from src.models.cifar10_models import VGG16, ResNet18
from src.models.mnist_models import MNIST_L5

class Client:
    def __init__(self, cid, train_samples_idx=None):
        self.cid = cid
        self.train_samples_idx = train_samples_idx


def setup(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # torch.use_deterministic_algorithms(mode=True)

    model = None
    match args.model_str:
        case "vgg16":
            model = VGG16
        case "mnist_l5":
            model = MNIST_L5
        case "resnet18":
            model = ResNet18

    data = None
    match args.dataset_str:
        case "cifar10":
            data = Cifar10(args)
        case "mnist":
            data = MNIST(args)

    watermark_dataset = WatermarkDataset(args, (data.image_shape[0], data.image_shape[1]),
                               data.num_classes, grayscale=data.image_shape[2] == 1)

    clients = [Client(f"c{i}") for i in range(args.clients)]
    data.init_train_split([f"c{i}" for i in range(0, args.clients)])
    client_set = data.get_client_set(clients[0])

    return model, data, watermark_dataset, client_set
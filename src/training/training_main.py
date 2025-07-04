import numpy as np
import torch
import os
from src.utils.logger import Logger
from src.dataloader.data import MNIST, Cifar10
from src.training.federated_training import FederatedExperiment
from src.models.cifar10_models import VGG16, ResNet18
from src.models.mnist_models import MNIST_L5
import argparse

DATASETS = ['cifar10', 'mnist']
DEVICES = ['cuda', 'cpu']
MODELS = ['vgg16', 'mnist_l5', 'resnet18']
BACKGROUND = ["random", "abstract", "black", "randomN"]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        help='name of dataset',
                        choices=DATASETS,
                        required=True)

    parser.add_argument('--model',
                        help='name of model',
                        choices=MODELS)

    parser.add_argument('--watermark',
                        help='enable watermarking',
                        action='store_true')

    parser.add_argument('--pretrain', help='use pretrained weights', action='store_true')

    parser.add_argument('--pattern',
                        help='pattern',
                        action='store_true')

    parser.add_argument('--background',
                        choices=BACKGROUND)

    parser.add_argument('--num_rounds',
                        help='# of communication rounds',
                        type=int)

    parser.add_argument('--clients_per_round',
                        help='# of selected clients per round',
                        type=int)

    parser.add_argument('--clients',
                        help='# of total clients',
                        type=int)

    parser.add_argument('--epoch',
                        help='# of epochs when clients train on data',
                        type=int)

    parser.add_argument('--max_watermark_epoch',
                        help='max # of epochs when server learns watermarks',
                        type=int)

    parser.add_argument('--watermark_threshold',
                        help='accuracy threshold for watermark retraining',
                        type=float)


    parser.add_argument('--batch_size',
                        help='batch size when clients train on data',
                        type=int)

    parser.add_argument('--lr',
                        help='learning rate for local optimizers',
                        type=float)

    parser.add_argument('--lr_pretrain',
                        help='learning rate for watermarking optimizer',
                        type=float)

    parser.add_argument('--lr_retrain',
                        help='learning rate for watermarking optimizer',
                        type=float)

    parser.add_argument('--momentum',
                        help='momentum for local optimizers',
                        type=float)

    parser.add_argument('--momentum_pretrain',
                        help='momentum for local optimizers',
                        type=float)

    parser.add_argument('--momentum_retrain',
                        help='momentum for local optimizers',
                        type=float)

    parser.add_argument('--decay',
                        help='weight decay - only used for mnist-l5',
                        type=float)

    parser.add_argument('--decay_pretrain',
                        help='weight decay - only used for mnist-l5',
                        type=float)

    parser.add_argument('--decay_retrain',
                        help='weight decay - only used for mnist-l5',
                        type=float)

    parser.add_argument('--seed',
                        help='random seed',
                        type=int)

    parser.add_argument('--device',
                        help='train on cuda or cpu',
                        choices=DEVICES)

    parser.add_argument('--model_checkpoint_rounds',
                        help='At these rounds the model is saved',
                        type=int,
                        nargs='+')

    parser.add_argument('--client_cpu',
                        help='number of cpus needed for client',
                        type=float)

    parser.add_argument('--num_cpu',
                        help='total number of cpus available for FL simulation',
                        default=8,
                        type=int)
    parser.add_argument('--num_gpu',
                        help='total number of gpus available for FL simulation',
                        default=1,
                        type=int)

    parser.add_argument('--client_cuda',
                        help='number of gpus needed for client',
                        type=float)

    parser.add_argument('--wm_size',
                        help='number of wm images',
                        type=int)
    parser.add_argument('--output_dir',
                        help='output directory for all model checkpoints and log files',
                        default="output/",
                        type=str)
    args = parser.parse_args()
    # Additional checks for dataset and model combination
    if args.dataset == 'mnist' and args.model != 'mnist_l5':
        raise ValueError("For 'mnist' dataset, only 'mnist_l5' model is valid.")
    elif args.dataset == 'cifar10' and args.model == 'mnist_l5':
        raise ValueError("For 'cifar10' dataset, 'mnist_l5' model is not valid.")

    args.client_cpu = args.client_cpu or 1
    args.client_cuda = args.client_cuda or 0

    # Set default values for relevant options only if watermark is provided
    if args.watermark:
        args.background = args.background or "random"
        args.wm_size = args.wm_size or 100

        if args.watermark:
            args.max_watermark_epoch = args.max_watermark_epoch or 100
            args.watermark_threshold = args.watermark_threshold or 0.98
            args.momentum_retrain = args.momentum_retrain or 0.0
            args.decay_retrain = args.decay_retrain or 0.0

            args.momentum_pretrain = args.momentum_pretrain or 0.5
            args.decay_pretrain = args.decay_pretrain or 5e-5

            if args.dataset == "mnist":
                args.lr_pretrain = args.lr_pretrain or 0.1
                args.lr_retrain = args.lr_retrain or 0.005
            else:
                args.lr_pretrain = args.lr_pretrain or 5e-4
                args.lr_retrain = args.lr_retrain or 5e-4

        watermark_options = [
            'max_watermark_epoch', 'watermark_threshold', 'lr_pretrain', 'lr_retrain', 'momentum_pretrain', 'momentum_retrain',
            'decay_pretrain', 'decay_retrain', 'background', 'max_watermark_epoch', 'watermark_threshold',
            'lr_pretrain', 'lr_retrain', 'momentum_pretrain', 'momentum_retrain',
            'decay_pretrain', 'decay_retrain', 'wm_size'
        ]


        # Check for watermark-specific options
        for option in watermark_options:
            if getattr(args, option) is not None:
                if not args.watermark:
                    raise ValueError(
                        f"Option --watermark is set")



    # Set default values for other options if they are not provided
    if args.dataset == "mnist":
        args.lr = args.lr or 0.1
    else:
        args.lr = args.lr or 0.01

    args.num_rounds = args.num_rounds or 250
    args.clients_per_round = args.clients_per_round or 10
    args.clients = args.clients or 100
    args.epoch = args.epoch or 1
    args.batch_size = args.batch_size or 50
    args.momentum = args.momentum or 0.0
    args.decay = args.decay or 0.0
    args.seed = args.seed or 42
    args.device = args.device or 'cpu'

    return args


# Example usage:
args = parse_args()
print(args)

if __name__ == '__main__':
    args = parse_args()
    logger = Logger(log_dir=args.output_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    match args.model:
        case "vgg16":
            model = VGG16
        case "mnist_l5":
            model = MNIST_L5
        case "resnet18":
            model = ResNet18
    match args.dataset:
        case "cifar10":
            dataset = Cifar10(args)
        case "mnist":
            dataset = MNIST(args)


    FederatedExperiment(args, model, dataset).run()
    logger.log_hparams(args)


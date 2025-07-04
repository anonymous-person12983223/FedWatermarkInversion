import os
import time
from src.attack.base_attack import InverseUnlearning
from src.attack.proxy_attack import InverseUnlearningWithProxy
from src.utils.logger import Logger
from src.utils.model_utils import evaluate_model, load_model
from src.utils.setup_utils import setup
from src.utils.losses import *
import argparse

DATASETS = ['cifar10', 'mnist']
DEVICES = ['cuda', 'cpu']
MODELS = ['vgg16', 'mnist_l5', 'resnet18']
BACKGROUND = ["random", "abstract", "black", "randomN"]

def verify_args(args):
    if args.device == "cuda":
        assert torch.cuda.is_available()

    if not args.generate_proxy_clean:
        assert (args.inverse_loss_weight and args.use_clean) or not (args.inverse_loss_weight or args.use_clean)
    else:
        assert not args.use_clean
    assert len(args.model_weights) == len(args.model_paths)
    assert sum(args.model_weights) == 1

def create_parser():
    parser = argparse.ArgumentParser(description="Main script for model inversion and unlearning.")
    parser.add_argument("--model_str", type=str, choices=MODELS, required=True, help="Model class to use.")
    parser.add_argument("--dataset_str", type=str, choices=DATASETS, required=True, help="Dataset to use.")
    parser.add_argument("--background", type=str, choices=BACKGROUND, required=True, help="Background type.")
    parser.add_argument("--pattern", action="store_true", help="Enable pattern mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed value.")
    parser.add_argument("--wm_size", type=int, default=100, help="Number of images in watermark dataset")
    parser.add_argument("--clients", type=int, default=20, help="Number of clients")
    parser.add_argument("--device", type=str, choices=DEVICES, required=True, help="Device to use (cuda or cpu).")
    parser.add_argument("--base_model_path", type=str, required=True, help="Base target path for output.")
    parser.add_argument("--model_paths", type=str, nargs="+", required=True, help="List of model file paths.")
    parser.add_argument("--marks_per_class", type=int, help="Number of marks per class.")
    parser.add_argument("--inversion_epochs", type=int, help="Number of inversion epochs.")
    parser.add_argument("--inversion_lr", type=float, help="Learning rate for inversion.")
    parser.add_argument("--model_weights", type=float, nargs="+", required=True,
                        help="how much to weigh models during inversion")
    parser.add_argument("--tv_weight", type=float, required=False,
                        help="how much to weigh tv loss during inversion", default=0.0)
    parser.add_argument("--l2_weight", type=float, required=False,
                        help="how much to weigh l2 loss during inversion", default=0.0)
    parser.add_argument("--unlearn_epochs", type=int, help="Number of unlearning epochs.")
    parser.add_argument("--unlearn_lr", type=float, help="Learning rate for unlearning.")
    parser.add_argument("--unlearn_batch_size", type=int, help="Batch size for unlearning.")
    parser.add_argument("--unlearn_momentum", type=float, help="Momentum for unlearning optimizer.")
    parser.add_argument("--use_clean", action="store_true", help="Use clean data during unlearning.")
    parser.add_argument("--inverse_loss_weight", type=float, help="Contribution of inversion set to unlearning", required=False)
    parser.add_argument("--split_by_salient_activations", action="store_true")
    parser.add_argument("--generate_proxy_clean", action="store_true")
    parser.add_argument("--split_fraction", default=0.5, type=float)
    parser.add_argument('--output_dir',
                        help='output directory for log files',
                        default="output/",
                        type=str)
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    logger = Logger(log_dir=os.path.join(args.output_dir, "logs"))
    verify_args(args)

    model_cls, data, watermark_dataset, attack_set = setup(args)
    print(len(attack_set))

    if args.use_clean:
        clean_unlearn_data = attack_set
    else:
        clean_unlearn_data = None

    models = [
        load_model(str(os.path.join(args.base_model_path, mp)), model_cls, device=args.device)
        for mp in args.model_paths
    ]
    target_model = models[-1]

    # Evaluate the target model and log metrics
    test_metrics_pre = evaluate_model(target_model, data.get_testloader(), args.device)
    wm_metrics_pre = evaluate_model(target_model, watermark_dataset.get_loader(), args.device)


    t0 = time.time()
    if not args.generate_proxy_clean:
        unlearner = InverseUnlearning(
            model_cls=model_cls,
            data=data,
            watermark_dataset=watermark_dataset,
            attack_set=clean_unlearn_data,
            models=models,
            args=args,
        )
        unlearner.run()

    else:
        unlearner_proxy = InverseUnlearningWithProxy(
            model_cls=model_cls,
            data=data,
            watermark_dataset=watermark_dataset,
            models=models,
            args=args,
        )
        unlearner_proxy.run()

    elapsed_time = time.time() - t0

    # Log time taken for unlearning
    logger.log({"unlearn_time": elapsed_time})

    # Evaluate the model after unlearning and log metrics
    test_metrics_post = evaluate_model(target_model, data.get_testloader(), args.device)
    wm_metrics_post = evaluate_model(target_model, watermark_dataset.get_loader(), args.device)
    print(test_metrics_post)
    print(wm_metrics_post)

    all_metrics = {
        **{f"test_{k}_pre": v for k, v in test_metrics_pre.items()},
        **{f"wm_{k}_pre": v for k, v in wm_metrics_pre.items()},
        **{f"test_{k}_post": v for k, v in test_metrics_post.items()},
        **{f"wm_{k}_post": v for k, v in wm_metrics_post.items()},
    }
    logger.log_hparams(args, all_metrics)






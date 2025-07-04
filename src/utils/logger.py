from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

class Logger:
    _instance = None  # singleton pattern

    def __new__(cls, log_dir=None):
        if cls._instance is None and log_dir is not None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._init_writer(log_dir)
        return cls._instance

    def _init_writer(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.latest_metrics = {}

    def log(self, data, step=None):
        for key, value in data.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
                self.latest_metrics[key] = value
            elif isinstance(value, torch.Tensor):
                if value.ndim in (2, 3):  # HxW or CxHxW
                    self.writer.add_image(key, self._prepare_image(value), step)
                elif value.ndim == 4:  # NxCxHxW
                    self.writer.add_images(key, value, step)
            elif isinstance(value, np.ndarray):
                self.writer.add_image(key, torch.from_numpy(value), step)
            else:
                print(f"[Logger] Unsupported type for key '{key}': {type(value)}")

    def log_hparams(self, args_namespace, summary_metrics=None):
        hparams = {k: str(v) for k, v in vars(args_namespace).items()}

        # Prepare metric dict by merging summary_metrics and latest_metrics
        summary_metrics = summary_metrics or {}
        all_metrics = {**self.latest_metrics, **summary_metrics}  # summary_metrics overrides if key duplicates

        # If no metrics at all, pass empty dict (TensorBoard requires a dict)
        if not all_metrics:
            all_metrics = {}

        self.writer.add_hparams(hparam_dict=hparams, metric_dict=all_metrics)

    def _prepare_image(self, img_tensor):
        if img_tensor.ndim == 2:  # H x W
            img_tensor = img_tensor.unsqueeze(0)  # -> 1 x H x W
        elif img_tensor.ndim == 3 and img_tensor.shape[0] not in [1, 3]:
            # probably HWC, convert to CHW
            img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor

    def close(self):
        self.writer.close()

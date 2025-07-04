import torch
from torch.utils.data import TensorDataset, DataLoader

from src.attack.inversion import Inversion
from src.attack.unlearn import ModelUnlearner
from src.models.cifar10_models import VGG16
from src.models.mnist_models import MNIST_L5
from src.utils.logger import Logger


class InverseUnlearning:
    def __init__(self, model_cls, data, watermark_dataset, attack_set, models, args):
        self.model_cls = model_cls
        self.data = data
        self.watermark_dataset = watermark_dataset
        self.attack_set = attack_set
        self.models = models
        self.args = args
        self.logger = Logger()
        self.inversion = Inversion(models, list(data.image_shape), args, None)
        self.num_classes = data.num_classes
        self.target_model = models[-1]
        self.activations = {}

    def _hook_function(self, module, input, output):
        self.activations["values"] = output.view(output.size(0), -1).cpu()

    def _split_by_salient_activations(self, inputs, model):
        split_num = int(0.5 * len(inputs))
        device = next(model.parameters()).device
        inputs = inputs.to(device)

        # Choose layer
        if isinstance(model, VGG16):
            layer_to_hook = model.model.features[28]
        elif isinstance(model, MNIST_L5):
            layer_to_hook = model.block[-1]
        else:
            raise ValueError("Model type not supported for saliency split.")

        # Register hook
        hook = layer_to_hook.register_forward_hook(self._hook_function)
        _ = model(inputs)
        hook.remove()

        if "values" not in self.activations:
            raise RuntimeError("Hook did not capture activations.")

        feats = self.activations["values"]
        feat_mean = torch.mean(feats, dim=0)
        feat_std = torch.std(feats, dim=0)
        neuron_impts = feat_mean / (feat_std + 1e-6)

        salient_threshold = torch.quantile(neuron_impts, 0.95)
        salient_neurons = torch.where(neuron_impts > salient_threshold)[0]
        sample_contribution = torch.mean(feats[:, salient_neurons], dim=1)
        sample_sorted_indices = sample_contribution.sort(descending=False)[1]
        proxy_wmk_indices = sample_sorted_indices[:split_num].detach().cpu().numpy()
        proxy_nor_indices = sample_sorted_indices[split_num:].detach().cpu().numpy()
        return proxy_wmk_indices, proxy_nor_indices

    def run(self):
        full_output = []
        true_labels = []
        kl_targets = []

        for i in range(self.num_classes):
            output = self.inversion.generate_inverted_samples(i)

            if self.args.split_by_salient_activations:
                proxy_wmk_indices, _ = self._split_by_salient_activations(output, self.models[-1])
                output = output[proxy_wmk_indices]

            self.logger.log({f"image_label_{i}": output[0]})

            labels_tensor = torch.full((output.size(0),), i, dtype=torch.long)
            full_output.append(output)
            true_labels.append(labels_tensor)

            kl_targets.append(torch.stack([
                torch.tensor([1.0 / self.num_classes] * self.num_classes)
                for _ in range(output.size(0))
            ]))

        full_output = torch.cat(full_output, dim=0)
        full_kl_targets = torch.cat(kl_targets, dim=0)
        true_labels = torch.cat(true_labels, dim=0)

        dataset = TensorDataset(full_output, full_kl_targets)

        if self.attack_set:
            assert self.args.use_clean
            data_loader = DataLoader(dataset, batch_size=self.args.unlearn_batch_size // 2, shuffle=True)
            attack_loader = DataLoader(self.attack_set, batch_size=self.args.unlearn_batch_size // 2, shuffle=True)
        else:
            data_loader = DataLoader(dataset, batch_size=self.args.unlearn_batch_size, shuffle=True)
            attack_loader = None

        unlearner = ModelUnlearner(
            model=self.target_model,
            inverse_loader=data_loader,
            clean_loader=attack_loader,
            args=self.args
        )
        unlearner.run()
        return full_output, true_labels

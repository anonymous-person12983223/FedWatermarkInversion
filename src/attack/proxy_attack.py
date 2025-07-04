import copy
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.attack.inversion import Inversion
from src.attack.unlearn import ModelUnlearner


class InverseUnlearningWithProxy:
    def __init__(self, model_cls, data, watermark_dataset, models, args):
        self.model_cls = model_cls
        self.data = data
        self.watermark_dataset = watermark_dataset
        self.models = models
        self.args = args
        self.target_model = models[-1]
        self.device = args.device
        self.num_classes = data.num_classes

        self.inversion1 = Inversion(models, data.image_shape, args, None)

        args2 = copy.deepcopy(args)
        args2.model_weights = [0.0 for _ in args.model_weights]
        args2.model_weights[-1] = 1.0
        args2.marks_per_class = 250
        self.args2 = args2
        self.inversion2 = Inversion(models, data.image_shape, args2, None)

    def _collect_samples_for_class(self, inversion, class_idx, model_idx, keep, marks_per_class):
        all_inputs, all_labels, new_labels = [], [], []
        total_collected = 0
        attempts = 0

        while total_collected < marks_per_class and attempts < 5:
            attempts += 1
            print(f"inv {class_idx} - attempt {attempts} - collected {total_collected}/{marks_per_class}")

            output = inversion.generate_inverted_samples(class_idx).to(self.device)
            labels_tensor = torch.full((output.size(0),), class_idx, dtype=torch.long).to(self.device)

            logits = self.models[model_idx](output)
            preds = logits.argmax(dim=1)

            mask = preds == labels_tensor if keep == 'correct' else preds != labels_tensor

            selected_inputs = output[mask]
            selected_labels = labels_tensor[mask]

            num_selected = selected_inputs.size(0)
            if num_selected > 0:
                assigned_new_labels = torch.stack([
                    torch.tensor([1.0 / self.num_classes] * self.num_classes)
                ] * num_selected).to(self.device)

                remaining_needed = marks_per_class - total_collected
                if num_selected > remaining_needed:
                    selected_inputs = selected_inputs[:remaining_needed]
                    selected_labels = selected_labels[:remaining_needed]
                    assigned_new_labels = assigned_new_labels[:remaining_needed]
                    num_selected = remaining_needed

                all_inputs.append(selected_inputs)
                all_labels.append(selected_labels)
                new_labels.append(assigned_new_labels)

                total_collected += num_selected

        if total_collected == 0:
            return None, None, None

        return torch.cat(all_inputs), torch.cat(all_labels), torch.cat(new_labels)

    def run(self):
        d1_inputs, d1_labels, d1_kl_targets = [], [], []
        for i in range(self.num_classes):
            output =self.inversion1.generate_inverted_samples(i)

            labels = torch.full((output.size(0),), i, dtype=torch.long)

            kl_targets =(torch.stack([
                torch.tensor([1.0 / self.num_classes] * self.num_classes)
                for _ in range(output.size(0))]))

            d1_inputs.append(output)
            d1_labels.append(labels)
            d1_kl_targets.append(kl_targets)

        d2_inputs, d2_labels = [], []
        for i in range(self.num_classes):
            inputs, labels, _ = self._collect_samples_for_class(
                self.inversion2, i, 0, 'incorrect', self.args2.marks_per_class
            )
            if inputs is not None:
                d2_inputs.append(inputs)
                d2_labels.append(labels)

        if not d1_inputs or not d2_inputs:
            print("Warning: Not enough valid samples collected!")
            return None, None

        d1_inputs = torch.cat(d1_inputs)
        d1_labels = torch.cat(d1_labels)
        d1_kl_targets = torch.cat(d1_kl_targets)

        d2_inputs = torch.cat(d2_inputs)
        d2_labels = torch.cat(d2_labels)

        loader_d1 = DataLoader(TensorDataset(d1_inputs, d1_kl_targets),
                               batch_size=self.args.unlearn_batch_size // 2, shuffle=True)
        loader_d2 = DataLoader(TensorDataset(d2_inputs, d2_labels),
                               batch_size=self.args.unlearn_batch_size // 2, shuffle=True)

        unlearner = ModelUnlearner(
            model=self.target_model,
            inverse_loader=loader_d1,
            clean_loader=loader_d2,
            args=self.args
        )
        unlearner.run()

        return d1_inputs, d1_labels
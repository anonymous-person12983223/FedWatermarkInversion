from typing import List, Tuple, Optional

import torch
from torch import nn

from src.utils.image_utils import normalize
from src.utils.losses import TVLoss, L2Loss, compute_weighted_ce_loss


class Inversion:
    def __init__(self, models: List[torch.nn.Module],
                 image_shape: Tuple[int, int, int],
                 attack_args,
                 init_mark: Optional[torch.Tensor] = None):
        """
       Initializes the FullInversion class for performing model inversion tasks.

       Parameters:
       models (List[torch.nn.Module]): A list of PyTorch model instances that the inversion process targets.
       defense_remask_epoch (int, optional): The number of epochs for the optimization process. Defaults to 10.
       defense_remask_lr (float, optional): The learning rate used in the optimization process. Defaults to 0.1.
       init_mark (Optional[torch.Tensor], optional): An initial mark image tensor to start optimization. If None,
                                                     optimization starts with random noise. Defaults to None.
        """

        self.inversion_epochs = attack_args.inversion_epochs
        self.inversion_lr = attack_args.inversion_lr
        self.init_mark = init_mark
        self.mark = None
        self.models = models
        self.image_shape = image_shape
        self.model_weights = attack_args.model_weights

        self.tv_weight = attack_args.tv_weight
        self.l2_weight = attack_args.l2_weight
        self.marks_per_class = attack_args.marks_per_class
        self.device = attack_args.device

    def generate_inverted_samples(self, label):
        for model in self.models:
            model.eval()
            model.to(self.device)
        xs_shape = (self.marks_per_class, self.image_shape[2], self.image_shape[0], self.image_shape[1])

        assert self.image_shape[2] == 1 or self.image_shape[2] == 3
        tv_criterion = TVLoss()
        l2_criterion = L2Loss()
        criterion = nn.CrossEntropyLoss()
        class_xs = []
        i_class = label
        xs_v = torch.randn(size=xs_shape).to(self.device)
        xs_v.requires_grad = True
        targets = torch.tensor([i_class] * self.marks_per_class).to(self.device)
        optimizer = torch.optim.Adam([xs_v], lr=self.inversion_lr, betas=(0.5, 0.9), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.inversion_epochs, eta_min=1e-4)

        for i in range(self.inversion_epochs):
            optimizer.zero_grad()
            imgs = (torch.tanh(xs_v) + 1.0) / 2.0
            imgs = normalize(imgs, xs_shape[1])

            cls_loss_total = compute_weighted_ce_loss(self.models, self.model_weights, imgs, targets, criterion)
            l2_reg = l2_criterion(imgs) * self.l2_weight
            tv_reg = tv_criterion(imgs) * self.tv_weight
            loss = cls_loss_total + l2_reg + tv_reg

            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            imgs = (torch.tanh(xs_v) + 1.0) / 2.0
            imgs = normalize(imgs, xs_shape[1])
        class_xs.append(imgs.detach().cpu())

        del_vars = ["xs_v", "targets", "l2_reg", "tv_reg", "loss", "cls_loss_total"]

        for var in del_vars:
            if var in locals():
                del locals()[var]

        torch.cuda.empty_cache()

        class_xs_tensor = torch.concat(class_xs)
        #print(class_xs_tensor.shape)
        return class_xs_tensor
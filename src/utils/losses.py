from torch import nn
import torch

class TVLoss(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return 2 * (h_tv / x[:, :, 1:, :].numel() + w_tv / x[:, :, :, 1:].numel()) / batch_size

class L2Loss(nn.Module):
    def forward(self, x):
        return torch.norm(x) / x.shape[0]

def compute_weighted_ce_loss(models, weights, imgs, targets, criterion):
    loss = 0.0
    for model, w in zip(models, weights):
        if w > 0:
            output = model(imgs)
            loss += criterion(output, targets) * w
    return loss
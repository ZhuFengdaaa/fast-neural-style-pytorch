import torch
import torch.nn as nn


class TotalVariationLoss(nn.Module):
    def __init__(self, weight):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight

    def forward(self, input):
        self.output = input.clone()
        self.loss = self.weight * torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:]) + \
                                            torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

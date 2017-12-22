import torch
import torch.nn as nn


class ContentLoss(nn.Module):
    def __init__(self, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, input):
        self.output = input.clone()
        if self.mode == 'capture':
            self.target = input.detach()
        elif self.mode == 'loss':
            self.loss = self.weight * self.criterion(input, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

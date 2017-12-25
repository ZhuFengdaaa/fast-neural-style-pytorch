import torch
import torch.nn as nn


class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a, b, c * d)  # resise F_XL into \hat F_XL

        G = torch.bmm(features, torch.transpose(features, 1, 2))  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, weight):
        super(StyleLoss, self).__init__()
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()
        self.mode = 'none'
        self.target = torch.Tensor()

    def set_mode(self, mode):
        self.mode = mode
    
    def forward(self, input):
        self.output = input.clone()
        if self.mode == 'capture':
            self.target = self.gram(input.detach())
        elif self.mode == 'loss':
            self.G = self.gram(input)
            self.loss = self.weight * self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

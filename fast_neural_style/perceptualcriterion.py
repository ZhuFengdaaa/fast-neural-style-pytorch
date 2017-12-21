import torch
import torch.nn as nn
from styleloss import StyleLoss
from contentloss import ContentLoss


class Perceptualcriterion(nn.Module):
    def __init__(self, cnn, opt):
        self.content_losses = []
        self.style_losses = []
        self.discriminator = nn.Sequential()
        if opt.gpu_ids > 0:
            self.discriminator = self.discriminator.cuda()

        content_layers = opt.content_layers
        style_layers = opt.style_layers

        self.target = None

        i = 1
        for layer in list(cnn):
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                self.discriminator.add_module(name, layer)
                if name in content_layers:
                    # add content loss:
                    # target = self.discriminator(self.content_img)
                    content_loss = ContentLoss(opt.content_weight)
                    self.discriminator.add_module("content_loss_" + str(i), content_loss)
                    self.content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    # target_feature = self.discriminator(self.style_img)
                    # target_feature_gram = self.gram(target_feature)
                    style_loss = StyleLoss(opt.style_weight)
                    self.discriminator.add_module("style_loss_" + str(i), style_loss)
                    self.style_losses.append(style_loss)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                self.discriminator.add_module(name, layer)

                if name in content_layers:
                    # add content loss:
                    # target = self.discriminator(self.content_img)
                    content_loss = ContentLoss(opt.content_weight)
                    self.discriminator.add_module("content_loss_" + str(i), content_loss)
                    self.content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    # target_feature = self.discriminator(self.style_img)
                    # target_feature_gram = self.gram(target_feature)
                    style_loss = StyleLoss(opt.style_weight)
                    self.discriminator.add_module("style_loss_" + str(i), style_loss)
                    self.style_losses.append(style_loss)
                i += 1

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                self.discriminator.add_module(name, layer)

    def set_content_target(self, input):
        for content_loss in self.content_losses:
            content_loss.set_mode("capture")
        for style_loss in self.style_losses:
            style_loss.set_mode("none")
        self.discriminator(input)

    def set_style_target(self, input):
        for content_loss in self.content_losses:
            content_loss.set_mode("none")
        for style_loss in self.style_losses:
            style_loss.set_mode("capture")
        self.discriminator(input)

    def forward(self, input):
        for content_loss in self.content_losses:
            content_loss.set_mode("loss")
        for style_loss in self.style_losses:
            style_loss.set_mode("loss")
        return self.discriminator(input)

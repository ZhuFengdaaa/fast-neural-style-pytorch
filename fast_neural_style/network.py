from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import functools

import init_weights


class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


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
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class TotalVaraitionLoss(nn.Module):
    def __init__(self, weight):
        self.weight = weight

    def forward(self, input):
        self.output = input.clone()
        self.loss = self.weight * torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, :1]) + \
                                            torch.abs(input[:, :, :-1, :] - input[:, :, 1, :]))
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias, use_dropout=False, padding_type='reflect'):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Generator(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(Generator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        prev_dim = 3
        next_dim = 32
        sequence = []
        sequence += [nn.Conv2d(prev_dim, next_dim, kernel_size=9, stride=1, padding=4),
                     norm_layer(next_dim),
                     nn.ReLU(True)
                     ]
        for i in range(2):
            prev_dim = next_dim
            next_dim *= 2
            sequence += [nn.Conv2d(prev_dim, next_dim, kernel_size=3, stride=2, padding=1),
                         norm_layer(next_dim),
                         nn.ReLU(True)
                         ]
        prev_dim = next_dim
        assert (prev_dim == next_dim == 128)
        for i in range(5):
            sequence += [ResnetBlock(next_dim, norm_layer, use_bias)]

        next_dim /= 2
        sequence += [nn.ConvTranspose2d(prev_dim, next_dim,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                     norm_layer(next_dim),
                     nn.ReLU(True)]
        prev_dim = next_dim
        next_dim /= 2
        sequence += [nn.ConvTranspose2d(prev_dim, next_dim,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                     norm_layer(next_dim),
                     nn.ReLU(True)]
        prev_dim = next_dim
        next_dim = 3
        sequence += [nn.Conv2d(prev_dim, next_dim, kernel_size=9, stride=1, padding=4),
                     norm_layer(next_dim),
                     nn.ReLU(True)
                     ]
        sequence += [nn.Tanh()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class SimpleModel():
    def __init__(self, opt):
        # desired depth layers to compute style/content losses :
        print("build start")
        self.opt = opt
        self.Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor
        nb = opt.batch_size
        size = opt.image_size
        content_layers = opt.content_layers
        style_layers = opt.style_layers
        self.image_tensor = self.Tensor(nb, 3, size, size)
        self.content_img = Variable(self.Tensor(nb, 3, size, size))
        self.style_img = Variable(self.Tensor(nb, 3, size, size))
        self.generated_img = Variable(self.Tensor(nb, 3, size, size))
        norm_layer = get_norm_layer(norm_type=opt.norm)
        self.model = nn.Sequential()
        generator = Generator(norm_layer, opt.gpu_ids)
<<<<<<< HEAD
=======
        if len(opt.gpu_ids) > 0:
            generator.cuda(device_id=opt.gpu_ids[0])
>>>>>>> e187cc3336d85dd7b236787f00911933a94f3c75
        init_weights.init_weights(generator)

        self.model.add_module('generator', generator)
        assert (opt.percep_loss_weight > 0)
        cnn = models.vgg16(pretrained=True).features

        self.gram = GramMatrix()
        self.content_losses = []
        self.style_losses = []
        i = 1
        if opt.gpu_ids > 0:
            self.model=self.model.cuda()
            self.gram=self.gram.cuda()
            cnn=cnn.cuda()
        for layer in list(cnn):
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                self.model.add_module(name, layer)

                if name in content_layers:
                    # add content loss:
                    target = self.model(self.content_img)
                    content_loss = ContentLoss(target, opt.content_weight)
                    self.model.add_module("content_loss_" + str(i), content_loss)
                    self.content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    target_feature = self.model(self.style_img)
                    target_feature_gram = self.gram(target_feature)
                    style_loss = StyleLoss(target_feature_gram, opt.style_weight)
                    self.model.add_module("style_loss_" + str(i), style_loss)
                    self.style_losses.append(style_loss)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                self.model.add_module(name, layer)

                if name in content_layers:
                    # add content loss:
                    target = self.model(self.content_img)
                    content_loss = ContentLoss(target, opt.content_weight)
                    self.model.add_module("content_loss_" + str(i), content_loss)
                    self.content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    target_feature = self.model(self.style_img)
                    target_feature_gram = self.gram(target_feature)
                    style_loss = StyleLoss(target_feature_gram, opt.style_weight)
                    self.model.add_module("style_loss_" + str(i), style_loss)
                    self.style_losses.append(style_loss)
                i += 1

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                self.model.add_module(name, layer)

        self.generator_optimizer = torch.optim.Adam([param for name, param in self.model.named_parameters() if name.startswith('generator')], lr=opt.lr)
        print("build end")

    def forward(self, data):
        self.image_tensor.resize_(data.size()).copy_(data)
        self.content_img = Variable(self.image_tensor)
        self.model.forward(self.content_img)

    def backward(self):
        self.generator_optimizer.zero_grad()
        content_score = 0
        style_score = 0
        for cl in self.content_losses:
            content_score += cl.backward()
        for sl in self.style_losses:
            style_score += sl.backward()
        return content_score, style_score

    def name(self):
        return 'SimpleModel'

    def save(self, network, network_label, iter_label):
        save_filename = '%s_%s.pth' % (network_label, iter_label)
        save_path = os.path.join(self.opt.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(self.opt.gpu_ids) and torch.cuda.is_available():
            network.cuda()

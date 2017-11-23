import torch
import torch.nn as nn
import functools


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


class FeatureNet(nn.module):
    def __init__(self, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(FeatureNet, self).__init__()
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
        assert (prev_dim == nnext_dim == 128)
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
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        nb = opt.batch_size
        size = opt.style_image_size
        self.input = self.Tensor(nb, 3, size, size)
        norm_layer = get_norm_layer(norm_type=opt.norm)
        feature_net = FeatureNet(norm_layer, gpu_ids)


        # Set up pixel loss function
        pixel_crit=None
        if opt.pixel_loss_weight > 0:
            if opt.pixel_loss_type=='L2':
                pixel_crit=nn.MSELoss()
            elif opt.pixel_loss_type=='L1':
                pixel_crit = nn.L1Loss()
            elif opt.pixel_loss_type=='SmoothL1':
                pixel_crit = nn.SmoothL1Loss()

        # Set up the perceptual loss function
        percep_crit=None
        if opt.percep_loss_weight > 0:
        loss_net = torch.load()
    def name(self):
        return 'SimpleModel'

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])


def create_model(opt):
    model = None
    model = SimpleModel()
    return model

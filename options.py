import argparse
import os
import torch


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--mode', required=True, choices=['train', 'test'])
        self.parser.add_argument('--data_roots', required=True, type=str, help='path to images, use comma to separate multiple path')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self.parser.add_argument('--image_size', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--percep_loss_weight', type=float, default='1.0', help='percep_loss_weight')
        self.parser.add_argument('--content_layers', type=str, default='relu_4', help='content_layers')
        self.parser.add_argument('--content_weight', type=float, default='1.0', help='content weight')
        self.parser.add_argument('--style_layers', type=str, default='relu_2,relu_4,relu_7,relu_10',
                                 help='style_layers')
        self.parser.add_argument('--style_weight', type=float, default='1000.0', help='style weight')
        self.parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument('--reg_tv', type=float, default=0.00001, help='total variation regularizer weight between 1e-6 to 1e-4')
        self.parser.add_argument('--epoch_iter', type=int, default=40000, help='# of iter per epoch')
        self.parser.add_argument('--num_epochs', type=int, default=2, help='# of epochs')
        self.parser.add_argument('--log_iter', type=int, default=10000, help='frequency of logging')
        self.parser.add_argument('--save_iter', type=int, default=1, help='frequency of saving checkpoints')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        def parse_comma(string, type_=str):
            res = []
            str_list = string.split(',')
            for items in str_list:
                res.append(type_(items))
            return res

        self.opt.data_roots = parse_comma(self.opt.data_roots)
        self.opt.gpu_ids = parse_comma(self.opt.gpu_ids, int)
        self.opt.content_layers = parse_comma(self.opt.content_layers)
        self.opt.style_layers = parse_comma(self.opt.style_layers)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt

        # # save to the disk
        # expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        # util.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write('------------ Options -------------\n')
        #     for k, v in sorted(args.items()):
        #         opt_file.write('%s: %s\n' % (str(k), str(v)))
        #     opt_file.write('-------------- End ----------------\n')
        # return self.opt

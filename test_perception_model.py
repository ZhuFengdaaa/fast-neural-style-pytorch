import os
import argparse
import torch


parser = argparse.ArgumentParser(description='Test Perception Model arguments: ')

parser.add_argument('--model_path', type=str, required=True, help='model path')
parser.add_argument('--image_path', type=str, help='image path')
parser.add_argument('--imageset_path', type=str, help='imageset path')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id, use -1 for CPU')

opt = parser.parse_args()

if os.path.isfile(opt.model_path):
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage.cuda(opt.gpu_id))
    print(checkpoint)
else:
    print("=> no checkpoint found at '{}'".format(opt.model_path))
import os
import argparse
import torch
from options import Options
opt = Options().parse()
from fast_neural_style import network

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

if not os.path.isfile(opt.model_path):
    print("=> no checkpoint found at '{}'".format(opt.model_path))
    exit(-1)

model = network.PerceptualModel(opt)
print("=> loading checkpoint '{}'".format(opt.model_path))
model.load_state_dict(torch.load(opt.model_path, map_location=lambda storage, loc: storage.cuda(opt.gpu_ids[0])))
print(type(model))

import torch
import torch.nn as nn
import os


def save(network, save_dir, network_label, iter_label):
    save_filename = '%s_%s.pth' % (network_label, iter_label)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    network.cuda()

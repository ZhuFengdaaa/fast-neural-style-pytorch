import torch
import torch.nn as nn
import os


def save(self, network, network_label, iter_label):
    save_filename = '%s_%s.pth' % (network_label, iter_label)
    save_path = os.path.join(self.opt.save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if len(self.opt.gpu_ids) and torch.cuda.is_available():
        network.cuda()

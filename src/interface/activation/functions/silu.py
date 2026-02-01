from ..base import Activation

import torch


class SiLU(Activation):

    def __init__(self, activation, pre_activation, index):
        super().__init__(activation, pre_activation, index)

    def phi(self, x):
        return x * torch.sigmoid(x)
    
    def phi_grad(self, x, phi):
        sigmoid = torch.sigmoid(x)
        return sigmoid * (1 + x * (1 - sigmoid))

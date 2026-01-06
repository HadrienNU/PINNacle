from interface.activation import Activation

import torch


class Tanh(Activation):

    def __init__(self, activation, pre_activation, index):
        super().__init__(activation, pre_activation, index, 3)

    def phi(self, x):
        return torch.tanh(x)
    
    def phi_grad(self, x, phi):
        return 1 - phi ** 2

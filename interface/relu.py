from interface.activation import Activation

import torch


class ReLU(Activation):

    def __init__(self, activation, pre_activation, index):
        super().__init__(activation, pre_activation, index)

    def phi(self, x):
        return torch.clamp(x, min=0)
    
    def phi_grad(self, x, phi):
        return (x > 0).int()
    
    

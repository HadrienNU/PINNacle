from interface.activation import Activation

import torch


class ReLU(Activation):

    ID_REGION = {}
    NB_REGION = 0

    def __init__(self, activation, pre_activation, index):
        super().__init__(activation, pre_activation, index)
        self.ID_REGION = {}
        self.NB_REGION = 0

    def phi(self, x):
        return torch.clamp(x, min=0)
    
    def phi_grad(self, x, phi):
        return (x > 0).int()
    
    def compute_region(self, input_storage):
        grad = self.phi_grad(self.pre_activation, None)
        map_region = {}
        num_inputs = len(input_storage)
        for i in range(num_inputs):
            input_point = input_storage[i].tolist()
            region = tuple(grad[i].tolist())
            if region not in self.ID_REGION:
                self.NB_REGION += 1
                self.ID_REGION[region] = self.NB_REGION
            id_region = self.ID_REGION[region]
            if id_region in map_region:
                map_region[id_region].append(input_point)
            else:
                map_region[id_region] = [input_point]
        return map_region

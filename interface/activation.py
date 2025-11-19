import matplotlib.pyplot as plt
import torch


class Activation:

    def __init__(self, activation, pre_activation, index):
        self.distances = []
        self.activation = activation
        self.pre_activation = pre_activation
        self.index = index

    def phi(self, x):
        return torch.zeros_like(x)
    
    def phi_grad(self, x, phi):
        return torch.zeros_like(x)
    
    def distance_custom(self, x, y):
        return self.distance_taylor(x, y)

    def distance_taylor(self, x, y):
        i, j = self.index[tuple(x.tolist())], self.index[tuple(y.tolist())]
        a, b = self.pre_activation[i], self.pre_activation[j]
        phi_a, phi_b = self.phi(a), self.phi(b)
        phi_da, phi_db = self.phi_grad(a, phi_a), self.phi_grad(b, phi_b)
        distance = 2 * (phi_a - phi_b) + (b - a) * (phi_da + phi_db) 
        dist = torch.sum(distance ** 2).item()
        self.distances.add(dist)
        return dist
    
    def export_distances(self, name):
        plt.figure()
        self.distances = sorted(self.distances)
        plt.plot(range(len(self.distances)), self.distances)
        plt.savefig(f"{name}.png")
        
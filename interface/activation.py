import matplotlib.pyplot as plt
import torch


class Activation:

    def __init__(self, activation, pre_activation, index):
        self.distances = []
        self.activation = activation
        self.pre_activation = pre_activation
        self.index = index
        self.distances_mem = {}

    def compute_region(self, input_storage):
        return None
    
    def phi(self, x):
        return torch.zeros_like(x)
    
    def phi_grad(self, x, phi):
        return torch.zeros_like(x)
    
    def distance_custom(self, x, y):
        return self.distance_taylor(x, y)

    def distance_taylor(self, x, y):
        i, j = self.index[tuple(x.tolist())], self.index[tuple(y.tolist())]
        a, b = self.pre_activation[i], self.pre_activation[j]
        if (i, j) in self.distances_mem:
            return self.distances_mem[(i, j)]
        phi_a, phi_b = self.phi(a), self.phi(b)
        phi_da, phi_db = self.phi_grad(a, phi_a), self.phi_grad(b, phi_b)
        distance = 2 * (phi_a - phi_b) + (b - a) * (phi_da + phi_db) 
        dist = torch.sum(distance ** 2).item()
        self.distances.append(dist)

        
        self.distances_mem[(i, j)] = dist
        return dist
    
    def export_distances(self, name):
        distances = sorted(self.distances)
        N = len(distances)
        probabilities = torch.linspace(0, 1, steps=N).tolist()  # 0→1

        plt.figure()
        plt.plot(distances, probabilities)
        plt.xlabel("Distance")
        plt.ylabel("CDF")
        plt.title("Cumulative Distribution of Distances")
        plt.grid(True)
        plt.savefig(f"{name}_cdf.png")
        plt.close()

    def get_distance(self, probability):
        p = max(0, min(1, probability))
        distances = sorted(self.distances)
        N = len(distances)

        if N == 0:
            return None  
        if N == 1:
            return distances[0]

        idx = p * (N - 1)
        i0 = int(idx)
        i1 = min(i0 + 1, N - 1)

        if i0 == i1:
            return distances[i0]

        t = idx - i0
        d = distances[i0] * (1 - t) + distances[i1] * t
        return d

        
import matplotlib.pyplot as plt
import torch


class Activation:

    ID_REGION = {}
    NB_REGION = 0
    N_SAMPLES = 5_000

    def __init__(self, activation, pre_activation, index, region_number=2):
        self.distances = []
        self.activation = activation
        self.pre_activation = pre_activation
        self.index = index
        self.distances_mem = {}
        self.ID_REGION = {}
        self.NB_REGION = 0
        self.region_number = region_number
    
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
        probabilities = torch.linspace(0, 1, steps=N).tolist()
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
    
    def export_splitted_region(self, thresholds, pre_activation_layer, layer_depth):
        n_sample = self.N_SAMPLES
        pre_activation = pre_activation_layer.flatten()
        random_indices = torch.randint(
            0, pre_activation.shape[0], 
            (n_sample,), device=pre_activation.device
        )
        pre_activation = pre_activation[random_indices]
        post_activation = self.activation(pre_activation).detach()
        linspace_x = torch.linspace(pre_activation.min(), pre_activation.max(), n_sample)
        linspace_y = torch.tensor([post_activation.min(), post_activation.max()])
        plt.figure()        
        plt.scatter(
            pre_activation.cpu().detach().numpy(),
            post_activation.cpu().detach().numpy(),
            color='r', marker='x', label=f"Data Distribution ({n_sample} samples)"
        )
        plt.plot(
            linspace_x.cpu().detach().numpy(), 
            self.activation(linspace_x).cpu().detach().numpy(),
            label="Activation"
        )
        for i, th in enumerate(thresholds):
            x = torch.ones_like(linspace_y) * th
            plt.plot(
                x.cpu().detach().numpy(), 
                linspace_y.cpu().detach().numpy(), 
                label=f"region_{i}"
            )
        plt.title("Splitted Regions")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"splitted_region_layer{layer_depth}_{self.activation.__name__}.png")
        plt.close()
    
    def split_region(self, x):        
        n = self.region_number
        assert n > 1

        results = []
        layer_depth = 0
        for layer in x:
            pre_activation_layer = layer
            layer = self.phi(layer)
            result = torch.zeros_like(layer, dtype=torch.long)
            x_flat, _ = layer.flatten().sort()
            max_samples = self.N_SAMPLES
            if len(x_flat) > max_samples:
                random_indices = torch.randint(
                    0, x_flat.shape[0], 
                    (max_samples,), device=layer.device
                )
                x_flat = x_flat[random_indices]

            quantiles = torch.tensor([(i / n) for i in range(1, n)], device=layer.device)
            quantiles = (quantiles * x_flat.shape[0]).int()
            thresholds = x_flat[quantiles]
            
            for th in thresholds:
                result += (layer >= th).long()
            self.export_splitted_region(thresholds, pre_activation_layer, layer_depth)
            results.append(result)
            layer_depth += 1
        
        regions = torch.cat(results, dim=1)
        return regions

    def compute_region(self, input_storage, pre_activation):
        grad = self.split_region(pre_activation)
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
        
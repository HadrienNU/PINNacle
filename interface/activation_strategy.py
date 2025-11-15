from interface.regions import Regions
from interface.relu import ReLU

from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import coo_matrix

from deepxde import backend as bkd

import numpy as np
import torch


class ActivationRegionStrategy:

    def __init__(self, model, register_ready):
        self.model = model
        self.register_ready = register_ready
        self.output_storage = []
        self.input_storage = []
        self.map_regions_id = {}                
        self.nb_regions = 0
        self.resolution = 500
        self.dim = len(model.pde.bbox) // 2
        self.init_strategy()        
        #self.model.net.activation = bkd.silu

    def init_strategy(self):
        self.activations_strategy = {
            "tanh": ReLU,
            "relu": ReLU,
            "silu": ReLU
        }

    def get_strategy(self, output, index):
        strategy = self.activations_strategy[self.activation_name]
        return strategy(self.model.net.activation, output, index)
    
    def get_output_hook(self):
        def get_hook(module, input, output):
            if self.register_ready():
                self.output_storage.append(output.cpu())
        return get_hook
    
    def get_input_hook(self):
        def get_hook(module, input):
            if self.register_ready():
                self.input_storage.append(input[0].cpu())
        return get_hook

    def register_hook(self):
        self.model.net.register_forward_pre_hook(self.get_input_hook())
        self.activation_name = self.model.net.activation.__name__
        print(f"Activation : {self.activation_name}")
        get_hook = self.get_output_hook()
        for module in self.model.net.modules():   
            if isinstance(module, torch.nn.modules.linear.Linear):
                module.register_forward_hook(get_hook)

    def compute_region(self, name):
        outputs = torch.cat(self.output_storage, dim=1)        
        inputs = self.input_storage[0].detach().cpu().numpy()
        num_inputs = inputs.shape[0]
        index = {tuple(inputs[i].tolist()) : i for i in range(num_inputs)}
        strategy = self.get_strategy(outputs, index)

        connectivity = kneighbors_graph(inputs, n_neighbors=8, include_self=False)
        connectivity = 0.5 * (connectivity + connectivity.T)
        clusterer = AgglomerativeClustering(
            metric=strategy.distance_taylor,
            n_clusters=None,
            distance_threshold=1e-13,
            linkage='single',
            connectivity=connectivity
        )        
        
        labels = clusterer.fit_predict(inputs)        
        map_region = {}
        for i in range(num_inputs):
            input_point = self.input_storage[0][i].tolist()
            id_region = labels[i].item()
            if id_region in map_region:
                map_region[id_region].append(input_point)
            else:
                map_region[id_region] = [input_point]

        strategy.export_distances(name)
        return map_region

    def evaluate_regions(self):
        self.output_storage.clear()
        self.input_storage.clear()
              
        if self.dim == 2:
            x_range = torch.linspace(self.model.pde.bbox[0], self.model.pde.bbox[1], self.resolution)
            y_range = torch.linspace(self.model.pde.bbox[2], self.model.pde.bbox[3], self.resolution)
            xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
            grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        elif self.dim == 3:
            self.resolution = 50  # Reduce resolution for 3D to limit memory usage
            x_range = torch.linspace(self.model.pde.bbox[0], self.model.pde.bbox[1], self.resolution)
            y_range = torch.linspace(self.model.pde.bbox[2], self.model.pde.bbox[3], self.resolution)
            z_range = torch.linspace(self.model.pde.bbox[4], self.model.pde.bbox[5], self.resolution)
            xx, yy, zz = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
            grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
        if self.dim > self.model.pde.geom.dim: # Time 
            inside_mask = self.model.pde.geom.inside(grid_points[:, :-1].cpu().numpy())
        else:
            inside_mask = self.model.pde.geom.inside(grid_points.cpu().numpy())
        valid_points = grid_points[inside_mask]
        _ = self.model.predict(valid_points.cpu().numpy())

    def export_regions(self, epoch, date):
        name = f"{date}-epoch{epoch}"
        self.evaluate_regions()
        # The last one does not get any activation
        self.output_storage.pop()
        activation_regions = self.compute_region(name)   

        regions = Regions(
            activation_regions,
            self.resolution,
            dim=self.dim
        )
        regions.export(name)
        
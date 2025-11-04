from interface.regions import Regions

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering
from scipy.sparse import coo_matrix

import numpy as np
import torch
import hdbscan
import scipy.sparse as sp


class ActivationRegionStrategy:

    def __init__(self, model, register_ready):
        self.model = model
        self.register_ready = register_ready
        self.activation_storage = []
        self.input_storage = []
        self.map_regions_id = {}                
        self.nb_regions = 0
        self.resolution = 200
        self.init_strategy()        

    def init_strategy(self):
        self.activations_output = {
            "tanh": self.tanh_output,
            "relu": self.relu_output
        }
        self.activations_pattern = {
            "tanh": self.tanh_pattern,
            "relu": self.relu_pattern
        }

    def get_activation_output(self):
        return self.activations_output[self.activation_name.lower()]
    
    def get_activation_pattern(self):
        return self.activations_pattern[self.activation_name.lower()]()
    
    def get_activation_hook(self):
        activation_output = self.get_activation_output()
        def get_hook(module, input, output):
            if self.register_ready():
                self.activation_storage.append(activation_output(output).cpu())
        return get_hook
    
    def get_input(self):
        def get_hook(module, input):
            if self.register_ready():
                self.input_storage.append(input[0].cpu())
        return get_hook

    def tanh_output(self, output):
        return torch.tanh(output)

    def relu_output(self, output):
        return (output > 0).int()   

    def register_hook(self):
        self.model.net.register_forward_pre_hook(self.get_input())
        self.activation_name = self.model.net.activation.__name__
        get_hook = self.get_activation_hook()
        for module in self.model.net.modules():   
            if isinstance(module, torch.nn.modules.linear.Linear):
                module.register_forward_hook(get_hook)

    def tanh_pattern(self):
        tanh_output = torch.cat(self.activation_storage, dim=1)
        tanh_output_slope = 1 - tanh_output * tanh_output
        activation_point = torch.stack((tanh_output, tanh_output_slope), dim=2)
        activation_point = activation_point.detach()

        num_inputs = tanh_output.shape[0]
        inputs = self.input_storage[0].detach().cpu().numpy()

        k = 8
        nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(inputs)
        _, neighbors = nn.kneighbors(inputs)

        def my_distance(a, b):
            Ai = activation_point[i]               
            Aj = activation_point[j]           
            diff = Aj - Ai
            squared_distance = torch.sqrt(torch.sum(diff * diff, dim=1)) 
            return torch.sum(squared_distance).item()
        
        rows, cols, vals = [], [], []
        for i in range(num_inputs):
            for j in neighbors[i][1:]:
                d = my_distance(i, j)
                rows.append(i)
                cols.append(j)
                vals.append(d)

        rows = np.array(rows, dtype=int)
        cols = np.array(cols, dtype=int)
        vals = np.array(vals, dtype=float)

        clusterer = hdbscan.HDBSCAN(
            metric='precomputed',
            min_samples=1,
            min_cluster_size=5,
            approx_min_span_tree=True,
            core_dist_n_jobs=-1     # multithread
        )

        dist_sparse = coo_matrix((vals, (rows, cols)), shape=(num_inputs, num_inputs))        
        # Sym but don't know if it is useful :
        dist_sparse = dist_sparse.minimum(dist_sparse.transpose())
        dist_sparse = dist_sparse.tocsr()
        clusterer.fit(dist_sparse)

        map_region = {}
        for i in range(num_inputs):
            input_point = self.input_storage[0][i].tolist()
            id_region = clusterer.labels_[i].item()
            if id_region in map_region:
                map_region[id_region].append(input_point)
            else:
                map_region[id_region] = [input_point]
        return map_region

    def relu_pattern(self):
        pattern = torch.cat(self.activation_storage, dim=1) 
        map_region = {}
        for i in range(len(pattern)):
            input_point = self.input_storage[0][i].tolist()
            region = tuple(pattern[i].tolist())
            if region not in self.map_regions_id:
                self.map_regions_id[region] = self.nb_regions
                self.nb_regions += 1
            id_region = self.map_regions_id[region]
            if id_region in map_region:                
                map_region[id_region].append(input_point)
            else:                
                map_region[id_region] = [input_point]
        return map_region

    def evaluate_regions(self):
        self.activation_storage.clear()
        self.input_storage.clear()
        dim = len(self.model.pde.bbox) // 2      
        if dim == 2:
            x_range = torch.linspace(self.model.pde.bbox[0], self.model.pde.bbox[1], self.resolution)
            y_range = torch.linspace(self.model.pde.bbox[2], self.model.pde.bbox[3], self.resolution)
            xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
            grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        elif dim == 3:
            self.resolution = 10  # Reduce resolution for 3D to limit memory usage
            x_range = torch.linspace(self.model.pde.bbox[0], self.model.pde.bbox[1], self.resolution)
            y_range = torch.linspace(self.model.pde.bbox[2], self.model.pde.bbox[3], self.resolution)
            z_range = torch.linspace(self.model.pde.bbox[4], self.model.pde.bbox[5], self.resolution)
            xx, yy, zz = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
            grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
        if dim > self.model.pde.geom.dim: # Time 
            inside_mask = self.model.pde.geom.inside(grid_points[:, :-1].cpu().numpy())
        else:
            inside_mask = self.model.pde.geom.inside(grid_points.cpu().numpy())
        valid_points = grid_points[inside_mask]
        _ = self.model.predict(valid_points.cpu().numpy())

    def export_regions(self, epoch, date):
        self.evaluate_regions()
        self.activation_storage.pop() # The last one does not get any activation
        activation_pattern = self.get_activation_pattern()   

        geom = self.model.pde.geom
        if geom.dim == 2:
            regions = Regions(
                activation_pattern,
                (geom.center[0], geom.center[1]),
                geom.radius,
                self.resolution,
                dim=2
            )
        elif geom.dim == 3:
            regions = Regions(
                activation_pattern,
                (geom.center[0], geom.center[1], geom.center[2]),
                geom.radius,
                self.resolution,
                dim=3
            )
        regions.export(f"{date}-epoch{epoch}")
        
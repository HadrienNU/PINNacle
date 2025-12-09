from interface.regions import Regions
from interface.relu import ReLU
from interface.silu import SiLU
from interface.tanh import Tanh

from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.cluster import AgglomerativeClustering

from deepxde import backend as bkd

import torch


class ActivationRegionStrategy:

    def __init__(self, model, register_ready, resolution, slice_resolution):
        self.model = model
        self.register_ready = register_ready
        self.output_storage = []
        self.input_storage = []
        self.map_regions_id = {}                
        self.nb_regions = 0
        self.dim = self.model.pde.geom.dim
        self.pdetime = len(model.pde.bbox) // 2 > self.dim
        self.resolution = resolution if self.dim == 2 else resolution / 5
        self.slice_resolution = slice_resolution if self.pdetime else self.resolution
        self.pdistance_threshold = 0.5
        self.init_strategy()        
        # self.model.net.activation = bkd.tanh

    def init_strategy(self):
        self.activations_strategy = {
            "tanh": Tanh,
            "relu": ReLU,
            "silu": SiLU
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

    def compute_region(self, inputs, outputs, name):
        num_inputs = inputs.shape[0]
        index = {tuple(inputs[i].tolist()) : i for i in range(num_inputs)}
        res = self.get_strategy(outputs, index).compute_region(inputs) 
        if res is not None:
            return res

        k = 8
        strategy = self.get_strategy(outputs, index)
        connectivity = kneighbors_graph(inputs, n_neighbors=k, include_self=False)
        for i in range(num_inputs):
            neighbors = connectivity[i].indices
            for j in neighbors:
                strategy.distance_custom(inputs[i], inputs[j])

        connectivity = 0.5 * (connectivity + connectivity.T)
        distance_threshold = strategy.get_distance(self.pdistance_threshold)
        print("distance_threshold :", distance_threshold)
        clusterer = AgglomerativeClustering(
            metric=strategy.distance_custom,
            n_clusters=None,
            distance_threshold=distance_threshold,
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

        # strategy.export_distances(name)
        return map_region

    def evaluate_regions(self):
        self.output_storage.clear()
        self.input_storage.clear()        
        x_range = torch.linspace(self.model.pde.bbox[0], self.model.pde.bbox[1], self.resolution)
        y_range = torch.linspace(self.model.pde.bbox[2], self.model.pde.bbox[3], self.resolution)        
        if self.dim == 3 or self.dim == 2 and self.pdetime:
            z_range = torch.linspace(self.model.pde.bbox[4], self.model.pde.bbox[5], self.slice_resolution)
            xx, yy, zz = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
            grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
        elif self.dim == 2 or self.dim == 1 and self.pdetime:            
            xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
            grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        if self.pdetime: # Time 
            inside_mask = self.model.pde.geom.inside(grid_points[:, :-1].cpu().numpy())
        else:
            inside_mask = self.model.pde.geom.inside(grid_points.cpu().numpy())
        valid_points = grid_points[inside_mask]
        _ = self.model.predict(valid_points.cpu().numpy())

    def compute_region_time(self, inputs, outputs, name):
        t_column = self.input_storage[0][:, -1] 
        t_vals = torch.unique(t_column)
        activation_regions = {}
        n = 0
        for t in t_vals:                
            t_mask = (t_column == t) 
            region = self.compute_region(
                inputs[t_mask], 
                outputs[t_mask], 
                f"{name}_t{outputs[t_mask][0][-1]}"
            )
            for _, vals in region.items():
                n += 1
                activation_regions[n] = vals.copy()
        return activation_regions

    def export_regions(self, epoch, date):
        name = f"{date}-epoch{epoch}"
        self.evaluate_regions()
        # The last one does not get any activation
        self.output_storage.pop()
        outputs = torch.cat(self.output_storage, dim=1)     
        inputs = self.input_storage[0].detach().cpu().numpy()        
        if self.pdetime: # Slices in time
            activation_regions = self.compute_region_time(inputs, outputs, name)                      
        else:
            activation_regions = self.compute_region(inputs, outputs, name)   

        regions = Regions(
            activation_regions,
            self.resolution,
            dim=self.dim
        )
        regions.export(name)
        
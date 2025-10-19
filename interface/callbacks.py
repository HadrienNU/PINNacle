from deepxde.callbacks import Callback
from interface.regions import Regions

import torch


class InterfaceCallback(Callback):

    def __init__(self, log_every=None):
        super(InterfaceCallback, self).__init__()
        self.log_every = log_every
        self.resolution = 500
        self.epoch = 0
        self.activation_storage = []
        self.input_storage = []
        self.map_regions_id = {}
        self.nb_regions = 0

    def evaluate_regions(self):
        self.activation_storage.clear()
        self.input_storage.clear()
        x_range = torch.linspace(self.model.pde.bbox[0], self.model.pde.bbox[1], self.resolution)
        y_range = torch.linspace(self.model.pde.bbox[2], self.model.pde.bbox[3], self.resolution)
        xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
        grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        inside_mask = self.model.pde.geom.inside(grid_points.cpu().numpy())
        valid_points = grid_points[inside_mask]
        _ = self.model.predict(valid_points.cpu().numpy())

    def register_ready(self):
        return self.epoch % self.log_every == 0

    def get_input(self):
        def get_hook(module, input):
            if self.register_ready():
                self.input_storage.append(input[0].cpu())
        return get_hook
    
    def get_activation_hook(self, activation_name):
        activations_output = {
            "tanh": self.tanh_output,
            "relu": self.relu_output
        }

        activation_output = activations_output[activation_name.lower()]
        def get_hook(module, input, output):
            if self.register_ready():
                self.activation_storage.append(activation_output(output).cpu())
        return get_hook

    def tanh_output(self, output):
        result = torch.ones_like(output, dtype=torch.int)
        result[output < -0.05] = 0
        result[output > 0.05] = 2
        return result

    def relu_output(self, output):
        return (output > 0).int()        

    def on_epoch_begin(self):
        """Called at the beginning of every epoch."""

    def on_epoch_end(self):
        """Called at the end of every epoch."""
        self.epoch += 1
        if not self.register_ready():   
            return 

        self.evaluate_regions()
        activation_pattern = torch.cat(self.activation_storage, dim=1) 
        map_region = {}

        for i in range(len(activation_pattern)):
            input_point = self.input_storage[0][i].tolist()
            region = tuple(activation_pattern[i].tolist())
            if region not in self.map_regions_id:
                self.map_regions_id[region] = self.nb_regions
                self.nb_regions += 1
            id_region = self.map_regions_id[region]
            if id_region in map_region:                
                map_region[id_region].append(input_point)
            else:                
                map_region[id_region] = [input_point]                

        geom = self.model.pde.geom
        regions = Regions(
            map_region,
            (geom.center[0], geom.center[1]),
            geom.radius,
            self.resolution
        )
        regions.export(f"epoch{self.epoch}")

    def on_batch_begin(self):
        """Called at the beginning of every batch."""
        pass

    def on_batch_end(self):
        """Called at the end of every batch."""
        pass

    def on_train_begin(self):
        if self.log_every is None:
            self.log_every = self.model.display_every

        self.model.net.register_forward_pre_hook(self.get_input())
        
        activation_name = self.model.net.activation.__name__
        get_hook = self.get_activation_hook(activation_name)
        for module in self.model.net.modules():   
            if isinstance(module, torch.nn.modules.linear.Linear):
                module.register_forward_hook(get_hook)

    def on_train_end(self):
        """Called at the end of model training."""
        pass

    def on_predict_begin(self):
        """Called at the beginning of prediction."""
        pass

    def on_predict_end(self):
        """Called at the end of prediction."""
        pass


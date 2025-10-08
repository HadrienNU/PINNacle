from deepxde.callbacks import Callback
import torch

import torch


class InterfaceCallback(Callback):

    def __init__(self, log_every=None):
        super(InterfaceCallback, self).__init__()
        self.log_every = log_every
        self.epoch = 0
        self.activation_storage = []
        self.map_region_count = {}
        self.input_storage = []

    def register_ready(self):
        return self.epoch % self.log_every == 0

    def get_input(self):
        def get_hook(module, input):
            if self.register_ready():
                self.input_storage.append(input[0].cpu())
        return get_hook

    def relu_output(self, output):
        return (output > 0).int()
    
    def get_activation_hook(self, activation_name):
        activations_output = {
            "tanh": self.relu_output, # Need to be changed
            "relu": self.relu_output
        }

        activation_output = activations_output[activation_name.lower()]
        def get_hook(module, input, output):
            if self.register_ready():
                self.activation_storage.append(activation_output(output).cpu())
        return get_hook
        

    def on_epoch_begin(self):
        """Called at the beginning of every epoch."""
        if self.register_ready():
            self.activation_storage.clear()
            self.map_region_count.clear()
            self.input_storage.clear()

    def on_epoch_end(self):
        """Called at the end of every epoch."""
        self.epoch += 1
        if self.register_ready():    
            activation_pattern = torch.cat(self.activation_storage, dim=1) 
            for point_activation in activation_pattern:
                point_activation = tuple(point_activation.tolist())
                if point_activation in self.map_region_count:
                    self.map_region_count[point_activation]+=1
                else:
                    self.map_region_count[point_activation] = 1
            print(len(self.activation_storage))
            print(self.activation_storage)
            print(f"Number of unique region : {len(self.map_region_count)}")
            print(f"len input_storage: {len(self.input_storage)}")
            print(f"Input shape : {self.input_storage[0].shape}")

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

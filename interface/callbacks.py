from deepxde.callbacks import Callback
import torch

class InterfaceCallback(Callback):

    def __init__(self, log_every=None):
        super(InterfaceCallback, self).__init__()
        self.log_every = log_every
        self.epoch = 0
        self.activation_storage = []
        self.relu_hooks = []

    def register_ready(self):
        return (self.epoch + 1) % self.log_every == 0

    def get_relu_hook(self):
        def hook(module, input, output):
            if self.register_ready():
                # Store binary ReLU activation pattern (1 = active, 0 = inactive)
                self.activation_storage.append((output > 0).int().cpu())
        return hook

    def on_epoch_begin(self):
        """Called at the beginning of every epoch."""
        self.epoch += 1
        if self.register_ready():
            self.activation_storage.clear()

    def on_epoch_end(self):
        """Called at the end of every epoch."""
        if self.register_ready():           
            print(len(self.activation_storage))
            print(self.activation_storage)
            print()

    def on_batch_begin(self):
        """Called at the beginning of every batch."""
        pass

    def on_batch_end(self):
        """Called at the end of every batch."""
        pass

    def on_train_begin(self):
        if self.log_every is None:
            self.log_every = self.model.display_every

        for module in self.model.net.modules():
            if isinstance(module, torch.nn.Linear):
                print(module)
                self.relu_hooks.append(
                    module.register_forward_hook(self.get_relu_hook())
                )

    def on_train_end(self):
        """Called at the end of model training."""
        pass

    def on_predict_begin(self):
        """Called at the beginning of prediction."""
        pass

    def on_predict_end(self):
        """Called at the end of prediction."""
        pass

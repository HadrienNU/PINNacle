from deepxde.callbacks import Callback
from interface.activation import ActivationRegionStrategy


class InterfaceCallback(Callback):

    def __init__(self, date, log_every=None):
        super(InterfaceCallback, self).__init__()
        self.log_every = log_every        
        self.date = date     
        self.epoch = 0   

    def register_ready(self):
        return self.epoch % self.log_every == 0     

    def on_epoch_begin(self):
        """Called at the beginning of every epoch."""

    def on_epoch_end(self):
        """Called at the end of every epoch."""
        self.epoch += 1
        if not self.register_ready():   
            return 
        self.activation_region.export_regions(self.epoch, self.date)        

    def on_batch_begin(self):
        """Called at the beginning of every batch."""
        pass

    def on_batch_end(self):
        """Called at the end of every batch."""
        pass

    def on_train_begin(self):
        if self.log_every is None:
            self.log_every = self.model.display_every
        self.activation_region = ActivationRegionStrategy(self.model, self.register_ready)
        self.activation_region.register_hook()

    def on_train_end(self):
        """Called at the end of model training."""
        pass

    def on_predict_begin(self):
        """Called at the beginning of prediction."""
        pass

    def on_predict_end(self):
        """Called at the end of prediction."""
        pass


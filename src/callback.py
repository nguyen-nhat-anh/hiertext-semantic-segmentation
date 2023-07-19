import numpy as np
from copy import deepcopy

class Callbacks:
    def __init__(self, callbacks):
        self.callbacks = callbacks
        
    def set_model_module(self, model_module):
        for callback in self.callbacks:
            callback.set_model_module(model_module)
            
    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()
            
    def on_epoch_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(**kwargs)
            
    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()
            
class EarlyStoppingCallback:
    def __init__(self, patience=1, monitor='val_loss', mode='min'):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.is_improvement = np.less if self.mode=='min' else np.greater
        
        self.model_module = None
        self.best = None
        self.best_epoch = None
        self.best_state_dict = None
        self.wait = None
        self.stopped_epoch = None
        
    def set_model_module(self, model_module):
        self.model_module = model_module
        
    def on_train_begin(self):
        self.best = np.inf if self.mode=='min' else -np.inf
        self.best_epoch = 0
        self.best_state_dict = None
        self.wait = 0
        
    def on_epoch_end(self, epoch, logs, **kwargs):
        current = logs.get(self.monitor)
        self.wait += 1
        if self.is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_state_dict = deepcopy(self.model_module.model.state_dict())
            self.wait = 0
        
        # Check early stopping condition (only check after the first epoch)
        if self.wait >= self.patience and epoch > 1:
            self.stopped_epoch = epoch
            self.model_module.stop_training = True
            print('Restore best weight with {} = {:.2f} at epoch {}'.format(self.monitor,
                                                                            self.best,
                                                                            self.best_epoch))
            self.model_module.model.load_state_dict(self.best_state_dict)
    
    def on_train_end(self):
        print('Epoch {}: early stopping'.format(self.stopped_epoch))
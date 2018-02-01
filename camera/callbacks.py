import numpy as np
from keras.callbacks import Callback
import keras.backend as K
from camera.networks import unfreeze_all_layers

class UnfreezeAfterEpoch(Callback):
    def __init__(self, epoch):
        super(UnfreezeAfterEpoch, self).__init__()
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs={}):
        if epoch == self.epoch:
            unfreeze_all_layers(self.model)
            self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=self.model.metrics)

class SGDWarmRestart(Callback):
    def __init__(self, steps_per_epoch, min_lr=1e-5, max_lr=0.05, period_in_epochs=10.0, period_growth_rate=2.0):
        super(SGDWarmRestart, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period_in_epochs = period_in_epochs
        self.period_growth_rate = period_growth_rate
        self.steps_per_epoch = steps_per_epoch
        self.progress = 0

    def on_train_begin(self, logs={}):
        self.progress = 0

    def on_batch_begin(self, batch, logs={}):
        new_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * self.progress / self.period_in_epochs))
        K.set_value(self.model.optimizer.lr, new_lr)
        self.progress += 1 / self.steps_per_epoch
        if self.progress >= self.period_in_epochs:
            self.progress = 0
            self.period_in_epochs = self.period_growth_rate * self.period_in_epochs

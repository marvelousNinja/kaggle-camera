"""Custom callbacks for Keras models"""
import numpy as np
import keras.backend as K
from keras.callbacks import Callback

from camera.networks import unfreeze_all_layers

class Unfreeze(Callback):
    """Unfreezes all model weights at the beinning of specified epoch"""
    def __init__(self, epoch, verbose=0):
        super().__init__()
        self.epoch = epoch
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.epoch:
            unfreeze_all_layers(self.model)
            self.model.compile(
                optimizer=self.model.optimizer,
                loss=self.model.loss,
                metrics=self.model.metrics,
                weighted_metrics=self.model.weighted_metrics
            )

            if self.verbose > 0:
                print(f'Epoch {epoch + 1}: Unfreeze is unfreezing layers')

class WarmRestartSGD(Callback):
    """Gradually decreases learning rate for each *batch* during epoch.
    After specific period measured in epochs, learning rate is reset to the initial value.

    SGDR: Stochastic Gradient Descent with Warm Restarts
    https://arxiv.org/abs/1608.03983
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(
            self, epoch, steps_per_epoch, min_lr=1e-6, max_lr=0.05,
            period_in_epochs=10.0, period_growth_rate=2.0, verbose=0
        ): # pylint: disable=too-many-arguments

        super().__init__()
        self.epoch = epoch
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period_in_epochs = period_in_epochs
        self.period_growth_rate = period_growth_rate
        self.steps_per_epoch = steps_per_epoch
        self.progress = 0
        self.verbose = verbose
        self.active = False

    def on_train_begin(self, logs=None):
        self.progress = 0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.epoch:
            self.active = True

            if self.verbose > 0:
                print(f'Epoch {epoch + 1}: WarmRestartSGD is activating')

    def on_batch_begin(self, batch, logs=None):
        if not self.active:
            return

        new_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * self.progress / self.period_in_epochs))
        K.set_value(self.model.optimizer.lr, new_lr)

        if self.verbose > 1:
            print(f'WarmRestartSGD is setting learning rate to {new_lr}')

        self.progress += 1 / self.steps_per_epoch
        if self.progress >= self.period_in_epochs:
            if self.verbose > 0:
                print(f'Period of {self.period_in_epochs} ended: WarmRestartSGD is restarting the optimizer')
            self.progress = 0
            self.period_in_epochs = self.period_growth_rate * self.period_in_epochs

class SwitchOptimizer(Callback):
    """Switches optimizer during training at the beginning of specified epoch.
    Improving Generalization Performance by Switching from Adam to SGD
    https://arxiv.org/abs/1712.07628
    """
    def __init__(self, epoch, optimizer, verbose=0):
        super().__init__()
        self.epoch = epoch
        self.optimizer = optimizer
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.epoch:
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.model.loss,
                metrics=self.model.metrics,
                weighted_metrics=self.model.weighted_metrics
            )

            if self.verbose > 0:
                print(f'Epoch {epoch + 1}: SwitchOptimizer switches optimizer to {self.optimizer}')

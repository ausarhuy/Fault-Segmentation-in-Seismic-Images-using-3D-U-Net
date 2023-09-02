import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau


class EarlyStoppingReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        self.restore_best_weights = kwargs.pop('restore_best_weights', False)
        self.early_stopping = kwargs.pop('early_stopping', False)
        super().__init__(*args, **kwargs)
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Learning rate reduction is conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.best_epoch = epoch+1
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()

            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                    min_lr = np.float32(self.min_lr)
                    if self.early_stopping and np.allclose(old_lr, min_lr, rtol=0, atol=min_lr):
                        self.model.stop_training = True
                        print(f"\nEpoch {epoch + 1}: early stopping")
                    else:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                        if self.restore_best_weights and self.best_weights is not None:
                            print('\nRestoring model weights from the end of the best epoch.')
                            self.model.set_weights(self.best_weights)
                        if self.verbose > 0:
                            print('Epoch %d: Reducing learning rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def on_train_end(self, logs=None):
        logs = logs or {}
        if self.restore_best_weights and self.best_weights is not None:
            current = logs.get(self.monitor)
            if self.monitor_op(self.best, current):
                print(f'Restoring model weights from the end of the best epoch: {self.best_epoch} at end of training.')
                self.model.set_weights(self.best_weights)

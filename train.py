import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from dataloader import DataGenerator
from utils import show_history
from model import unet

np.random.seed(12345)
tf.random.set_seed(12345)


def train():
    # input image dimensions
    params = {'batch_size': 1,
              'dim': (128, 128, 128),
              'n_channels': 1,
              'shuffle': True}
    train_path = "./data/train/fault/"
    val_path = "./data/validation/seis/"

    train_ids = range(len(os.listdir(train_path)))
    val_ids = range(len(os.listdir(val_path)))
    train_generator = DataGenerator(path=train_path,
                                    ids=train_ids, **params)
    valid_generator = DataGenerator(path=val_path,
                                    ids=val_ids, **params)
    model = unet(input_size=(None, None, None, 1))
    model.compile(optimizer=Adam(lr=1e-4), loss='bce',
                  metrics=['accuracy'])
    model.summary()

    # checkpoint
    filepath = "model/model-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                                 verbose=1, save_best_only=False, mode='max')
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                              patience=20, min_lr=1e-8)
    callbacks_list = [checkpoint]
    print("data prepared, ready to train!")
    # Fit the model
    history = model.fit(generator=train_generator,
                        validation_data=valid_generator, epochs=100, callbacks=callbacks_list, verbose=1)
    model.save('model/model.h5')
    show_history(history)


if __name__ == 'main':
    train()

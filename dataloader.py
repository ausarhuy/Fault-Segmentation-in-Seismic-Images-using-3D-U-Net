import numpy as np
from tensorflow.keras.utils import Sequence
from utils import data_norm


def load_sample(file_path, shape=(128, 128, 128), dtype=np.single, norm=True):
    sample = np.fromfile(file_path, dtype=dtype).reshape(shape)
    if norm:
        sample = data_norm(sample)
    # In seismic processing, the dimensions of a seismic array is often arranged as
    # a[n3][n2][n1] where n1 represents the vertical dimension.
    return np.transpose(sample)


class DataGenerator(Sequence):
    """Generates data for keras"""

    def __init__(self, path, ids, batch_size=1, shape=(128, 128, 128),
                 n_channels=1, shuffle=True):
        """Initialization"""
        self.shape = shape
        self.path = path
        self.batch_size = batch_size
        self.ids = ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data"""
        # Generate indexes of the batch
        bsize = self.batch_size
        indexes = self.indexes[index * bsize:(index + 1) * bsize]

        # Find list of IDs
        ids = [self.ids[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(ids)

        return X, Y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids):
        """Generates data containing batch_size samples"""
        # Initialization
        seis = load_sample(f'{self.path}/seis/{ids[0]}.dat', norm=True)
        fault = load_sample(f'{self.path}/fault/{ids[0]}.dat', norm=False)
        seis = np.reshape(seis, self.shape)

        # Generate data
        X = np.zeros((2, *self.shape, self.n_channels), dtype=np.single)
        Y = np.zeros((2, *self.shape, self.n_channels), dtype=np.single)
        X[0,] = np.reshape(seis, (*self.shape, self.n_channels))
        Y[0,] = np.reshape(fault, (*self.shape, self.n_channels))
        X[1,] = np.reshape(np.flipud(seis), (*self.shape, self.n_channels))
        Y[1,] = np.reshape(np.flipud(fault), (*self.shape, self.n_channels))
        '''
        for i in range(4):
          X[i,] = np.reshape(np.rot90(seis,i,(2,1)), (*self.shape,self.n_channels))
          Y[i,] = np.reshape(np.rot90(fault,i,(2,1)), (*self.shape,self.n_channels))  
        '''
        return X, Y

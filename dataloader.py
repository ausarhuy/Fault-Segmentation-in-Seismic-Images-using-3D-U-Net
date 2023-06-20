import os
import numpy as np
import tensorflow as tf


def load_sample(file_path, shape=(128, 128, 128), dtype=np.single):
    return np.fromfile(file_path, dtype=dtype).reshape(shape).transpose((2, 1, 0))


def load_data(data_dir, shape=(128, 128, 128), batch_size=None):
    seis_data = []
    fault_data = []

    # Load input seis data
    seis_dir = os.path.join(data_dir, 'seis')
    for filename in os.listdir(seis_dir):
        file_path = os.path.join(seis_dir, filename)
        sample = load_sample(file_path, shape)
        seis_data.append(sample)

    # Load output fault data
    fault_dir = os.path.join(data_dir, 'fault')
    for filename in os.listdir(fault_dir):
        file_path = os.path.join(fault_dir, filename)
        sample = load_sample(file_path, shape)
        fault_data.append(sample)

    seis_data = tf.data.Dataset.from_tensor_slices(np.array(seis_data))
    fault_data = tf.data.Dataset.from_tensor_slices(np.array(fault_data))
    if batch_size:
        return tf.data.Dataset.zip((seis_data, fault_data)).batch(batch_size=batch_size)
    return tf.data.Dataset.zip((seis_data, fault_data))

